import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
import robohive
from robohive.utils import gym as rhgym
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "relocate" / "train_relocate_ppo_parallel.json"


class RoboHiveSB3Compat(gymnasium.Wrapper):
    """Convert RoboHive/Gym-style outputs to Gymnasium outputs for SB3 2.x."""

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.env.unwrapped.seed(seed)

        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated, truncated = done, False

        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_single_env(env_id, seed):
    raw_env = rhgym.make(env_id, seed=seed, disable_env_checker=True)
    return RoboHiveSB3Compat(raw_env)


def make_env_fn(config, rank):
    env_id = config["env_id"]
    seed = int(config.get("seed", 0)) + rank

    def _init():
        return make_single_env(env_id, seed)

    return _init


def build_train_env(config, run_dir):
    n_envs = int(config.get("n_envs", 1))
    vec_env = config.get("vec_env", "subproc").lower()
    env_fns = [make_env_fn(config, rank) for rank in range(n_envs)]

    if n_envs == 1 or vec_env == "dummy":
        env = DummyVecEnv(env_fns)
    elif vec_env == "subproc":
        env = SubprocVecEnv(env_fns, start_method=config.get("start_method", "forkserver"))
    else:
        raise ValueError(f"Unsupported vec_env: {vec_env}")

    monitor_path = run_dir / config.get("monitor_filename", "train_monitor.csv")
    return VecMonitor(
        env,
        filename=str(monitor_path),
        info_keywords=tuple(config.get("info_keywords", ["solved"])),
    )


def build_eval_env(config):
    eval_seed = int(config.get("seed", 0)) + 100000
    return VecMonitor(
        DummyVecEnv([lambda: make_single_env(config["env_id"], eval_seed)]),
        info_keywords=tuple(config.get("info_keywords", ["solved"])),
    )


def scale_freq(freq, n_envs):
    if freq is None or int(freq) <= 0:
        return None
    return max(int(freq) // max(n_envs, 1), 1)


def build_callbacks(config, run_dir, n_envs):
    callbacks = []

    checkpoint_freq = scale_freq(config.get("checkpoint_freq", 0), n_envs)
    if checkpoint_freq:
        checkpoint_dir = run_dir / "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix=config.get("model_name", "rl_relocate_parallel"),
                save_replay_buffer=bool(config.get("save_replay_buffer", False)),
                save_vecnormalize=bool(config.get("save_vecnormalize", False)),
            )
        )

    eval_freq = scale_freq(config.get("eval_freq", 0), n_envs)
    if eval_freq:
        eval_env = build_eval_env(config)
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best_model"),
                log_path=str(run_dir / "eval"),
                eval_freq=eval_freq,
                n_eval_episodes=int(config.get("eval_episodes", 20)),
                deterministic=bool(config.get("deterministic_eval", True)),
                render=False,
            )
        )

    if not callbacks:
        return None
    return CallbackList(callbacks)


def get_algorithm_class(algorithm: str):
    algo = algorithm.upper()
    classes = {
        "SAC": SAC,
        "PPO": PPO,
        "TD3": TD3,
        "DDPG": DDPG,
        "A2C": A2C,
    }
    if algo not in classes:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. Supported: {sorted(classes.keys())}"
        )
    return algo, classes[algo]


def get_algorithm_kwargs(config: Dict[str, Any], algorithm: str) -> Dict[str, Any]:
    """
    Read algorithm-specific kwargs.

    Supported config styles:
      1) {"algorithm": "PPO", "algo_kwargs": {...}}
      2) {"algorithm": "PPO", "ppo": {...}}
      3) legacy SAC style: {"sac": {...}}
    """
    algo_key = algorithm.lower()
    kwargs = dict(config.get("algo_kwargs", {}))
    kwargs.update(dict(config.get(algo_key, {})))
    return kwargs


def build_action_noise(config: Dict[str, Any], env) -> Optional[Any]:
    """Build action noise for TD3/DDPG if requested."""
    noise_cfg = config.get("action_noise", None)
    if not noise_cfg:
        return None

    noise_type = str(noise_cfg.get("type", "normal")).lower()
    sigma = float(noise_cfg.get("sigma", 0.1))
    mean = float(noise_cfg.get("mean", 0.0))

    action_dim = int(np.prod(env.action_space.shape))
    mean_vec = np.full(action_dim, mean, dtype=np.float32)
    sigma_vec = np.full(action_dim, sigma, dtype=np.float32)

    if noise_type in {"normal", "normal_action_noise"}:
        return NormalActionNoise(mean=mean_vec, sigma=sigma_vec)
    if noise_type in {"ornstein", "ou", "ornstein_uhlenbeck"}:
        return OrnsteinUhlenbeckActionNoise(mean=mean_vec, sigma=sigma_vec)

    raise ValueError(f"Unsupported action_noise.type: {noise_type}")


def add_algorithm_dependent_kwargs(algorithm: str, kwargs: Dict[str, Any], config, env):
    """Inject optional kwargs that depend on algorithm family."""
    if algorithm in {"TD3", "DDPG"} and "action_noise" not in kwargs:
        action_noise = build_action_noise(config, env)
        if action_noise is not None:
            kwargs["action_noise"] = action_noise
    return kwargs


def train(config):
    algorithm, model_cls = get_algorithm_class(config.get("algorithm", "SAC"))

    default_run_dir = f"runs/relocate_{algorithm.lower()}_parallel"
    run_dir = resolve_project_path(config.get("run_dir", default_run_dir))
    os.makedirs(run_dir, exist_ok=True)

    n_envs = int(config.get("n_envs", 1))
    env = build_train_env(config, run_dir)
    callback = build_callbacks(config, run_dir, n_envs)

    algo_kwargs = get_algorithm_kwargs(config, algorithm)
    algo_kwargs = add_algorithm_dependent_kwargs(algorithm, algo_kwargs, config, env)

    model = model_cls(
        config.get("policy", "MlpPolicy"),
        env,
        verbose=int(config.get("verbose", 1)),
        device=config.get("device", "cuda"),
        tensorboard_log=str(run_dir / "tb"),
        **algo_kwargs,
    )

    print(f"algorithm={algorithm}")
    print(f"run_dir={run_dir}")
    print(f"n_envs={n_envs}")
    print(f"vec_env={config.get('vec_env', 'subproc')}")
    print(f"device={config.get('device', 'cuda')}")
    print(f"total_timesteps={int(config.get('total_timesteps', 100000))}")
    print(f"checkpoint_freq={int(config.get('checkpoint_freq', 0))}")
    print(f"eval_freq={int(config.get('eval_freq', 0))}")
    print(f"algo_kwargs={algo_kwargs}")

    learn_kwargs = {}
    if "log_interval" in config:
        learn_kwargs["log_interval"] = int(config["log_interval"])
    if "tb_log_name" in config:
        learn_kwargs["tb_log_name"] = str(config["tb_log_name"])

    model.learn(
        total_timesteps=int(config.get("total_timesteps", 100_000)),
        callback=callback,
        progress_bar=bool(config.get("progress_bar", False)),
        **learn_kwargs,
    )

    model_path = run_dir / config.get("model_name", f"{algorithm.lower()}_relocate_parallel")
    model.save(str(model_path))
    print(f"saved_model={model_path}.zip")

    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the JSON training config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    print(f"config={args.config}")
    train(config)


if __name__ == "__main__":
    main()
