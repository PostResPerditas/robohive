import argparse
import json
import os
from pathlib import Path

import gymnasium
import numpy as np
import robohive
from robohive.utils import gym as rhgym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "train_relocate_sac_parallel.json"


class RoboHiveSB3Compat(gymnasium.Wrapper):
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
                name_prefix=config.get("model_name", "sac_relocate_parallel"),
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


def train(config):
    run_dir = resolve_project_path(config.get("run_dir", "runs/relocate_sac_parallel"))
    os.makedirs(run_dir, exist_ok=True)

    n_envs = int(config.get("n_envs", 1))
    env = build_train_env(config, run_dir)
    callback = build_callbacks(config, run_dir, n_envs)

    sac_kwargs = dict(config.get("sac", {}))
    model = SAC(
        config.get("policy", "MlpPolicy"),
        env,
        verbose=int(config.get("verbose", 1)),
        device=config.get("device", "cuda"),
        tensorboard_log=str(run_dir / "tb"),
        **sac_kwargs,
    )

    print(f"run_dir={run_dir}")
    print(f"n_envs={n_envs}")
    print(f"device={config.get('device', 'cuda')}")
    print(f"total_timesteps={int(config.get('total_timesteps', 100000))}")
    print(f"checkpoint_freq={int(config.get('checkpoint_freq', 0))}")
    print(f"eval_freq={int(config.get('eval_freq', 0))}")
    print(f"sac={sac_kwargs}")

    model.learn(
        total_timesteps=int(config.get("total_timesteps", 100_000)),
        callback=callback,
        progress_bar=bool(config.get("progress_bar", False)),
    )

    model_path = run_dir / config.get("model_name", "sac_relocate_parallel")
    model.save(str(model_path))
    print(f"saved_model={model_path}.zip")

    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the JSON parallel training config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    print(f"config={args.config}")
    train(config)


if __name__ == "__main__":
    main()
