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
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "train_relocate_sac.json"


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


def build_env(config):
    raw_env = rhgym.make(
        config["env_id"],
        seed=config.get("seed", 0),
        disable_env_checker=True,
    )
    return RoboHiveSB3Compat(raw_env)


def build_eval_env(config):
    eval_config = dict(config)
    eval_config["seed"] = int(config.get("seed", 0)) + 100000
    return Monitor(
        build_env(eval_config),
        info_keywords=tuple(config.get("info_keywords", ["solved"])),
    )


def build_callbacks(config, run_dir):
    callbacks = []

    checkpoint_freq = int(config.get("checkpoint_freq", 0))
    if checkpoint_freq > 0:
        checkpoint_dir = run_dir / "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix=config.get("model_name", "sac_relocate"),
                save_replay_buffer=bool(config.get("save_replay_buffer", False)),
                save_vecnormalize=bool(config.get("save_vecnormalize", False)),
            )
        )

    eval_freq = int(config.get("eval_freq", 0))
    if eval_freq > 0:
        callbacks.append(
            EvalCallback(
                build_eval_env(config),
                best_model_save_path=str(run_dir / "best_model"),
                log_path=str(run_dir / "eval"),
                eval_freq=eval_freq,
                n_eval_episodes=int(config.get("eval_episodes", 5)),
                deterministic=bool(config.get("deterministic_eval", True)),
                render=False,
            )
        )

    if not callbacks:
        return None
    return CallbackList(callbacks)


def train(config):
    run_dir = resolve_project_path(config.get("run_dir", "runs/relocate_sac"))
    os.makedirs(run_dir, exist_ok=True)

    monitor_filename = config.get("monitor_filename", "train_monitor.csv")
    monitor_path = run_dir / monitor_filename
    info_keywords = tuple(config.get("info_keywords", ["solved"]))

    env = Monitor(
        build_env(config),
        filename=str(monitor_path),
        info_keywords=info_keywords,
    )
    callback = build_callbacks(config, run_dir)

    sac_kwargs = dict(config.get("sac", {}))
    model = SAC(
        config.get("policy", "MlpPolicy"),
        env,
        verbose=config.get("verbose", 1),
        device=config.get("device", "cuda"),
        tensorboard_log=str(run_dir / "tb"),
        **sac_kwargs,
    )

    print(f"run_dir={run_dir}")
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

    model_path = run_dir / config.get("model_name", "sac_relocate")
    model.save(str(model_path))
    print(f"saved_model={model_path}.zip")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=int(config.get("eval_episodes", 5)),
        deterministic=bool(config.get("deterministic_eval", True)),
    )

    print("eval_mean_reward:", mean_reward)
    print("eval_std_reward:", std_reward)

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
