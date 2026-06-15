import argparse
import json
import time
from pathlib import Path

import gymnasium
import numpy as np
import robohive
from robohive.utils import gym as rhgym
from stable_baselines3 import SAC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "visualize_relocate_sac.json"


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

        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def visualize(config):
    raw_env = rhgym.make(
        config["env_id"],
        seed=config.get("seed", 123),
        disable_env_checker=True,
    )
    env = RoboHiveSB3Compat(raw_env)
    model_path = resolve_project_path(config["model_path"])
    model = SAC.load(str(model_path), device=config.get("device", "cuda"))

    episodes = int(config.get("episodes", 3))
    horizon = int(config.get("horizon", 200))
    sleep = float(config.get("sleep", 0.01))
    deterministic = bool(config.get("deterministic", True))
    seed = int(config.get("seed", 123))

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        solved = False

        for _ in range(horizon):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            raw_env.unwrapped.mj_render()
            if sleep > 0:
                time.sleep(sleep)

            ep_return += reward
            solved = solved or bool(info.get("solved", False))

            if terminated or truncated:
                break

        print(f"episode={ep} return={ep_return:.3f} solved={solved}")

    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the JSON visualization config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    print(f"config={args.config}")
    visualize(config)


if __name__ == "__main__":
    main()
