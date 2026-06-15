"""Visualize trained Stable-Baselines3 policies on RoboHive environments.

This script is the visualization counterpart of train_parallel_algos.py.  It can
load SAC/PPO/TD3/DDPG/A2C checkpoints by reading either a dedicated visualization
JSON config or the training JSON config used for a run.

Typical usage:
    python visualize_algos.py --config config/visualize_relocate_ppo.json

    # Or directly use the training config and ask for the best model:
    python visualize_algos.py --train-config config/train_relocate_ppo_parallel.json \
        --model-choice best --episodes 5

Config fields accepted by this script:
    env_id: RoboHive environment id, e.g. "relocate-v1".
    algorithm: one of SAC, PPO, TD3, DDPG, A2C.
    model_path: explicit path to a .zip model. Optional.
    train_config: path to the training config. Optional.
    run_dir: run directory used during training. Optional if model_path exists.
    model_name: final model name used by training. Optional.
    model_choice: "best", "final", or "path". Default: "best".
    device: "cuda", "cpu", or "auto".
    seed, episodes, horizon, sleep, deterministic.
    render: whether to call raw_env.unwrapped.mj_render().
    summary_csv: optional CSV path for episode-level results.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import gymnasium
import numpy as np
import robohive  # noqa: F401  # importing registers RoboHive envs
from robohive.utils import gym as rhgym
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "pen" / "train_pen_ppo_parallel.json"

ALGO_REGISTRY: Dict[str, Type[Any]] = {
    "SAC": SAC,
    "PPO": PPO,
    "TD3": TD3,
    "DDPG": DDPG,
    "A2C": A2C,
}


class RoboHiveSB3Compat(gymnasium.Wrapper):
    """Convert RoboHive/Gym reset-step outputs to Gymnasium-style outputs."""

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
    if path is None or str(path) == "":
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_config(config: Dict[str, Any], train_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge visualization config over training config.

    The visualization config has priority. This allows using the original
    training JSON directly, while still overriding episodes/model_choice/etc.
    """
    merged: Dict[str, Any] = {}
    if train_config:
        merged.update(train_config)
    merged.update({k: v for k, v in config.items() if v is not None})
    return merged


def infer_algorithm(config: Dict[str, Any]) -> str:
    algo = str(config.get("algorithm", "SAC")).upper()
    if algo not in ALGO_REGISTRY:
        raise ValueError(
            f"Unsupported algorithm: {algo}. Supported: {sorted(ALGO_REGISTRY)}"
        )
    return algo


def ensure_zip_suffix(path: Path) -> Path:
    if path.suffix == ".zip":
        return path
    zip_path = Path(str(path) + ".zip")
    return zip_path if zip_path.exists() else path


def infer_model_path(config: Dict[str, Any], algorithm: str) -> Path:
    explicit = resolve_project_path(config.get("model_path"))
    if explicit is not None:
        return ensure_zip_suffix(explicit)

    run_dir = resolve_project_path(config.get("run_dir"))
    if run_dir is None:
        raise ValueError(
            "model_path is not set and run_dir is unavailable. "
            "Provide either model_path or train_config/run_dir."
        )

    model_choice = str(config.get("model_choice", "best")).lower()

    if model_choice == "best":
        candidates = [
            run_dir / "best_model" / "best_model.zip",
            run_dir / "best_model.zip",
        ]
    elif model_choice == "final":
        model_name = config.get("model_name")
        if model_name:
            candidates = [run_dir / f"{model_name}.zip", run_dir / str(model_name)]
        else:
            candidates = [
                run_dir / f"{algorithm.lower()}_model.zip",
                run_dir / "model.zip",
            ]
    else:
        raise ValueError("model_choice must be one of: best, final, path")

    for candidate in candidates:
        candidate = ensure_zip_suffix(candidate)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not infer model path. Checked:\n"
        + "\n".join(f"  - {ensure_zip_suffix(c)}" for c in candidates)
        + "\nSet model_path explicitly if the checkpoint is elsewhere."
    )


def build_env(config: Dict[str, Any]) -> Tuple[RoboHiveSB3Compat, Any]:
    raw_env = rhgym.make(
        config["env_id"],
        seed=int(config.get("seed", 123)),
        disable_env_checker=True,
    )
    env = RoboHiveSB3Compat(raw_env)
    return env, raw_env


def load_model(config: Dict[str, Any]):
    algorithm = infer_algorithm(config)
    model_path = infer_model_path(config, algorithm)
    device = config.get("device", "cuda")
    algo_cls = ALGO_REGISTRY[algorithm]
    print(f"algorithm={algorithm}")
    print(f"model_path={model_path}")
    print(f"device={device}")
    return algo_cls.load(str(model_path), device=device), model_path, algorithm


def maybe_render(raw_env: Any, render_enabled: bool):
    if not render_enabled:
        return
    if hasattr(raw_env, "unwrapped") and hasattr(raw_env.unwrapped, "mj_render"):
        raw_env.unwrapped.mj_render()
    elif hasattr(raw_env, "mj_render"):
        raw_env.mj_render()
    elif hasattr(raw_env, "render"):
        raw_env.render()


def write_summary_csv(path: Optional[Path], rows):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "return", "length", "solved", "terminated", "truncated"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary_csv={path}")


def visualize(config: Dict[str, Any]):
    env, raw_env = build_env(config)
    model, model_path, algorithm = load_model(config)

    episodes = int(config.get("episodes", 3))
    horizon = int(config.get("horizon", 200))
    sleep = float(config.get("sleep", 0.01))
    deterministic = bool(config.get("deterministic", True))
    seed = int(config.get("seed", 123))
    render_enabled = bool(config.get("render", True))
    stop_on_done = bool(config.get("stop_on_done", True))
    summary_csv = resolve_project_path(config.get("summary_csv"))

    print(f"env_id={config['env_id']}")
    print(f"episodes={episodes}")
    print(f"horizon={horizon}")
    print(f"deterministic={deterministic}")
    print(f"render={render_enabled}")

    rows = []
    solved_count = 0
    returns = []
    lengths = []

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_return = 0.0
            ep_len = 0
            solved = False
            terminated = False
            truncated = False

            for step in range(horizon):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)

                maybe_render(raw_env, render_enabled)
                if sleep > 0:
                    time.sleep(sleep)

                ep_return += float(reward)
                ep_len = step + 1
                solved = solved or bool(info.get("solved", False))

                if stop_on_done and (terminated or truncated):
                    break

            solved_count += int(solved)
            returns.append(ep_return)
            lengths.append(ep_len)
            row = {
                "episode": ep,
                "return": f"{ep_return:.6f}",
                "length": ep_len,
                "solved": int(solved),
                "terminated": int(terminated),
                "truncated": int(truncated),
            }
            rows.append(row)
            print(
                f"episode={ep} return={ep_return:.3f} length={ep_len} "
                f"solved={solved} terminated={terminated} truncated={truncated}"
            )
    finally:
        env.close()

    write_summary_csv(summary_csv, rows)

    mean_return = float(np.mean(returns)) if returns else float("nan")
    std_return = float(np.std(returns)) if returns else float("nan")
    mean_len = float(np.mean(lengths)) if lengths else float("nan")
    success_rate = solved_count / max(episodes, 1)

    print("=" * 60)
    print(f"algorithm={algorithm}")
    print(f"model_path={model_path}")
    print(f"episodes={episodes}")
    print(f"mean_return={mean_return:.3f}")
    print(f"std_return={std_return:.3f}")
    print(f"mean_length={mean_len:.2f}")
    print(f"success_rate={success_rate:.3f} ({solved_count}/{episodes})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Visualization JSON config. If omitted, uses DEFAULT_CONFIG when it exists.",
    )
    parser.add_argument(
        "--train-config",
        default=None,
        help="Training JSON config. Useful for reusing env_id/run_dir/model_name/algorithm.",
    )
    parser.add_argument("--algorithm", default=None, help="Override algorithm: SAC/PPO/TD3/DDPG/A2C")
    parser.add_argument("--model-path", default=None, help="Override explicit .zip model path")
    parser.add_argument("--model-choice", default=None, choices=["best", "final", "path"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic actions")
    parser.add_argument("--stochastic", action="store_true", help="Force stochastic actions")
    parser.add_argument("--no-render", action="store_true", help="Do not render; only evaluate and print")
    parser.add_argument("--sleep", type=float, default=None)
    parser.add_argument("--summary-csv", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    config: Dict[str, Any] = {}
    config_path = resolve_project_path(args.config) if args.config else None
    if config_path is None and DEFAULT_CONFIG.exists():
        config_path = DEFAULT_CONFIG
    if config_path is not None:
        config.update(load_json(config_path))
        print(f"config={config_path}")

    train_config_path = resolve_project_path(args.train_config or config.get("train_config"))
    train_config = None
    if train_config_path is not None:
        train_config = load_json(train_config_path)
        print(f"train_config={train_config_path}")

    config = merge_config(config, train_config)

    overrides = {
        "algorithm": args.algorithm,
        "model_path": args.model_path,
        "model_choice": args.model_choice,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "seed": args.seed,
        "device": args.device,
        "sleep": args.sleep,
        "summary_csv": args.summary_csv,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    if args.deterministic:
        config["deterministic"] = True
    if args.stochastic:
        config["deterministic"] = False
    if args.no_render:
        config["render"] = False

    if "env_id" not in config:
        raise ValueError("env_id is required. Provide it in config or train_config.")

    visualize(config)


if __name__ == "__main__":
    main()
