import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from refine_tabletop_env import make_refine_tabletop_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "scripts" / "config" / "refine" / "eval_refine_tabletop_bulk_zero_action.json"
)


def resolve_project_path(path: Optional[str]) -> Optional[Path]:
    if path is None or str(path) == "":
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_summary_csv(path: Optional[Path], rows):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "object_name",
        "mode",
        "initial_grasp_index",
        "target_grasp_index",
        "return",
        "length",
        "solved",
        "terminated",
        "truncated",
        "final_pos_err",
        "final_rot_err",
        "final_hand_err",
        "final_contact_count",
        "final_thumb_contact_count",
        "final_opposing_contact_count",
        "final_opposing_finger_groups",
        "final_opposition_score",
        "lift_offset",
        "dropped",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary_csv={path}")


def evaluate(config: Dict[str, Any]):
    env_config = dict(config.get("eval_refine_tabletop", config.get("refine_tabletop", {})))
    episodes = int(config.get("episodes", 50))
    horizon = int(config.get("horizon", env_config.get("horizon", 100)))
    seed = int(config.get("seed", 123))
    render = bool(config.get("render", False))
    sleep = float(config.get("sleep", 0.0))
    step_log_interval = int(config.get("step_log_interval", 0))
    stop_on_done = bool(config.get("stop_on_done", True))
    summary_csv = resolve_project_path(config.get("summary_csv"))

    env = make_refine_tabletop_env(env_config, seed=seed)
    rows = []
    returns = []
    lengths = []
    solved_count = 0
    dropped_count = 0

    print(f"episodes={episodes}")
    print(f"horizon={horizon}")
    print(f"manifest_path={env_config.get('manifest_path')}")
    print(f"sample_mode={env_config.get('sample_mode')}")
    print(f"gravity_scale={env_config.get('gravity_scale')}")
    print(f"action_scale={env_config.get('action_scale')}")

    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            ep_return = 0.0
            ep_len = 0
            solved = bool(info.get("solved", False))
            terminated = False
            truncated = False
            final_info = dict(info)

            print(
                f"episode={ep} start object={info.get('object_name', '')} "
                f"mode={info.get('mode', '')} "
                f"init={info.get('initial_grasp_index', '')} "
                f"lift={info.get('lift_offset', '')} "
                f"contacts={info.get('contact_count', '')} "
                f"dropped={info.get('dropped', '')}"
            )

            for step in range(horizon):
                obs, reward, terminated, truncated, info = env.step(action)
                final_info = dict(info)
                ep_return += float(reward)
                ep_len = step + 1
                solved = solved or bool(info.get("solved", False))

                if render:
                    env.render()
                if sleep > 0:
                    time.sleep(sleep)
                if step_log_interval > 0 and (
                    step % step_log_interval == 0 or terminated or truncated
                ):
                    print(
                        f"  step={step:04d} reward={float(reward):.3f} "
                        f"pos={float(info.get('pos_err', float('nan'))):.4f} "
                        f"rot={float(info.get('rot_err', float('nan'))):.4f} "
                        f"contacts={info.get('contact_count', '')} "
                        f"dropped={info.get('dropped', '')} "
                        f"solved={info.get('solved', '')}"
                    )
                if stop_on_done and (terminated or truncated):
                    break

            row = {
                "episode": ep,
                "object_name": final_info.get("object_name", ""),
                "mode": final_info.get("mode", ""),
                "initial_grasp_index": final_info.get("initial_grasp_index", ""),
                "target_grasp_index": final_info.get("target_grasp_index", ""),
                "return": f"{ep_return:.6f}",
                "length": ep_len,
                "solved": int(solved),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "final_pos_err": f"{float(final_info.get('pos_err', float('nan'))):.6f}",
                "final_rot_err": f"{float(final_info.get('rot_err', float('nan'))):.6f}",
                "final_hand_err": f"{float(final_info.get('hand_err', float('nan'))):.6f}",
                "final_contact_count": final_info.get("contact_count", ""),
                "final_thumb_contact_count": final_info.get("thumb_contact_count", ""),
                "final_opposing_contact_count": final_info.get("opposing_contact_count", ""),
                "final_opposing_finger_groups": final_info.get("opposing_finger_groups", ""),
                "final_opposition_score": f"{float(final_info.get('opposition_score', float('nan'))):.6f}",
                "lift_offset": final_info.get("lift_offset", ""),
                "dropped": int(bool(final_info.get("dropped", False))),
            }
            rows.append(row)
            returns.append(ep_return)
            lengths.append(ep_len)
            solved_count += int(solved)
            dropped_count += int(bool(final_info.get("dropped", False)))
            print(
                f"episode={ep} return={ep_return:.3f} length={ep_len} "
                f"solved={solved} dropped={row['dropped']} "
                f"final_pos={row['final_pos_err']} final_rot={row['final_rot_err']} "
                f"contacts={row['final_contact_count']}"
            )
    finally:
        env.close()

    write_summary_csv(summary_csv, rows)
    print("=" * 60)
    print(f"mean_return={float(np.mean(returns)) if returns else float('nan'):.3f}")
    print(f"mean_length={float(np.mean(lengths)) if lengths else float('nan'):.2f}")
    print(f"success_rate={solved_count / max(episodes, 1):.3f} ({solved_count}/{episodes})")
    print(f"drop_rate={dropped_count / max(episodes, 1):.3f} ({dropped_count}/{episodes})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=None)
    parser.add_argument("--step-log-interval", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json(resolve_project_path(args.config))
    overrides = {
        "episodes": args.episodes,
        "horizon": args.horizon,
        "seed": args.seed,
        "summary_csv": args.summary_csv,
        "sleep": args.sleep,
        "step_log_interval": args.step_log_interval,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    if args.render:
        config["render"] = True
    print(f"config={args.config}")
    evaluate(config)


if __name__ == "__main__":
    main()
