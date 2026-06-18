import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from refine_tabletop_env import make_refine_tabletop_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "scripts" / "config" / "refine" / "eval_refine_tabletop_bulk_all_zero_action.json"
)


def resolve_project_path(path: Optional[str]) -> Optional[Path]:
    if path is None or str(path) == "":
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_summary_csv(path: Optional[Path], rows: List[Dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_index",
        "new_index",
        "object_name",
        "mode",
        "return",
        "length",
        "success",
        "ever_solved",
        "dropped",
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
        "source_grasp_path",
        "source_scene_path",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary_csv={path}")


def entry_indices(num_entries: int, start_index: int, max_count: int) -> List[int]:
    end_index = num_entries if max_count <= 0 else min(num_entries, start_index + max_count)
    return list(range(max(0, start_index), end_index))


def evaluate_entry(
    env_config: Dict[str, Any],
    source_index: int,
    horizon: int,
    seed: int,
    stop_on_done: bool,
    render: bool,
    sleep: float,
    step_log_interval: int,
) -> Dict[str, Any]:
    fixed_config = dict(env_config)
    fixed_config["sample_mode"] = "fixed"
    fixed_config["target_sample_mode"] = "same_initial"
    fixed_config["initial_grasp_index"] = int(source_index)
    fixed_config["target_grasp_index"] = int(source_index)
    fixed_config["horizon"] = int(horizon)

    env = make_refine_tabletop_env(fixed_config, seed=seed)
    try:
        _, info = env.reset(seed=seed)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        ep_return = 0.0
        ep_len = 0
        ever_solved = bool(info.get("solved", False))
        any_dropped = bool(info.get("dropped", False))
        terminated = False
        truncated = False
        final_info = dict(info)

        for step in range(horizon):
            _, reward, terminated, truncated, info = env.step(action)
            final_info = dict(info)
            ep_return += float(reward)
            ep_len = step + 1
            ever_solved = ever_solved or bool(info.get("solved", False))
            any_dropped = any_dropped or bool(info.get("dropped", False))

            if render:
                env.render()
            if sleep > 0:
                time.sleep(sleep)
            if step_log_interval > 0 and (
                step % step_log_interval == 0 or terminated or truncated
            ):
                print(
                    f"  source_index={source_index} step={step:04d} "
                    f"reward={float(reward):.3f} "
                    f"pos={float(info.get('pos_err', float('nan'))):.4f} "
                    f"rot={float(info.get('rot_err', float('nan'))):.4f} "
                    f"contacts={info.get('contact_count', '')} "
                    f"dropped={info.get('dropped', '')} "
                    f"solved={info.get('solved', '')}"
                )
            if stop_on_done and (terminated or truncated):
                break
    finally:
        env.close()

    dropped = any_dropped or bool(final_info.get("dropped", False))
    success = bool(final_info.get("solved", False)) and not dropped
    return {
        "source_index": source_index,
        "new_index": "",
        "object_name": final_info.get("object_name", ""),
        "mode": final_info.get("mode", ""),
        "return": f"{ep_return:.6f}",
        "length": ep_len,
        "success": int(success),
        "ever_solved": int(ever_solved),
        "dropped": int(dropped),
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
        "source_grasp_path": final_info.get("source_grasp_path", ""),
        "source_scene_path": final_info.get("source_scene_path", ""),
        "error": "",
    }


def subset_npz(source_npz_path: Optional[Path], output_npz_path: Path, stable_indices: List[int]) -> None:
    if source_npz_path is None or not source_npz_path.exists():
        print(f"skip_npz_copy source_npz_missing={source_npz_path}")
        return
    arrays = np.load(source_npz_path, allow_pickle=True)
    selected = np.asarray(stable_indices, dtype=np.int64)
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        **{name: arrays[name][selected] for name in arrays.files},
    )
    print(f"npz_path={output_npz_path}")


def write_filtered_manifest(
    source_manifest: Dict[str, Any],
    source_manifest_path: Path,
    output_dir: Path,
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    stable_rows = [row for row in rows if int(row["success"]) == 1]
    stable_indices = [int(row["source_index"]) for row in stable_rows]
    source_entries = list(source_manifest.get("entries", []))

    entries = []
    for new_index, source_index in enumerate(stable_indices):
        entry = copy.deepcopy(source_entries[source_index])
        entry["source_manifest_index"] = int(source_index)
        entry["source_entry_index"] = int(entry.get("index", source_index))
        entry["index"] = int(new_index)
        entry["hold_validation"] = dict(stable_rows[new_index])
        entry["hold_validation"]["new_index"] = int(new_index)
        entries.append(entry)
        stable_rows[new_index]["new_index"] = new_index

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "grasp_states.npz"
    source_npz = resolve_project_path(source_manifest.get("npz_path"))
    subset_npz(source_npz, npz_path, stable_indices)

    manifest = copy.deepcopy(source_manifest)
    manifest["schema"] = "refine_grasp_dataset.v1.hold_filtered"
    manifest["source_manifest_path"] = str(source_manifest_path)
    manifest["npz_path"] = str(npz_path)
    manifest["hold_filter"] = {
        "success_definition": "final_solved_and_not_dropped",
        "config": config,
    }
    manifest["counts"] = {
        **dict(source_manifest.get("counts", {})),
        "num_source_entries": len(source_entries),
        "num_evaluated_entries": len(rows),
        "num_entries": len(entries),
        "num_failed_entries": len(rows) - len(entries),
        "num_dropped_entries": sum(int(row["dropped"]) for row in rows),
    }
    manifest["entries"] = entries

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest_path={manifest_path}")
    return manifest


def filter_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    input_manifest_path = resolve_project_path(config["manifest_path"])
    output_dir = resolve_project_path(config["output_dir"])
    summary_csv = resolve_project_path(config.get("summary_csv"))
    assert input_manifest_path is not None
    assert output_dir is not None

    source_manifest = load_json(input_manifest_path)
    source_entries = list(source_manifest.get("entries", []))
    if not source_entries:
        raise ValueError(f"No entries in manifest: {input_manifest_path}")

    env_config = dict(config.get("eval_refine_tabletop", config.get("refine_tabletop", {})))
    env_config["manifest_path"] = str(input_manifest_path)
    horizon = int(config.get("horizon", env_config.get("horizon", 100)))
    seed = int(config.get("seed", 123))
    stop_on_done = bool(config.get("stop_on_done", True))
    render = bool(config.get("render", False))
    sleep = float(config.get("sleep", 0.0))
    step_log_interval = int(config.get("step_log_interval", 0))
    progress_interval = int(config.get("progress_interval", 10))
    indices = entry_indices(
        len(source_entries),
        int(config.get("start_index", 0)),
        int(config.get("max_count", 0)),
    )

    print(f"input_manifest={input_manifest_path}")
    print(f"output_dir={output_dir}")
    print(f"entries_total={len(source_entries)}")
    print(f"entries_to_evaluate={len(indices)}")
    print(f"horizon={horizon}")
    print("success_definition=final_solved_and_not_dropped")

    rows = []
    for offset, source_index in enumerate(indices, start=1):
        source_entry = source_entries[source_index]
        try:
            row = evaluate_entry(
                env_config=env_config,
                source_index=source_index,
                horizon=horizon,
                seed=seed + source_index,
                stop_on_done=stop_on_done,
                render=render,
                sleep=sleep,
                step_log_interval=step_log_interval,
            )
        except Exception as exc:
            row = {
                "source_index": source_index,
                "new_index": "",
                "object_name": source_entry.get("object_name", ""),
                "mode": source_entry.get("mode", ""),
                "return": "nan",
                "length": 0,
                "success": 0,
                "ever_solved": 0,
                "dropped": 0,
                "terminated": 0,
                "truncated": 0,
                "final_pos_err": "nan",
                "final_rot_err": "nan",
                "final_hand_err": "nan",
                "final_contact_count": "",
                "final_thumb_contact_count": "",
                "final_opposing_contact_count": "",
                "final_opposing_finger_groups": "",
                "final_opposition_score": "nan",
                "lift_offset": "",
                "source_grasp_path": source_entry.get("source_grasp_path", ""),
                "source_scene_path": source_entry.get("source_scene_path", ""),
                "error": str(exc),
            }
            print(f"entry_error source_index={source_index} error={exc}")
        row["object_name"] = row.get("object_name") or source_entry.get("object_name", "")
        row["mode"] = row.get("mode") or source_entry.get("mode", "")
        row["source_grasp_path"] = source_entry.get("source_grasp_path", "")
        row["source_scene_path"] = source_entry.get("source_scene_path", "")
        rows.append(row)
        if progress_interval > 0 and (
            offset == 1 or offset % progress_interval == 0 or offset == len(indices)
        ):
            success_count = sum(int(item["success"]) for item in rows)
            drop_count = sum(int(item["dropped"]) for item in rows)
            print(
                f"progress={offset}/{len(indices)} "
                f"source_index={source_index} "
                f"success={row['success']} dropped={row['dropped']} "
                f"stable_so_far={success_count} dropped_so_far={drop_count}"
            )

    manifest = write_filtered_manifest(
        source_manifest=source_manifest,
        source_manifest_path=input_manifest_path,
        output_dir=output_dir,
        rows=rows,
        config=config,
    )
    write_summary_csv(summary_csv, rows)

    evaluated = max(len(rows), 1)
    success_count = sum(int(row["success"]) for row in rows)
    drop_count = sum(int(row["dropped"]) for row in rows)
    result = {
        "input_manifest": str(input_manifest_path),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv) if summary_csv is not None else None,
        "evaluated": len(rows),
        "stable": success_count,
        "failed": len(rows) - success_count,
        "dropped": drop_count,
        "success_rate": success_count / evaluated,
        "drop_rate": drop_count / evaluated,
        "manifest_entries": len(manifest.get("entries", [])),
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--manifest-path", "--manifest_path", dest="manifest_path", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--summary-csv", "--summary_csv", dest="summary_csv", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=None)
    parser.add_argument("--step-log-interval", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json(resolve_project_path(args.config))
    overrides = {
        "manifest_path": args.manifest_path,
        "output_dir": args.output_dir,
        "summary_csv": args.summary_csv,
        "horizon": args.horizon,
        "seed": args.seed,
        "start_index": args.start_index,
        "max_count": args.max_count,
        "progress_interval": args.progress_interval,
        "sleep": args.sleep,
        "step_log_interval": args.step_log_interval,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    if args.render:
        config["render"] = True
    filter_manifest(config)


if __name__ == "__main__":
    main()
