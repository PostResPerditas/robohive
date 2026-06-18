import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: Optional[str]) -> Optional[Path]:
    if path is None or str(path) == "":
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_pose(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 7:
        raise ValueError(f"{name} must have 7 values, got {arr.size}")
    arr = arr.copy()
    arr[3:7] = quat_normalize(arr[3:7])
    return arr


def json_array(array: np.ndarray):
    return [float(x) for x in np.asarray(array).reshape(-1).tolist()]


def quat_normalize(quat) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quat = quat / norm
    return quat if quat[0] >= 0 else -quat


def quat_mul(qa, qb) -> np.ndarray:
    qa = quat_normalize(qa)
    qb = quat_normalize(qb)
    return quat_normalize(
        np.asarray(
            [
                qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3],
                qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2],
                qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1],
                qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0],
            ],
            dtype=np.float64,
        )
    )


def quat_conj(quat) -> np.ndarray:
    quat = quat_normalize(quat)
    return np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_rotate(quat, vec) -> np.ndarray:
    quat = quat_normalize(quat)
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    qvec = quat[1:4]
    uv = np.cross(qvec, vec)
    uuv = np.cross(qvec, uv)
    return vec + 2.0 * (quat[0] * uv + uuv)


def pose_inv(pose) -> np.ndarray:
    pose = as_pose(pose, "pose")
    quat_inv = quat_conj(pose[3:7])
    trans_inv = -quat_rotate(quat_inv, pose[:3])
    return np.concatenate([trans_inv, quat_inv])


def pose_mul(pa, pb) -> np.ndarray:
    pa = as_pose(pa, "pa")
    pb = as_pose(pb, "pb")
    pos = pa[:3] + quat_rotate(pa[3:7], pb[:3])
    quat = quat_mul(pa[3:7], pb[3:7])
    return np.concatenate([pos, quat])


def pose_error(pa, pb) -> Dict[str, float]:
    pa = as_pose(pa, "pa")
    pb = as_pose(pb, "pb")
    dot = abs(float(np.dot(pa[3:7], pb[3:7])))
    dot = float(np.clip(dot, -1.0, 1.0))
    return {
        "trans": float(np.linalg.norm(pa[:3] - pb[:3])),
        "rot_deg": float(2.0 * math.acos(dot) * 180.0 / math.pi),
    }


def get_pose(entry: Dict[str, Any], key: str, fallback_key: Optional[str] = None) -> np.ndarray:
    if key in entry and entry[key] is not None:
        return as_pose(entry[key], key)
    if fallback_key is not None and fallback_key in entry and entry[fallback_key] is not None:
        return as_pose(entry[fallback_key], fallback_key)
    raise KeyError(f"Missing pose key {key!r}")


def canonicalize_entry(entry: Dict[str, Any], canonical_hand_pose: np.ndarray) -> Dict[str, Any]:
    new_entry = copy.deepcopy(entry)
    if "hold_validation" in new_entry:
        new_entry["source_hold_validation"] = new_entry.pop("hold_validation")
    source_hand = get_pose(entry, "refine_hand_pose_world_wxyz")
    source_object_scene = get_pose(entry, "object_scene_pose_wxyz", "object_pose_world_wxyz")
    source_object_world = get_pose(entry, "object_pose_world_wxyz", "object_scene_pose_wxyz")

    hand_inv = pose_inv(source_hand)
    object_scene_in_hand = pose_mul(hand_inv, source_object_scene)
    object_world_in_hand = pose_mul(hand_inv, source_object_world)
    canonical_object_scene = pose_mul(canonical_hand_pose, object_scene_in_hand)
    canonical_object_world = pose_mul(canonical_hand_pose, object_world_in_hand)

    new_entry["source_refine_hand_pose_world_wxyz"] = json_array(source_hand)
    new_entry["source_object_scene_pose_wxyz"] = json_array(source_object_scene)
    new_entry["source_object_pose_world_wxyz"] = json_array(source_object_world)
    new_entry["refine_hand_pose_world_wxyz"] = json_array(canonical_hand_pose)
    new_entry["object_scene_pose_wxyz"] = json_array(canonical_object_scene)
    new_entry["object_pose_world_wxyz"] = json_array(canonical_object_world)
    new_entry["object_pose_in_hand_wxyz"] = json_array(object_scene_in_hand)

    if "table_pose_wxyz" in entry and entry["table_pose_wxyz"] is not None:
        source_table = as_pose(entry["table_pose_wxyz"], "table_pose_wxyz")
        table_in_hand = pose_mul(hand_inv, source_table)
        canonical_table = pose_mul(canonical_hand_pose, table_in_hand)
        new_entry["source_table_pose_wxyz"] = json_array(source_table)
        new_entry["table_pose_wxyz"] = json_array(canonical_table)
        new_entry["table_pose_in_hand_wxyz"] = json_array(table_in_hand)

    if "refine_qpos_world_wxyz" in new_entry:
        qpos = np.asarray(new_entry["refine_qpos_world_wxyz"], dtype=np.float64).reshape(-1)
        if qpos.size >= 7:
            qpos[:7] = canonical_hand_pose
            new_entry["refine_qpos_world_wxyz"] = json_array(qpos)

    return new_entry


def load_canonical_pose(manifest: Dict[str, Any], args) -> np.ndarray:
    if args.canonical_hand_pose is not None:
        return as_pose(args.canonical_hand_pose, "canonical_hand_pose")
    entries = manifest.get("entries", [])
    if not entries:
        raise ValueError("Input manifest has no entries")
    base_entry_index = int(args.base_entry_index)
    if base_entry_index < 0 or base_entry_index >= len(entries):
        raise ValueError(f"base_entry_index={base_entry_index} is outside [0, {len(entries)})")
    return get_pose(entries[base_entry_index], "refine_hand_pose_world_wxyz")


def update_npz(source_npz_path: Optional[Path], output_npz_path: Path, entries):
    if source_npz_path is None or not source_npz_path.exists():
        print(f"skip_npz_update source_npz_missing={source_npz_path}")
        return
    arrays = dict(np.load(source_npz_path, allow_pickle=True))
    num_entries = len(entries)
    if arrays:
        first = next(iter(arrays.values()))
        if getattr(first, "shape", (num_entries,))[0] != num_entries:
            raise ValueError(
                f"NPZ row count {getattr(first, 'shape', ['?'])[0]} does not match manifest entries {num_entries}"
            )

    hand = np.asarray([entry["refine_hand_pose_world_wxyz"] for entry in entries], dtype=np.float32)
    obj = np.asarray([entry["object_pose_world_wxyz"] for entry in entries], dtype=np.float32)
    obj_in_hand = np.asarray([entry["object_pose_in_hand_wxyz"] for entry in entries], dtype=np.float32)
    source_hand = np.asarray([entry["source_refine_hand_pose_world_wxyz"] for entry in entries], dtype=np.float32)
    source_obj = np.asarray([entry["source_object_pose_world_wxyz"] for entry in entries], dtype=np.float32)

    arrays["refine_hand_pose_wxyz"] = hand
    arrays["object_pose_wxyz"] = obj
    arrays["object_pose_in_hand_wxyz"] = obj_in_hand
    arrays["source_refine_hand_pose_wxyz"] = source_hand
    arrays["source_object_pose_wxyz"] = source_obj
    if "refine_qpos29" in arrays:
        qpos = np.asarray(arrays["refine_qpos29"]).copy()
        qpos[:, :7] = hand.astype(qpos.dtype)
        arrays["refine_qpos29"] = qpos

    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz_path, **arrays)
    print(f"npz_path={output_npz_path}")


def canonicalize_manifest(args):
    input_manifest_path = resolve_project_path(args.manifest_path)
    output_dir = resolve_project_path(args.output_dir)
    assert input_manifest_path is not None
    assert output_dir is not None
    manifest = load_json(input_manifest_path)
    entries = manifest.get("entries", [])
    if not entries:
        raise ValueError(f"No entries in manifest: {input_manifest_path}")

    canonical_hand_pose = load_canonical_pose(manifest, args)
    output_entries = [canonicalize_entry(entry, canonical_hand_pose) for entry in entries]

    errors = [
        pose_error(entry["refine_hand_pose_world_wxyz"], canonical_hand_pose)
        for entry in output_entries
    ]
    max_hand_trans = max(error["trans"] for error in errors)
    max_hand_rot = max(error["rot_deg"] for error in errors)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz_path = output_dir / "grasp_states.npz"
    source_npz = resolve_project_path(manifest.get("npz_path"))
    update_npz(source_npz, output_npz_path, output_entries)

    output_manifest = copy.deepcopy(manifest)
    output_manifest["schema"] = "refine_grasp_dataset.v1.canonical_hand_base"
    output_manifest["source_manifest_path"] = str(input_manifest_path)
    output_manifest["npz_path"] = str(output_npz_path)
    output_manifest["canonical_hand_base"] = {
        "pose_wxyz": json_array(canonical_hand_pose),
        "base_entry_index": None if args.canonical_hand_pose is not None else int(args.base_entry_index),
        "rule": "preserve T_hand_to_object and rewrite all world poses under one hand base",
        "object_pose_in_hand_key": "object_pose_in_hand_wxyz",
        "source_hold_validation_key": "source_hold_validation",
        "requires_hold_revalidation": True,
        "revalidation_reason": "Canonicalizing the hand base changes gravity/table relation for many entries.",
        "max_output_hand_pose_trans_error": max_hand_trans,
        "max_output_hand_pose_rot_error_deg": max_hand_rot,
    }
    output_manifest["entries"] = output_entries

    output_manifest_path = output_dir / "manifest.json"
    with output_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(output_manifest, f, indent=2)
    print(f"manifest_path={output_manifest_path}")
    print(
        json.dumps(
            {
                "input_manifest": str(input_manifest_path),
                "output_dir": str(output_dir),
                "num_entries": len(output_entries),
                "canonical_hand_pose_wxyz": json_array(canonical_hand_pose),
                "max_output_hand_pose_trans_error": max_hand_trans,
                "max_output_hand_pose_rot_error_deg": max_hand_rot,
            },
            indent=2,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", "--manifest_path", dest="manifest_path", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--base-entry-index", type=int, default=0)
    parser.add_argument("--canonical-hand-pose", type=float, nargs=7, default=None)
    return parser.parse_args()


def main():
    canonicalize_manifest(parse_args())


if __name__ == "__main__":
    main()
