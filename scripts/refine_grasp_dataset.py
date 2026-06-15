import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "scripts" / "config" / "refine" / "refine_grasp_dataset_tabletop.json"

REFINE_JOINT_NAMES = [
    "rh_FFJ4",
    "rh_FFJ3",
    "rh_FFJ2",
    "rh_FFJ1",
    "rh_MFJ4",
    "rh_MFJ3",
    "rh_MFJ2",
    "rh_MFJ1",
    "rh_RFJ4",
    "rh_RFJ3",
    "rh_RFJ2",
    "rh_RFJ1",
    "rh_LFJ5",
    "rh_LFJ4",
    "rh_LFJ3",
    "rh_LFJ2",
    "rh_LFJ1",
    "rh_THJ5",
    "rh_THJ4",
    "rh_THJ3",
    "rh_THJ2",
    "rh_THJ1",
]

ADROIT_HAND_JOINT_NAMES = [
    "WRJ1",
    "WRJ0",
    "FFJ3",
    "FFJ2",
    "FFJ1",
    "FFJ0",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "MFJ0",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "RFJ0",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "LFJ0",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
    "THJ0",
]

REFINE_TO_ADROIT = {
    "rh_FFJ4": "FFJ3",
    "rh_FFJ3": "FFJ2",
    "rh_FFJ2": "FFJ1",
    "rh_FFJ1": "FFJ0",
    "rh_MFJ4": "MFJ3",
    "rh_MFJ3": "MFJ2",
    "rh_MFJ2": "MFJ1",
    "rh_MFJ1": "MFJ0",
    "rh_RFJ4": "RFJ3",
    "rh_RFJ3": "RFJ2",
    "rh_RFJ2": "RFJ1",
    "rh_RFJ1": "RFJ0",
    "rh_LFJ5": "LFJ4",
    "rh_LFJ4": "LFJ3",
    "rh_LFJ3": "LFJ2",
    "rh_LFJ2": "LFJ1",
    "rh_LFJ1": "LFJ0",
    "rh_THJ5": "THJ4",
    "rh_THJ4": "THJ3",
    "rh_THJ3": "THJ2",
    "rh_THJ2": "THJ1",
    "rh_THJ1": "THJ0",
}

CONTACT_GROUPS = ["palm", "thumb", "index", "middle", "ring", "little", "other"]


def project_path(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_npy_dict(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        data = data.item()
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data)!r}")
    return data


def resolve_grasp_path(path: str) -> Optional[Path]:
    candidates = [Path(path).expanduser()]
    if "/succ_grasp/" in path:
        candidates.append(Path(path.replace("/succ_grasp/", "/grasp_data/")))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def derive_scene_path(grasp_path: Path) -> Optional[Path]:
    parts = list(grasp_path.parts)
    if "grasp_data" not in parts:
        return None
    idx = parts.index("grasp_data")
    after = parts[idx + 1 :]
    if len(after) < 5:
        return None

    _, object_name, mode_name, scale_name, file_name = after[:5]
    scene_file = file_name.replace("_grasp.npy", ".npy")
    scene_root = Path(*parts[:idx]) / "scene_cfg"
    return scene_root / object_name / mode_name / scale_name / scene_file


def resolve_scene_path(result: Dict[str, Any], grasp_data: Dict[str, Any], grasp_path: Path) -> Optional[Path]:
    candidates = []
    if result.get("scene_path"):
        candidates.append(Path(str(result["scene_path"])).expanduser())

    derived = derive_scene_path(grasp_path)
    if derived is not None:
        candidates.append(derived)

    if grasp_data.get("scene_path"):
        candidates.append(Path(str(grasp_data["scene_path"])).expanduser())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return derived


def scene_metadata(scene_path: Optional[Path], object_name: Optional[str]) -> Dict[str, Any]:
    if scene_path is None or not scene_path.exists():
        return {}
    scene = load_npy_dict(scene_path)
    task = scene.get("task", {})
    obj_name = task.get("obj_name", object_name)
    obj_cfg = scene.get("scene", {}).get(obj_name, {})
    table_cfg = scene.get("scene", {}).get("table", {})
    meta = {
        "scene_task": task,
        "object_xml_path": str(obj_cfg.get("xml_path", "")),
        "object_mesh_path": str(obj_cfg.get("file_path", "")),
        "object_urdf_path": str(obj_cfg.get("urdf_path", "")),
        "object_info_path": str(obj_cfg.get("info_path", "")),
        "object_scale": json_array(np.asarray(obj_cfg.get("scale", [1.0, 1.0, 1.0]))),
        "object_scene_pose_wxyz": json_array(np.asarray(obj_cfg.get("pose", [0, 0, 0, 1, 0, 0, 0]))),
        "table_pose_wxyz": json_array(np.asarray(table_cfg.get("pose", [0, 0, 0, 1, 0, 0, 0]))),
        "table_size": json_array(np.asarray(table_cfg.get("size", [0.34, 0.34, 0.01]))),
        "table_axis": str(task.get("table_axis", "xz")),
    }
    return meta


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def passes_filter(result: Dict[str, Any], config: Dict[str, Any]) -> bool:
    mode = str(config.get("mode", "tabletop"))
    if mode != "all" and result.get("mode") != mode:
        return False
    if as_bool(config.get("require_success", True)) and not as_bool(result.get("success", False)):
        return False
    if as_bool(config.get("require_gravity_success", True)) and not as_bool(result.get("gravity_success", False)):
        return False

    max_trans = config.get("max_trans", None)
    if max_trans is not None:
        metric = str(config.get("max_trans_metric", "gravity_max_trans"))
        if float(result.get(metric, float("inf"))) > float(max_trans):
            return False

    max_rot_deg = config.get("max_rot_deg", None)
    if max_rot_deg is not None:
        metric = str(config.get("max_rot_metric", "gravity_max_rot_deg"))
        if float(result.get(metric, float("inf"))) > float(max_rot_deg):
            return False

    min_groups = config.get("min_final_contact_groups", None)
    if min_groups is not None:
        groups = int(result.get("gravity_final_contact_groups", 0))
        if groups < int(min_groups):
            return False

    return True


def normalize_qpos(qpos: Any, expected_size: int, name: str) -> np.ndarray:
    arr = np.asarray(qpos, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    arr = arr.reshape(-1)
    if arr.size != expected_size:
        raise ValueError(f"{name} must have {expected_size} values, got {arr.size}")
    return arr


def refine_to_adroit_hand_qpos(refine_finger_qpos: np.ndarray) -> np.ndarray:
    refine_finger_qpos = normalize_qpos(refine_finger_qpos, 22, "refine_finger_qpos")
    refine_by_name = dict(zip(REFINE_JOINT_NAMES, refine_finger_qpos.tolist()))
    adroit = np.zeros(len(ADROIT_HAND_JOINT_NAMES), dtype=np.float32)
    adroit_index = {name: idx for idx, name in enumerate(ADROIT_HAND_JOINT_NAMES)}
    for refine_name, adroit_name in REFINE_TO_ADROIT.items():
        adroit[adroit_index[adroit_name]] = refine_by_name[refine_name]
    return adroit


def contact_group(body_name: str) -> str:
    name = body_name.lower()
    if "palm" in name:
        return "palm"
    if "th" in name:
        return "thumb"
    if "ff" in name:
        return "index"
    if "mf" in name:
        return "middle"
    if "rf" in name:
        return "ring"
    if "lf" in name:
        return "little"
    return "other"


def contact_group_mask(body_names: Iterable[str]) -> np.ndarray:
    mask = np.zeros(len(CONTACT_GROUPS), dtype=np.int8)
    index = {name: idx for idx, name in enumerate(CONTACT_GROUPS)}
    for body_name in body_names:
        mask[index[contact_group(str(body_name))]] = 1
    return mask


def json_array(array: np.ndarray) -> List[float]:
    return [float(x) for x in np.asarray(array).reshape(-1).tolist()]


def build_entry(
    index: int,
    result: Dict[str, Any],
    grasp_path: Path,
    grasp_data: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    qpos_key = str(result.get("qpos_key", config.get("qpos_key", "grasp_qpos")))
    if qpos_key not in grasp_data:
        qpos_key = str(config.get("qpos_key", "grasp_qpos"))
    if qpos_key not in grasp_data:
        raise KeyError(f"Missing qpos key {qpos_key!r} in {grasp_path}")

    refine_qpos29 = normalize_qpos(grasp_data[qpos_key], 29, qpos_key)
    refine_hand_pose_wxyz = refine_qpos29[:7]
    refine_finger_qpos22 = refine_qpos29[7:]
    adroit_hand_qpos24 = refine_to_adroit_hand_qpos(refine_finger_qpos22)

    object_pose_wxyz = normalize_qpos(grasp_data.get("obj_pose", np.zeros(7)), 7, "obj_pose")
    gravity_vector = normalize_qpos(result.get("gravity_vector", [0.0, 0.0, -9.81]), 3, "gravity_vector")
    scene_path = resolve_scene_path(result, grasp_data, grasp_path)
    scene_meta = scene_metadata(scene_path, result.get("object_name"))
    contact_bodies = [str(x) for x in grasp_data.get("hand_cbody", [])]
    group_mask = contact_group_mask(contact_bodies)

    metrics = {
        key: value
        for key, value in result.items()
        if key.startswith("gravity_") or key in {"success", "mode", "object_name", "qpos_key"}
    }

    entry = {
        "index": index,
        "mode": result.get("mode"),
        "object_name": result.get("object_name"),
        "source_grasp_path": str(grasp_path),
        "source_scene_path": str(scene_path) if scene_path is not None else None,
        **scene_meta,
        "qpos_key": qpos_key,
        "hand_name": str(grasp_data.get("hand_name", "")),
        "template_name": str(grasp_data.get("tmpl_name", "")),
        "contact_bodies": contact_bodies,
        "contact_groups": [
            name for name, present in zip(CONTACT_GROUPS, group_mask.tolist()) if int(present)
        ],
        "metrics": metrics,
        "refine_qpos_world_wxyz": json_array(refine_qpos29),
        "refine_hand_pose_world_wxyz": json_array(refine_hand_pose_wxyz),
        "refine_finger_qpos22": json_array(refine_finger_qpos22),
        "adroit_hand_qpos24": json_array(adroit_hand_qpos24),
        "object_pose_world_wxyz": json_array(object_pose_wxyz),
    }
    arrays = {
        "refine_qpos29": refine_qpos29,
        "refine_hand_pose_wxyz": refine_hand_pose_wxyz,
        "refine_finger_qpos22": refine_finger_qpos22,
        "adroit_hand_qpos24": adroit_hand_qpos24,
        "object_pose_wxyz": object_pose_wxyz,
        "gravity_vector": gravity_vector,
        "contact_group_mask": group_mask,
    }
    return entry, arrays


def prepare_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    stability_path = project_path(config["stability_json"])
    stability = load_json(stability_path)
    output_dir = project_path(config.get("output_dir", "runs/refine_grasps/tabletop_stable"))
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    arrays_by_name: Dict[str, List[np.ndarray]] = {}
    skipped_missing = 0
    skipped_error = 0
    missing_paths = []
    error_paths = []
    max_count = int(config.get("max_count", 0))

    for result in stability.get("results", []):
        if not passes_filter(result, config):
            continue

        grasp_path = resolve_grasp_path(str(result.get("path", "")))
        if grasp_path is None:
            skipped_missing += 1
            missing_paths.append(str(result.get("path", "")))
            continue

        try:
            grasp_data = load_npy_dict(grasp_path)
            entry, arrays = build_entry(len(entries), result, grasp_path, grasp_data, config)
        except Exception as exc:
            skipped_error += 1
            error_paths.append({"path": str(grasp_path), "error": str(exc)})
            print(f"skip_error path={grasp_path} error={exc}")
            continue

        entries.append(entry)
        for name, value in arrays.items():
            arrays_by_name.setdefault(name, []).append(value)

        if max_count > 0 and len(entries) >= max_count:
            break

    if not entries:
        raise RuntimeError("No stable refine grasps matched the requested filters.")

    npz_arrays = {
        name: np.stack(values, axis=0) for name, values in arrays_by_name.items()
    }
    npz_arrays["source_grasp_path"] = np.asarray([e["source_grasp_path"] for e in entries])
    npz_arrays["source_scene_path"] = np.asarray([e["source_scene_path"] or "" for e in entries])
    npz_arrays["object_name"] = np.asarray([e["object_name"] or "" for e in entries])
    npz_arrays["mode"] = np.asarray([e["mode"] or "" for e in entries])

    manifest = {
        "schema": "refine_grasp_dataset.v1",
        "stability_json": str(stability_path),
        "input_root": stability.get("input_root"),
        "qpos_format": "refine qpos = hand xyz + hand quat wxyz + 22 ShadowHand finger joints",
        "adroit_hand_qpos24_order": ADROIT_HAND_JOINT_NAMES,
        "contact_group_order": CONTACT_GROUPS,
        "filters": config,
        "counts": {
            "num_results_total": len(stability.get("results", [])),
            "num_entries": len(entries),
            "skipped_missing": skipped_missing,
            "skipped_error": skipped_error,
        },
        "missing_paths": missing_paths,
        "error_paths": error_paths,
        "npz_path": str(output_dir / "grasp_states.npz"),
        "entries": entries,
    }

    manifest_path = output_dir / "manifest.json"
    npz_path = output_dir / "grasp_states.npz"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    np.savez_compressed(npz_path, **npz_arrays)

    return {
        "manifest_path": str(manifest_path),
        "npz_path": str(npz_path),
        "num_entries": len(entries),
        "skipped_missing": skipped_missing,
        "skipped_error": skipped_error,
    }


def load_config(path: Path) -> Dict[str, Any]:
    return load_json(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stability-json", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mode", default=None, choices=["tabletop", "unconstrained", "all"])
    parser.add_argument("--qpos-key", default=None)
    parser.add_argument("--max-trans", type=float, default=None)
    parser.add_argument("--max-rot-deg", type=float, default=None)
    parser.add_argument("--min-final-contact-groups", type=int, default=None)
    parser.add_argument("--max-count", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(project_path(args.config))
    overrides = {
        "stability_json": args.stability_json,
        "output_dir": args.output_dir,
        "mode": args.mode,
        "qpos_key": args.qpos_key,
        "max_trans": args.max_trans,
        "max_rot_deg": args.max_rot_deg,
        "min_final_contact_groups": args.min_final_contact_groups,
        "max_count": args.max_count,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    summary = prepare_dataset(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
