import copy
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAND_XML = Path(
    "/data/Project/Grasp_Refine/grasp_refine/assets/hand/shadow/right.xml"
)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "runs" / "refine_tabletop_models"

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

ACTUATOR_NAMES = [
    "rh_A_THJ5",
    "rh_A_THJ4",
    "rh_A_THJ3",
    "rh_A_THJ2",
    "rh_A_THJ1",
    "rh_A_FFJ4",
    "rh_A_FFJ3",
    "rh_A_FFJ0",
    "rh_A_MFJ4",
    "rh_A_MFJ3",
    "rh_A_MFJ0",
    "rh_A_RFJ4",
    "rh_A_RFJ3",
    "rh_A_RFJ0",
    "rh_A_LFJ5",
    "rh_A_LFJ4",
    "rh_A_LFJ3",
    "rh_A_LFJ0",
]

FINGER_GROUP_PREFIXES = {
    "thumb": ("rh_th",),
    "ff": ("rh_ff",),
    "mf": ("rh_mf",),
    "rf": ("rh_rf",),
    "lf": ("rh_lf",),
    "palm": ("rh_palm",),
}

OPPOSING_FINGER_GROUPS = ("ff", "mf", "rf", "lf")
CONTACT_FEATURE_KEYS = [
    "thumb_contact_count",
    "opposing_contact_count",
    "opposing_finger_groups",
    "palm_contact_count",
    "opposition_score",
    "has_thumb_opposition",
]


def resolve_project_path(path: Optional[str]) -> Optional[Path]:
    if path is None or str(path) == "":
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_vec(value, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name} must have {size} values, got {arr.size}")
    return arr


def vec_str(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in np.asarray(values).reshape(-1))


def quat_normalize(quat) -> np.ndarray:
    quat = as_vec(quat, 4, "quat")
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


def quat_error_angle(qa, qb) -> float:
    qa = quat_normalize(qa)
    qb = quat_normalize(qb)
    dot = abs(float(np.dot(qa, qb)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def axis_angle_quat(axis, angle) -> np.ndarray:
    axis = as_vec(axis, 3, "axis")
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / norm
    half = 0.5 * float(angle)
    return quat_normalize([math.cos(half), *(math.sin(half) * axis)])


def xml_children(parent, tag):
    child = parent.find(tag)
    return list(child) if child is not None else []


def first_scalar(value, default: float) -> float:
    arr = np.asarray(value if value is not None else default, dtype=np.float64).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def as_optional_range(value, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, str):
        arr = np.fromstring(value, sep=" ", dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        limit = abs(float(arr[0]))
        return np.asarray([-limit, limit], dtype=np.float64)
    if arr.size == 2:
        return arr.astype(np.float64)
    raise ValueError(f"{name} must have one symmetric limit or two range values, got {arr.size}")


class RefineTabletopEnv(gymnasium.Env):
    """Fixed-base ShadowHand tabletop env built from Grasp_Refine assets."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        super().__init__()
        self.config = dict(config)
        self.rng = np.random.RandomState(seed if seed is not None else self.config.get("seed", 0))
        self.frame_skip = int(self.config.get("frame_skip", 5))
        self.horizon = int(self.config.get("horizon", 200))
        self.step_count = 0
        self.viewer = None
        self.sample_mode = str(self.config.get("sample_mode", "fixed")).lower()
        if self.sample_mode not in {"fixed", "random_env", "random_reset"}:
            raise ValueError("sample_mode must be one of: fixed, random_env, random_reset")
        self.target_sample_mode = str(self.config.get("target_sample_mode", "same_initial")).lower()
        if self.target_sample_mode not in {"same_initial", "random_same_object"}:
            raise ValueError("target_sample_mode must be one of: same_initial, random_same_object")
        self._base_gravity = as_vec(
            self.config.get("gravity_vector", [0.0, -9.81, 0.0]),
            3,
            "gravity_vector",
        )
        self._gravity_scale = float(self.config.get("gravity_scale", 1.0))

        manifest_path = resolve_project_path(self.config.get("manifest_path"))
        if manifest_path is None:
            raise ValueError("refine_tabletop.manifest_path is required")
        self.manifest = load_json(manifest_path)
        self.entries = list(self.manifest.get("entries", []))
        if not self.entries:
            raise ValueError(f"No entries in manifest: {manifest_path}")
        self.entry_indices = self._eligible_entry_indices()
        self.entries_by_object = self._group_entries_by_object()

        self.hand_xml = resolve_project_path(self.config.get("hand_xml")) or DEFAULT_HAND_XML
        self.action_mode = str(self.config.get("action_mode", "delta")).lower()
        self.action_scale = float(self.config.get("action_scale", 0.15))
        self._last_action_norm = 0.0
        self._select_entries(initial=True)
        self._load_model_for_current_entries()
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

    def seed(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        return [seed]

    @property
    def unwrapped(self):
        return self

    def _eligible_entry_indices(self):
        object_names = self.config.get("object_names", None)
        if object_names is None and self.config.get("object_name", None) is not None:
            object_names = [self.config["object_name"]]
        object_names = set(object_names or [])
        indices = []
        for index, entry in enumerate(self.entries):
            if object_names and entry.get("object_name") not in object_names:
                continue
            indices.append(index)
        if not indices:
            raise ValueError("No manifest entries matched object_name/object_names filter")
        return indices

    def _group_entries_by_object(self):
        groups = {}
        for index in self.entry_indices:
            groups.setdefault(self.entries[index].get("object_name", ""), []).append(index)
        return groups

    def _select_entries(self, initial=False):
        if self.sample_mode == "fixed" or (initial and self.sample_mode == "random_reset"):
            initial_index = int(self.config.get("initial_grasp_index", self.entry_indices[0]))
        else:
            initial_index = int(self.rng.choice(self.entry_indices))

        if initial_index not in self.entry_indices:
            raise ValueError(f"initial_grasp_index={initial_index} is not in the eligible entry set")

        self.initial_index = initial_index
        self.initial_entry = self.entries[self.initial_index]
        self.target_index = self._resolve_target_index()
        self.target_entry = self.entries[self.target_index]
        if self.target_entry.get("object_name") != self.initial_entry.get("object_name"):
            raise ValueError("initial_grasp_index and target_grasp_index must use the same object_name")

    def _resolve_target_index(self) -> int:
        target_index = self.config.get("target_grasp_index", None)
        if target_index is not None and self.sample_mode == "fixed":
            return int(target_index)
        if self.target_sample_mode == "random_same_object":
            candidates = self.entries_by_object.get(self.initial_entry.get("object_name", ""), [])
            return int(self.rng.choice(candidates)) if candidates else self.initial_index
        return self.initial_index

    def _load_model_for_current_entries(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        self.model_xml_path = self._build_model_xml()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_xml_path))
        self.data = mujoco.MjData(self.model)
        self._cache_ids()

        self.fixed_base_pose = self._pose_with_lift(
            as_vec(self.initial_entry["refine_hand_pose_world_wxyz"], 7, "hand pose"),
            self.initial_entry,
        )
        self.init_hand_qpos = as_vec(self.initial_entry["refine_finger_qpos22"], 22, "init hand qpos")
        self.init_object_pose = self._pose_with_lift(
            as_vec(
                self.initial_entry.get("object_scene_pose_wxyz", self.initial_entry["object_pose_world_wxyz"]),
                7,
                "init object pose",
            ),
            self.initial_entry,
        )
        self.target_hand_qpos = as_vec(
            self.target_entry.get("refine_finger_qpos22", self.initial_entry["refine_finger_qpos22"]),
            22,
            "target hand qpos",
        )
        self.target_object_pose = self._target_object_pose()

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float64)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float64)
        self.ctrl_limited = np.asarray(self.model.actuator_ctrllimited, dtype=bool)
        self.default_ctrl = self._initial_ctrl_from_qpos(self.init_hand_qpos)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        self.model.opt.gravity[:] = self._base_gravity * self._gravity_scale

    def _target_object_pose(self) -> np.ndarray:
        if self.config.get("target_object_pose_wxyz") is not None:
            return as_vec(self.config["target_object_pose_wxyz"], 7, "target_object_pose_wxyz")
        pose = self._pose_with_lift(
            as_vec(
                self.target_entry.get("object_scene_pose_wxyz", self.target_entry["object_pose_world_wxyz"]),
                7,
                "target object pose",
            ),
            self.initial_entry,
        )
        if self.config.get("target_relative_axis") is not None:
            axis = as_vec(self.config.get("target_relative_axis"), 3, "target_relative_axis")
            angle = float(self.config.get("target_relative_angle", 0.0))
            pose[3:7] = quat_mul(axis_angle_quat(axis, angle), pose[3:7])
        return pose

    def _table_up_vector(self, entry: Optional[Dict[str, Any]] = None) -> np.ndarray:
        if self.config.get("lift_direction") is not None:
            direction = as_vec(self.config["lift_direction"], 3, "lift_direction")
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                raise ValueError("lift_direction must be non-zero")
            return direction / norm
        entry = entry or self.initial_entry
        axis = str(entry.get("table_axis", "xz")).lower()
        if axis == "xz":
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        if axis == "xy":
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if axis == "yz":
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        raise ValueError(f"Unsupported table_axis for lift: {axis}")

    def _lift_offset(self, entry: Optional[Dict[str, Any]] = None) -> np.ndarray:
        entry = entry or self.initial_entry
        direction = self._table_up_vector(entry)
        distance = float(self.config.get("lift_distance", self.config.get("lift_after_grasp_distance", 0.0)))
        offset = direction * distance
        min_height = self.config.get("min_object_height_above_table", None)
        if min_height is not None:
            object_pose = as_vec(
                entry.get("object_scene_pose_wxyz", entry["object_pose_world_wxyz"]),
                7,
                "object pose",
            )
            table_pose = as_vec(entry.get("table_pose_wxyz", [0, 0, 0, 1, 0, 0, 0]), 7, "table_pose")
            clearance = float(np.dot(object_pose[:3] + offset - table_pose[:3], direction))
            extra = float(min_height) - clearance
            if extra > 0.0:
                offset = offset + direction * extra
        return offset

    def _pose_with_lift(self, pose, entry: Optional[Dict[str, Any]] = None) -> np.ndarray:
        pose = as_vec(pose, 7, "pose").copy()
        pose[:3] += self._lift_offset(entry)
        return pose

    def _build_model_xml(self) -> Path:
        model_dir = resolve_project_path(self.config.get("model_dir")) or DEFAULT_MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        friction = self._friction()
        object_density = self._object_density_from_config()
        xml_key = json.dumps(
            {
                "object": self.initial_entry.get("object_name"),
                "object_xml": self.initial_entry.get("object_xml_path"),
                "scale": self.initial_entry.get("object_scale"),
                "hand_pose": self.initial_entry.get("refine_hand_pose_world_wxyz"),
                "hand_xml": str(self.hand_xml),
                "collision_filter": self.config.get("collision_filter", "object_hand_table"),
                "initial_index": self.initial_index,
                "target_index": self.target_index,
                "lift_distance": float(self.config.get("lift_distance", self.config.get("lift_after_grasp_distance", 0.0))),
                "lift_direction": self.config.get("lift_direction", None),
                "min_object_height_above_table": self.config.get("min_object_height_above_table", None),
                "include_table": bool(self.config.get("include_table", True)),
                "target_pose": self._target_object_pose_from_entries().round(9).tolist(),
                "integrator": self.config.get("integrator", "implicitfast"),
                "timestep": float(self.config.get("timestep", 0.004)),
                "noslip_iterations": int(self.config.get("noslip_iterations", 2)),
                "impratio": float(self.config.get("impratio", 10.0)),
                "friction": friction.round(9).tolist(),
                "object_density": round(float(object_density), 9),
                "object_margin": float(self.config.get("object_margin", 0.001)),
                "hand_margin": float(self.config.get("hand_margin", 0.001)),
                "plane_margin": float(self.config.get("plane_margin", 0.002)),
                "actuator_kp_scale": float(self.config.get("actuator_kp_scale", 1.0)),
                "actuator_kp": self.config.get("actuator_kp", self.config.get("actuator_kp_override", None)),
                "actuator_force_scale": float(self.config.get("actuator_force_scale", 1.0)),
                "actuator_forcerange": self.config.get(
                    "actuator_forcerange", self.config.get("actuator_force_range", None)
                ),
                "schema": 6,
            },
            sort_keys=True,
        )
        xml_hash = hashlib.sha1(xml_key.encode("utf-8")).hexdigest()[:12]
        xml_path = model_dir / f"refine_tabletop_{xml_hash}.xml"
        if xml_path.exists() and not bool(self.config.get("force_rebuild_xml", False)):
            return xml_path

        hand_root = ET.parse(self.hand_xml).getroot()
        self._apply_hand_actuator_tuning(hand_root)
        object_xml = Path(self.initial_entry["object_xml_path"])
        object_root = ET.parse(object_xml).getroot()

        root = ET.Element("mujoco", {"model": "refine_tabletop_fixed_base"})
        compiler = copy.deepcopy(hand_root.find("compiler"))
        if compiler is None:
            compiler = ET.Element("compiler", {"angle": "radian", "autolimits": "true"})
        hand_meshdir = self.hand_xml.parent / compiler.get("meshdir", ".")
        compiler.set("meshdir", str(hand_meshdir.resolve()))
        root.append(compiler)
        ET.SubElement(
            root,
            "option",
            {
                "timestep": str(float(self.config.get("timestep", 0.004))),
                "integrator": str(self.config.get("integrator", "implicitfast")),
                "cone": "elliptic",
                "impratio": str(float(self.config.get("impratio", 10.0))),
                "noslip_iterations": str(int(self.config.get("noslip_iterations", 2))),
                "gravity": vec_str(self.config.get("gravity_vector", [0.0, -9.81, 0.0])),
            },
        )
        ET.SubElement(root, "size", {"njmax": "1000", "nconmax": "300"})

        for child in xml_children(hand_root, "default"):
            default_root = root.find("default")
            if default_root is None:
                default_root = ET.SubElement(root, "default")
            default_root.append(copy.deepcopy(child))

        asset = ET.SubElement(root, "asset")
        for child in xml_children(hand_root, "asset"):
            asset.append(copy.deepcopy(child))
        object_mesh_prefix = "objmesh_"
        object_meshdir = object_xml.parent / object_root.find("compiler").get("meshdir", ".")
        object_mesh_scale = as_vec(self.initial_entry.get("object_scale", [1.0, 1.0, 1.0]), 3, "object_scale")
        mesh_name_map = {}
        for mesh in object_root.findall("./asset/mesh"):
            mesh_copy = copy.deepcopy(mesh)
            old_name = mesh_copy.get("name")
            new_name = object_mesh_prefix + old_name
            mesh_name_map[old_name] = new_name
            mesh_copy.set("name", new_name)
            mesh_copy.set("file", str((object_meshdir / mesh_copy.get("file")).resolve()))
            mesh_copy.set("scale", vec_str(object_mesh_scale))
            asset.append(mesh_copy)

        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "light", {"pos": "0 -1 1", "dir": "0 1 -1", "diffuse": "0.8 0.8 0.8"})
        ET.SubElement(worldbody, "camera", {"name": "fixed", "pos": "0 -0.55 0.55", "euler": "0.8 0 0"})
        if bool(self.config.get("include_table", True)):
            table_pose = as_vec(self.initial_entry.get("table_pose_wxyz", [0, 0, 0, 1, 0, 0, 0]), 7, "table_pose")
            table_size = as_vec(self.initial_entry.get("table_size", [0.34, 0.34, 0.01]), 3, "table_size")
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": "table",
                    "type": "plane",
                    "pos": vec_str(table_pose[:3]),
                    "quat": vec_str(table_pose[3:7]),
                    "size": vec_str(table_size),
                    "friction": vec_str(friction),
                    "condim": "4",
                    "margin": str(float(self.config.get("plane_margin", 0.002))),
                    "contype": "4",
                    "conaffinity": "2",
                },
            )

        hand_pose = self._pose_with_lift(
            as_vec(self.initial_entry["refine_hand_pose_world_wxyz"], 7, "hand pose"),
            self.initial_entry,
        )
        fixed_hand = ET.SubElement(
            worldbody,
            "body",
            {
                "name": "fixed_hand_base",
                "pos": vec_str(hand_pose[:3]),
                "quat": vec_str(hand_pose[3:7]),
            },
        )
        for child in xml_children(hand_root, "worldbody"):
            hand_child = copy.deepcopy(child)
            self._apply_hand_collision_filter(hand_child)
            fixed_hand.append(hand_child)

        object_pose = self._pose_with_lift(
            as_vec(
                self.initial_entry.get("object_scene_pose_wxyz", self.initial_entry["object_pose_world_wxyz"]),
                7,
                "object pose",
            ),
            self.initial_entry,
        )
        object_body = ET.SubElement(
            worldbody,
            "body",
            {"name": "Object", "pos": vec_str(object_pose[:3]), "quat": vec_str(object_pose[3:7])},
        )
        ET.SubElement(object_body, "freejoint", {"name": "object_freejoint"})
        density = str(float(object_density))
        for geom in object_root.findall("./worldbody/body/geom"):
            geom_copy = copy.deepcopy(geom)
            if geom_copy.get("name"):
                geom_copy.set("name", "obj_" + geom_copy.get("name"))
            if geom_copy.get("mesh") in mesh_name_map:
                geom_copy.set("mesh", mesh_name_map[geom_copy.get("mesh")])
            if geom_copy.get("contype", "1") != "0":
                geom_copy.set("density", density)
                geom_copy.set("condim", "4")
                geom_copy.set("friction", vec_str(friction))
                geom_copy.set("margin", str(float(self.config.get("object_margin", 0.001))))
                geom_copy.set("contype", "2")
                geom_copy.set("conaffinity", "5")
            object_body.append(geom_copy)

        target_pose = self._target_object_pose_from_entries()
        ET.SubElement(
            worldbody,
            "site",
            {
                "name": "target",
                "type": "sphere",
                "pos": vec_str(target_pose[:3]),
                "size": "0.015",
                "rgba": "0 1 0 0.5",
            },
        )

        for section_name in ["contact", "tendon", "actuator"]:
            section = hand_root.find(section_name)
            if section is not None:
                root.append(copy.deepcopy(section))

        ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
        return xml_path

    def _apply_hand_actuator_tuning(self, hand_root: ET.Element) -> None:
        kp_scale = float(self.config.get("actuator_kp_scale", 1.0))
        kp_override = self.config.get("actuator_kp", self.config.get("actuator_kp_override", None))
        force_scale = float(self.config.get("actuator_force_scale", 1.0))
        force_override = as_optional_range(
            self.config.get("actuator_forcerange", self.config.get("actuator_force_range", None)),
            "actuator_forcerange",
        )
        if kp_scale == 1.0 and kp_override is None and force_scale == 1.0 and force_override is None:
            return

        for actuator in hand_root.iter("position"):
            if kp_override is not None:
                actuator.set("kp", f"{float(kp_override):.9g}")
            elif actuator.get("kp") is not None and kp_scale != 1.0:
                actuator.set("kp", f"{float(actuator.get('kp')) * kp_scale:.9g}")

            if force_override is not None:
                actuator.set("forcerange", vec_str(force_override))
            elif actuator.get("forcerange") is not None and force_scale != 1.0:
                force_range = as_optional_range(actuator.get("forcerange"), "forcerange")
                actuator.set("forcerange", vec_str(force_range * force_scale))

    def _friction(self) -> np.ndarray:
        if self.config.get("friction") is not None:
            return as_vec(self.config["friction"], 3, "friction")
        miu_coef = as_vec(self.config.get("miu_coef", [0.6, 0.02]), 2, "miu_coef")
        roll = float(self.config.get("friction_roll", 0.0001))
        return np.asarray([miu_coef[0], miu_coef[1], roll], dtype=np.float64)

    def _object_density_from_config(self) -> float:
        if self.config.get("object_mass") is None:
            return float(self.config.get("object_density", 1000.0))

        obj_mass = float(self.config.get("object_mass"))
        fallback_density = float(self.config.get("object_density", 1000.0))
        info_path = self.initial_entry.get("object_info_path", None)
        if not info_path or not Path(info_path).exists():
            return fallback_density

        info = load_json(Path(info_path))
        info_scale = first_scalar(info.get("scale", 1.0), 1.0)
        base_mass = first_scalar(info.get("mass", obj_mass), obj_mass)
        base_density = first_scalar(info.get("density", fallback_density), fallback_density)
        coef = base_mass / max(base_density * info_scale**3, 1e-12)
        object_scale = as_vec(self.initial_entry.get("object_scale", [1.0, 1.0, 1.0]), 3, "object_scale")
        return float(obj_mass / max(coef * float(np.prod(object_scale)), 1e-12))

    def _apply_hand_collision_filter(self, body: ET.Element) -> None:
        if self.config.get("collision_filter", "object_hand_table") == "default":
            return
        for geom in body.iter("geom"):
            geom_class = str(geom.get("class", ""))
            if "visual" in geom_class or geom.get("contype") == "0":
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
            else:
                geom.set("contype", "1")
                geom.set("conaffinity", "2")
                geom.set("condim", "4")
                geom.set("friction", vec_str(self._friction()))
                geom.set("margin", str(float(self.config.get("hand_margin", 0.001))))
                geom.set("solimp", "0.5 0.99 0.0001")
                geom.set("solref", "0.005 1")

    def _target_object_pose_from_entries(self) -> np.ndarray:
        if self.config.get("target_object_pose_wxyz") is not None:
            return as_vec(self.config["target_object_pose_wxyz"], 7, "target_object_pose_wxyz")
        pose = self._pose_with_lift(
            as_vec(
                self.target_entry.get("object_scene_pose_wxyz", self.target_entry["object_pose_world_wxyz"]),
                7,
                "target object pose",
            ),
            self.initial_entry,
        )
        if self.config.get("target_relative_axis") is not None:
            axis = as_vec(self.config.get("target_relative_axis"), 3, "target_relative_axis")
            angle = float(self.config.get("target_relative_angle", 0.0))
            pose[3:7] = quat_mul(axis_angle_quat(axis, angle), pose[3:7])
        return pose

    def _cache_ids(self):
        self.hand_qpos_addr = np.asarray([self._joint_qpos_addr(name) for name in REFINE_JOINT_NAMES], dtype=np.int32)
        self.hand_qvel_addr = np.asarray([self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in REFINE_JOINT_NAMES], dtype=np.int32)
        self.object_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
        self.object_qpos_addr = int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_qvel_addr = int(self.model.jnt_dofadr[self.object_joint_id])
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Object")
        self.fixed_hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "fixed_hand_base")
        self.object_geom_ids = self._descendant_geom_ids(self.object_body_id)
        self.hand_geom_ids = self._descendant_geom_ids(self.fixed_hand_body_id)
        self.hand_geom_group_by_id = self._hand_geom_groups()
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES]

    def _descendant_geom_ids(self, root_body_id: int) -> set:
        geom_ids = set()
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            while body_id >= 0:
                if body_id == root_body_id:
                    geom_ids.add(geom_id)
                    break
                parent_id = int(self.model.body_parentid[body_id])
                if parent_id == body_id:
                    break
                body_id = parent_id
        return geom_ids

    def _hand_geom_groups(self) -> Dict[int, str]:
        groups = {}
        for geom_id in self.hand_geom_ids:
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            body_name = body_name.lower()
            group = "other"
            for candidate, prefixes in FINGER_GROUP_PREFIXES.items():
                if any(body_name.startswith(prefix) for prefix in prefixes):
                    group = candidate
                    break
            groups[int(geom_id)] = group
        return groups

    def _joint_qpos_addr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"Missing joint in model: {name}")
        return int(self.model.jnt_qposadr[joint_id])

    def _object_pose(self) -> np.ndarray:
        return self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 7].copy()

    def _object_vel(self) -> np.ndarray:
        return self.data.qvel[self.object_qvel_addr : self.object_qvel_addr + 6].copy()

    def _get_obs(self) -> np.ndarray:
        hand_qpos = self.data.qpos[self.hand_qpos_addr].copy()
        hand_qvel = self.data.qvel[self.hand_qvel_addr].copy()
        object_pose = self._object_pose()
        object_vel = self._object_vel()
        pos_err = object_pose[:3] - self.target_object_pose[:3]
        rot_err = np.asarray([quat_error_angle(object_pose[3:7], self.target_object_pose[3:7])], dtype=np.float64)
        hand_err = hand_qpos - self.target_hand_qpos
        contact_count = np.asarray([float(self._object_hand_contact_count())], dtype=np.float64)
        obs_parts = [
            hand_qpos,
            hand_qvel,
            object_pose,
            object_vel,
            self.target_object_pose,
            pos_err,
            rot_err,
            hand_err,
            contact_count,
        ]
        if self.config.get("include_contact_features_in_obs", False):
            features = self._contact_features()
            obs_parts.append(np.asarray([float(features[key]) for key in CONTACT_FEATURE_KEYS], dtype=np.float64))
        return np.concatenate(obs_parts).astype(np.float32)

    def _initial_ctrl_from_qpos(self, hand_qpos: np.ndarray) -> np.ndarray:
        qpos_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()
        self.data.qpos[:] = qpos_backup
        self.data.qpos[self.hand_qpos_addr] = as_vec(hand_qpos, len(REFINE_JOINT_NAMES), "hand_qpos")
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        for actuator_id in range(self.model.nu):
            trn_type = int(self.model.actuator_trntype[actuator_id])
            trn_id = int(self.model.actuator_trnid[actuator_id, 0])
            if trn_type == int(mujoco.mjtTrn.mjTRN_JOINT):
                qpos_addr = int(self.model.jnt_qposadr[trn_id])
                ctrl[actuator_id] = self.data.qpos[qpos_addr]
            elif trn_type == int(mujoco.mjtTrn.mjTRN_TENDON):
                ctrl[actuator_id] = self.data.ten_length[trn_id]
            else:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
                raise ValueError(f"Unsupported actuator transmission for {name}: {trn_type}")

        self.data.qpos[:] = qpos_backup
        self.data.qvel[:] = qvel_backup
        mujoco.mj_forward(self.model, self.data)
        return self._clip_ctrl(ctrl)

    def _clip_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        ctrl = np.asarray(ctrl, dtype=np.float64).copy()
        ctrl[self.ctrl_limited] = np.clip(
            ctrl[self.ctrl_limited],
            self.ctrl_low[self.ctrl_limited],
            self.ctrl_high[self.ctrl_limited],
        )
        return ctrl

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        if self.sample_mode == "random_reset":
            self._select_entries(initial=False)
            self._load_model_for_current_entries()
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.data.qpos[self.hand_qpos_addr] = self.init_hand_qpos
        self.data.qpos[self.object_qpos_addr : self.object_qpos_addr + 7] = self.init_object_pose
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_ctrl
        self._last_action_norm = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action = np.clip(action, -1.0, 1.0)
        self._last_action_norm = float(np.mean(np.square(action))) if action.size else 0.0
        if self.action_mode == "absolute":
            ctrl = self.default_ctrl + self.action_scale * action
            ctrl[self.ctrl_limited] = (
                self.ctrl_low[self.ctrl_limited]
                + 0.5
                * (action[self.ctrl_limited] + 1.0)
                * (self.ctrl_high[self.ctrl_limited] - self.ctrl_low[self.ctrl_limited])
            )
        elif self.action_mode == "delta":
            ctrl = self.default_ctrl + self.action_scale * action
        else:
            raise ValueError("action_mode must be 'delta' or 'absolute'")
        ctrl = self._clip_ctrl(ctrl)
        self.data.ctrl[:] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        obs = self._get_obs()
        info = self._info()
        terminated = bool(info["dropped"] and self.config.get("done_on_drop", True))
        truncated = self.step_count >= self.horizon
        return obs, float(info["reward"]), terminated, truncated, info

    def _info(self) -> Dict[str, Any]:
        object_pose = self._object_pose()
        hand_qpos = self.data.qpos[self.hand_qpos_addr].copy()
        pos_err = float(np.linalg.norm(object_pose[:3] - self.target_object_pose[:3]))
        rot_err = float(quat_error_angle(object_pose[3:7], self.target_object_pose[3:7]))
        hand_err = float(np.linalg.norm(hand_qpos - self.target_hand_qpos) / math.sqrt(len(hand_qpos)))
        contact_features = self._contact_features()
        contacts = int(contact_features["contact_count"])
        dropped = self._dropped(object_pose)
        reward = self._reward(pos_err, rot_err, hand_err, contact_features, dropped)
        success_min_contacts = int(self.config.get("success_min_contacts", self.config.get("min_contact_count", 0)))
        success_min_thumb_contacts = int(self.config.get("success_min_thumb_contacts", 0))
        success_min_opposing_contacts = int(self.config.get("success_min_opposing_contacts", 0))
        success_min_opposing_groups = int(self.config.get("success_min_opposing_finger_groups", 0))
        success_opposition_threshold = float(self.config.get("success_opposition_threshold", -1.0))
        solved = (
            pos_err < float(self.config.get("success_pos_threshold", 0.03))
            and rot_err < float(self.config.get("success_rot_threshold", 0.35))
            and contacts >= success_min_contacts
            and int(contact_features["thumb_contact_count"]) >= success_min_thumb_contacts
            and int(contact_features["opposing_contact_count"]) >= success_min_opposing_contacts
            and int(contact_features["opposing_finger_groups"]) >= success_min_opposing_groups
            and float(contact_features["opposition_score"]) >= success_opposition_threshold
            and not dropped
        )
        return {
            "reward": reward,
            "pos_err": pos_err,
            "rot_err": rot_err,
            "hand_err": hand_err,
            "contact_count": contacts,
            "thumb_contact_count": int(contact_features["thumb_contact_count"]),
            "opposing_contact_count": int(contact_features["opposing_contact_count"]),
            "opposing_finger_groups": int(contact_features["opposing_finger_groups"]),
            "palm_contact_count": int(contact_features["palm_contact_count"]),
            "opposition_score": float(contact_features["opposition_score"]),
            "has_thumb_opposition": bool(contact_features["has_thumb_opposition"]),
            "action_norm": float(getattr(self, "_last_action_norm", 0.0)),
            "action_scale": float(self.action_scale),
            "hand_weight": float(self.config.get("hand_weight", 0.05)),
            "dropped": dropped,
            "solved": solved,
            "fixed_base_pose_wxyz": self.fixed_base_pose.tolist(),
            "initial_grasp_index": self.initial_index,
            "target_grasp_index": self.target_index,
            "mode": self.initial_entry.get("mode", ""),
            "lift_offset": self._lift_offset(self.initial_entry).tolist(),
            "object_name": self.initial_entry.get("object_name", ""),
            "object_density": float(self._object_density_from_config()),
            "object_model_mass": float(self.model.body_subtreemass[self.object_body_id]),
            "model_xml_path": str(self.model_xml_path),
        }

    def _reward(self, pos_err, rot_err, hand_err, contact_features, dropped) -> float:
        contacts = int(contact_features["contact_count"])
        thumb_contacts = int(contact_features["thumb_contact_count"])
        opposing_contacts = int(contact_features["opposing_contact_count"])
        opposing_groups = int(contact_features["opposing_finger_groups"])
        opposition_score = float(contact_features["opposition_score"])
        reward = 0.0
        reward -= float(self.config.get("pos_weight", 10.0)) * pos_err
        reward -= float(self.config.get("rot_weight", 1.0)) * rot_err
        reward -= float(self.config.get("hand_weight", 0.05)) * hand_err
        reward += float(self.config.get("contact_bonus", 0.02)) * min(contacts, 5)
        reward += float(self.config.get("thumb_contact_bonus", 0.0)) * min(thumb_contacts, 3)
        reward += float(self.config.get("opposing_contact_bonus", 0.0)) * min(opposing_contacts, 5)
        reward += float(self.config.get("opposing_finger_group_bonus", 0.0)) * opposing_groups
        if thumb_contacts > 0 and opposing_contacts > 0:
            reward += float(self.config.get("opposition_bonus", 0.0)) * opposition_score
        elif contacts == 0:
            reward -= float(self.config.get("no_contact_penalty", 0.0))
        min_contacts = int(self.config.get("min_contact_count", 0))
        if min_contacts > 0 and contacts < min_contacts:
            reward -= float(self.config.get("contact_loss_penalty", 0.0)) * (min_contacts - contacts)
        min_thumb_contacts = int(self.config.get("min_thumb_contacts", 0))
        if min_thumb_contacts > 0 and thumb_contacts < min_thumb_contacts:
            reward -= float(self.config.get("thumb_contact_loss_penalty", 0.0)) * (
                min_thumb_contacts - thumb_contacts
            )
        min_opposing_contacts = int(self.config.get("min_opposing_contacts", 0))
        if min_opposing_contacts > 0 and opposing_contacts < min_opposing_contacts:
            reward -= float(self.config.get("opposing_contact_loss_penalty", 0.0)) * (
                min_opposing_contacts - opposing_contacts
            )
        min_opposing_groups = int(self.config.get("min_opposing_finger_groups", 0))
        if min_opposing_groups > 0 and opposing_groups < min_opposing_groups:
            reward -= float(self.config.get("opposing_group_loss_penalty", 0.0)) * (
                min_opposing_groups - opposing_groups
            )
        opposition_threshold = float(self.config.get("min_opposition_score", -1.0))
        if thumb_contacts > 0 and opposing_contacts > 0 and opposition_score < opposition_threshold:
            reward -= float(self.config.get("opposition_loss_penalty", 0.0)) * (
                opposition_threshold - opposition_score
            )
        reward += float(self.config.get("alive_bonus", 0.0))
        reward -= float(self.config.get("action_weight", 0.0)) * float(getattr(self, "_last_action_norm", 0.0))
        success_min_contacts = int(self.config.get("success_min_contacts", min_contacts))
        success_min_thumb_contacts = int(self.config.get("success_min_thumb_contacts", min_thumb_contacts))
        success_min_opposing_contacts = int(self.config.get("success_min_opposing_contacts", min_opposing_contacts))
        success_min_opposing_groups = int(self.config.get("success_min_opposing_finger_groups", min_opposing_groups))
        success_opposition_threshold = float(self.config.get("success_opposition_threshold", opposition_threshold))
        if (
            pos_err < float(self.config.get("success_pos_threshold", 0.03))
            and rot_err < float(self.config.get("success_rot_threshold", 0.35))
            and contacts >= success_min_contacts
            and thumb_contacts >= success_min_thumb_contacts
            and opposing_contacts >= success_min_opposing_contacts
            and opposing_groups >= success_min_opposing_groups
            and opposition_score >= success_opposition_threshold
            and not dropped
        ):
            reward += float(self.config.get("success_bonus", 0.0))
        if dropped:
            reward -= float(self.config.get("drop_penalty", 10.0))
        return float(reward)

    def _contact_features(self) -> Dict[str, Any]:
        counts = {group: 0 for group in ["thumb", *OPPOSING_FINGER_GROUPS, "palm", "other"]}
        points = {group: [] for group in counts}

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self.object_geom_ids and geom2 in self.hand_geom_ids:
                hand_geom = geom2
            elif geom2 in self.object_geom_ids and geom1 in self.hand_geom_ids:
                hand_geom = geom1
            else:
                continue
            group = self.hand_geom_group_by_id.get(hand_geom, "other")
            counts[group] = counts.get(group, 0) + 1
            points.setdefault(group, []).append(np.asarray(contact.pos, dtype=np.float64).copy())

        thumb_count = counts.get("thumb", 0)
        opposing_count = sum(counts.get(group, 0) for group in OPPOSING_FINGER_GROUPS)
        opposing_groups = sum(1 for group in OPPOSING_FINGER_GROUPS if counts.get(group, 0) > 0)
        palm_count = counts.get("palm", 0)
        total_count = thumb_count + opposing_count + palm_count + counts.get("other", 0)

        opposition_score = 0.0
        if thumb_count > 0 and opposing_count > 0:
            thumb_center = np.mean(np.asarray(points["thumb"], dtype=np.float64), axis=0)
            opposing_points = [point for group in OPPOSING_FINGER_GROUPS for point in points.get(group, [])]
            opposing_center = np.mean(np.asarray(opposing_points, dtype=np.float64), axis=0)
            object_center = np.asarray(self.data.xpos[self.object_body_id], dtype=np.float64)
            thumb_vec = thumb_center - object_center
            opposing_vec = opposing_center - object_center
            thumb_norm = np.linalg.norm(thumb_vec)
            opposing_norm = np.linalg.norm(opposing_vec)
            if thumb_norm > 1e-8 and opposing_norm > 1e-8:
                cosine = float(np.dot(thumb_vec, opposing_vec) / (thumb_norm * opposing_norm))
                opposition_score = float(np.clip(-cosine, 0.0, 1.0))

        opposition_threshold = float(self.config.get("thumb_opposition_threshold", 0.25))
        has_thumb_opposition = bool(
            thumb_count > 0 and opposing_count > 0 and opposition_score >= opposition_threshold
        )

        return {
            "contact_count": int(total_count),
            "thumb_contact_count": int(thumb_count),
            "opposing_contact_count": int(opposing_count),
            "opposing_finger_groups": int(opposing_groups),
            "palm_contact_count": int(palm_count),
            "opposition_score": float(opposition_score),
            "has_thumb_opposition": has_thumb_opposition,
        }

    def _object_hand_contact_count(self) -> int:
        return int(self._contact_features()["contact_count"])

    def _dropped(self, object_pose) -> bool:
        axis = str(self.initial_entry.get("table_axis", "xz"))
        up_index = 1 if axis == "xz" else 2
        drop_reference = str(self.config.get("drop_reference", "table")).lower()
        if drop_reference == "initial":
            reference_pose = self.init_object_pose
        elif drop_reference == "table":
            reference_pose = as_vec(self.initial_entry.get("table_pose_wxyz", [0, 0, 0, 1, 0, 0, 0]), 7, "table_pose")
        else:
            raise ValueError("drop_reference must be one of: table, initial")
        return bool(object_pose[up_index] < reference_pose[up_index] - float(self.config.get("drop_margin", 0.03)))

    def set_gravity_vector(self, gravity_vector, gravity_scale=None):
        self._base_gravity = as_vec(gravity_vector, 3, "gravity_vector")
        if gravity_scale is not None:
            self._gravity_scale = float(gravity_scale)
        self.set_gravity_scale(getattr(self, "_gravity_scale", 1.0))

    def set_gravity_scale(self, gravity_scale):
        self._gravity_scale = float(gravity_scale)
        self.model.opt.gravity[:] = self._base_gravity * self._gravity_scale
        return self._gravity_scale

    def set_runtime_param(self, name, value):
        value = float(value)
        self.config[str(name)] = value
        if str(name) == "action_scale":
            self.action_scale = value
        return value

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def mj_render(self):
        return self.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def make_refine_tabletop_env(config: Dict[str, Any], seed: Optional[int] = None) -> RefineTabletopEnv:
    return RefineTabletopEnv(config=config, seed=seed)
