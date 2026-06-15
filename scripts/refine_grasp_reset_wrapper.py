import json
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium
import numpy as np
from robohive.utils.quat_math import mat2quat, quat2euler, quat2mat


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

BASE_JOINT_NAMES = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"]
OBJECT_TRANSLATION_JOINTS = ["OBJTx", "OBJTy", "OBJTz"]
OBJECT_ROTATION_JOINTS = ["OBJRx", "OBJRy", "OBJRz"]

AXIS_TRANSFORMS = {
    "identity": np.eye(3, dtype=np.float64),
    # refine tabletop gravity is [0, -9.81, 0]. This maps it to [0, 0, -9.81].
    "refine_y_up_to_robo_z_up": np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    ),
}


def resolve_project_path(path: Optional[str]) -> Optional[Path]:
    if path is None or str(path) == "":
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_vec(value, size, name, dtype=np.float64):
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name} must have {size} values, got {arr.size}")
    return arr


def quat_normalize(quat):
    quat = as_vec(quat, 4, "quat")
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quat = quat / norm
    return quat if quat[0] >= 0 else -quat


class RefineGraspResetWrapper(gymnasium.Wrapper):
    """Inject stable Grasp_Refine tabletop grasps into RoboHive reset states.

    This wrapper is intentionally conservative: it does not replace the RoboHive
    MJCF object mesh. For `relocate-v1`, the refine object state is projected to
    the built-in sphere, while the hand-object relative pose and 24 Adroit hand
    qpos are initialized from the filtered refine grasp dataset.
    """

    def __init__(self, env, config: Dict[str, Any], seed: Optional[int] = None):
        super().__init__(env)
        self.config = dict(config)
        self.rng = np.random.RandomState(seed if seed is not None else self.config.get("seed", 0))
        self.sample_mode = str(self.config.get("sample_mode", "random")).lower()
        self.fixed_index = int(self.config.get("fixed_index", 0))
        self._cycle_index = 0
        self.last_grasp_index = None

        manifest_path = resolve_project_path(self.config.get("manifest_path"))
        npz_path = resolve_project_path(self.config.get("npz_path"))
        if manifest_path is None and npz_path is None:
            raise ValueError("refine_grasp_reset requires manifest_path or npz_path")
        self.manifest = load_json(manifest_path) if manifest_path is not None else {}
        if npz_path is None:
            npz_path = resolve_project_path(self.manifest.get("npz_path"))
        if npz_path is None:
            raise ValueError("Could not resolve refine grasp npz_path")

        data = np.load(npz_path, allow_pickle=True)
        self.refine_qpos29 = data["refine_qpos29"].astype(np.float64)
        self.adroit_hand_qpos24 = data["adroit_hand_qpos24"].astype(np.float64)
        self.object_pose_wxyz = data["object_pose_wxyz"].astype(np.float64)
        self.num_grasps = int(self.refine_qpos29.shape[0])
        if self.num_grasps <= 0:
            raise ValueError(f"No refine grasps found in {npz_path}")

        self.axis_transform = self._axis_transform(self.config.get("coordinate_frame"))
        self.object_position = self._default_object_position()
        self.object_position_noise = as_vec(
            self.config.get("object_position_noise", [0.0, 0.0, 0.0]),
            3,
            "object_position_noise",
        )
        self.hand_object_offset_scale = float(self.config.get("hand_object_offset_scale", 1.0))
        self.hand_object_offset_clip = self._optional_vec(
            self.config.get("hand_object_offset_clip", None), 3, "hand_object_offset_clip"
        )
        self.base_euler_offset = as_vec(
            self.config.get("base_euler_offset", [0.0, 0.0, 0.0]),
            3,
            "base_euler_offset",
        )
        self.base_euler_override = self._optional_vec(
            self.config.get("base_euler_override", None), 3, "base_euler_override"
        )
        self.base_translation_initial = as_vec(
            self.config.get("base_translation_initial", [0.0, 0.0, 0.0]),
            3,
            "base_translation_initial",
        )
        self.orientation_transform_mode = str(
            self.config.get("orientation_transform_mode", "active")
        ).lower()
        self.align_palm_site = bool(self.config.get("align_palm_site", True))
        self.palm_site_name = str(self.config.get("palm_site_name", "S_grasp"))
        self.palm_offset_override = self._optional_vec(
            self.config.get("palm_object_offset_override", None),
            3,
            "palm_object_offset_override",
        )
        self.palm_offset_extra = as_vec(
            self.config.get("palm_object_offset_extra", [0.0, 0.0, 0.0]),
            3,
            "palm_object_offset_extra",
        )
        self.target_position = self._optional_vec(
            self.config.get("target_position", None), 3, "target_position"
        )
        self.target_offset = self._optional_vec(
            self.config.get("target_offset_from_object", None),
            3,
            "target_offset_from_object",
        )
        self.set_object_rotation = bool(self.config.get("set_object_rotation", True))
        self.zero_qvel = bool(self.config.get("zero_qvel", True))
        self.clip_to_joint_range = bool(self.config.get("clip_to_joint_range", True))
        self.hold_initial_ctrl = bool(self.config.get("hold_initial_ctrl", True))
        self.settle_steps = int(self.config.get("settle_steps", 0))
        self.align_iterations = int(self.config.get("align_iterations", 4))
        self.align_tolerance = float(self.config.get("align_tolerance", 0.005))
        self.align_damping = float(self.config.get("align_damping", 1e-5))
        self.align_step_size = float(self.config.get("align_step_size", 1.0))

    @staticmethod
    def _optional_vec(value, size, name):
        if value is None:
            return None
        return as_vec(value, size, name)

    def _axis_transform(self, frame_name):
        frame_name = str(frame_name or "refine_y_up_to_robo_z_up")
        if frame_name not in AXIS_TRANSFORMS:
            raise ValueError(
                f"Unsupported coordinate_frame={frame_name!r}. "
                f"Supported: {sorted(AXIS_TRANSFORMS)}"
            )
        return AXIS_TRANSFORMS[frame_name]

    def _default_object_position(self):
        if self.config.get("object_position") is not None:
            return as_vec(self.config["object_position"], 3, "object_position")
        table_z = float(self.config.get("table_z", 0.0))
        object_radius = float(self.config.get("object_radius", 0.035))
        object_xy = as_vec(self.config.get("object_xy", [0.0, 0.0]), 2, "object_xy")
        return np.asarray([object_xy[0], object_xy[1], table_z + object_radius])

    def _sample_index(self):
        if self.sample_mode == "random":
            return int(self.rng.randint(self.num_grasps))
        if self.sample_mode == "cycle":
            idx = self._cycle_index % self.num_grasps
            self._cycle_index += 1
            return int(idx)
        if self.sample_mode == "fixed":
            return int(np.clip(self.fixed_index, 0, self.num_grasps - 1))
        raise ValueError("sample_mode must be one of: random, cycle, fixed")

    def _transform_vec(self, vec):
        return self.axis_transform @ as_vec(vec, 3, "vec")

    def _transform_quat(self, quat):
        quat = quat_normalize(quat)
        rot = quat2mat(quat)
        if self.orientation_transform_mode == "active":
            mapped = self.axis_transform @ rot
        elif self.orientation_transform_mode in {"basis", "sandwich"}:
            mapped = self.axis_transform @ rot @ self.axis_transform.T
        else:
            raise ValueError("orientation_transform_mode must be active or basis")
        return quat_normalize(mat2quat(mapped))

    def _joint_addr(self, model, joint_name):
        return int(model.jnt_qposadr[model.joint_name2id(joint_name)])

    def _joint_value(self, model, joint_name, value):
        jid = model.joint_name2id(joint_name)
        value = float(value)
        if self.clip_to_joint_range and bool(model.jnt_limited[jid]):
            low, high = model.jnt_range[jid]
            value = float(np.clip(value, low, high))
        return value

    def _set_joint(self, qpos, model, joint_name, value):
        qpos[self._joint_addr(model, joint_name)] = self._joint_value(model, joint_name, value)

    def _set_joint_group(self, qpos, model, joint_names, values):
        for joint_name, value in zip(joint_names, values):
            self._set_joint(qpos, model, joint_name, value)

    def _sample_object_position(self):
        noise = self.rng.uniform(-self.object_position_noise, self.object_position_noise)
        return self.object_position + noise

    def _reset_output(self, obs, info, base_returned_info):
        if base_returned_info:
            info = dict(info)
            info["refine_grasp_index"] = self.last_grasp_index
            return obs, info
        return obs

    def reset(self, *args, **kwargs):
        out = self.env.reset(*args, **kwargs)
        base_returned_info = isinstance(out, tuple) and len(out) == 2
        info = out[1] if base_returned_info else {}

        self.apply_refine_grasp()
        base_env = self.env.unwrapped
        obs = base_env.get_obs() if hasattr(base_env, "get_obs") else out[0]
        return self._reset_output(obs, info, base_returned_info)

    def apply_refine_grasp(self, grasp_index: Optional[int] = None):
        idx = self._sample_index() if grasp_index is None else int(grasp_index)
        idx = int(np.clip(idx, 0, self.num_grasps - 1))
        self.last_grasp_index = idx

        sim = self.env.unwrapped.sim
        model = sim.model
        qpos = sim.data.qpos.ravel().copy()
        qvel = np.zeros_like(sim.data.qvel.ravel()) if self.zero_qvel else sim.data.qvel.ravel().copy()

        refine_qpos = self.refine_qpos29[idx]
        refine_hand_pos = refine_qpos[:3]
        refine_hand_quat = refine_qpos[3:7]
        refine_obj_pos = self.object_pose_wxyz[idx, :3]
        refine_obj_quat = self.object_pose_wxyz[idx, 3:7]

        object_pos = self._sample_object_position()
        hand_object_offset = self._transform_vec(refine_hand_pos - refine_obj_pos)
        hand_object_offset *= self.hand_object_offset_scale
        if self.hand_object_offset_clip is not None:
            hand_object_offset = np.clip(
                hand_object_offset,
                -self.hand_object_offset_clip,
                self.hand_object_offset_clip,
            )

        base_euler = quat2euler(self._transform_quat(refine_hand_quat))
        if self.base_euler_override is not None:
            base_euler = self.base_euler_override.copy()
        base_euler = base_euler + self.base_euler_offset

        self._set_joint_group(qpos, model, BASE_JOINT_NAMES[:3], self.base_translation_initial)
        self._set_joint_group(qpos, model, BASE_JOINT_NAMES[3:], base_euler)
        self._set_joint_group(qpos, model, ADROIT_HAND_JOINT_NAMES, self.adroit_hand_qpos24[idx])
        self._set_joint_group(qpos, model, OBJECT_TRANSLATION_JOINTS, [0.0, 0.0, 0.0])
        if self.set_object_rotation:
            object_euler = quat2euler(self._transform_quat(refine_obj_quat))
            self._set_joint_group(qpos, model, OBJECT_ROTATION_JOINTS, object_euler)

        obj_bid = model.body_name2id("Object")
        model.body_pos[obj_bid] = object_pos
        sim.set_state(qpos=qpos, qvel=qvel)
        sim.forward()

        if self.align_palm_site:
            desired_offset = hand_object_offset
            if self.palm_offset_override is not None:
                desired_offset = self.palm_offset_override.copy()
            desired_palm = object_pos + desired_offset + self.palm_offset_extra
            self._align_site_to_position(sim, model, qvel, self.palm_site_name, desired_palm)

        if self.target_position is not None or self.target_offset is not None:
            target_sid = model.site_name2id("target")
            if self.target_position is not None:
                model.site_pos[target_sid] = self.target_position
            else:
                model.site_pos[target_sid] = object_pos + self.target_offset
            sim.forward()

        if self.hold_initial_ctrl:
            self._set_ctrl_from_qpos(qpos=sim.data.qpos.ravel().copy(), model=model, sim=sim)

        for _ in range(max(self.settle_steps, 0)):
            sim.step()
        return idx

    def _align_site_to_position(self, sim, model, qvel, site_name, desired_pos):
        site_id = model.site_name2id(site_name)
        desired_pos = as_vec(desired_pos, 3, "desired_pos")

        for _ in range(max(self.align_iterations, 0)):
            qpos = sim.data.qpos.ravel().copy()
            current = sim.data.site_xpos[site_id].copy()
            error = desired_pos - current
            if np.linalg.norm(error) <= self.align_tolerance:
                break

            jac = self._translation_jacobian(sim, model, qpos, qvel, site_id)
            lhs = jac.T @ jac + self.align_damping * np.eye(len(BASE_JOINT_NAMES[:3]))
            rhs = jac.T @ error
            delta_q = np.linalg.solve(lhs, rhs) * self.align_step_size

            for joint_name, delta in zip(BASE_JOINT_NAMES[:3], delta_q):
                addr = self._joint_addr(model, joint_name)
                self._set_joint(qpos, model, joint_name, qpos[addr] + delta)

            sim.set_state(qpos=qpos, qvel=qvel)
            sim.forward()

    def _translation_jacobian(self, sim, model, qpos, qvel, site_id):
        eps = float(self.config.get("align_fd_eps", 1e-4))
        base_pos = sim.data.site_xpos[site_id].copy()
        jac = np.zeros((3, len(BASE_JOINT_NAMES[:3])), dtype=np.float64)

        for col, joint_name in enumerate(BASE_JOINT_NAMES[:3]):
            qpert = qpos.copy()
            addr = self._joint_addr(model, joint_name)
            old_value = qpert[addr]
            self._set_joint(qpert, model, joint_name, old_value + eps)
            actual_eps = qpert[addr] - old_value
            if abs(actual_eps) < 1e-9:
                self._set_joint(qpert, model, joint_name, old_value - eps)
                actual_eps = qpert[addr] - old_value
            if abs(actual_eps) < 1e-9:
                continue
            sim.set_state(qpos=qpert, qvel=qvel)
            sim.forward()
            jac[:, col] = (sim.data.site_xpos[site_id] - base_pos) / actual_eps

        sim.set_state(qpos=qpos, qvel=qvel)
        sim.forward()
        return jac

    def _set_ctrl_from_qpos(self, qpos, model, sim):
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            qpos_addr = int(model.jnt_qposadr[joint_id])
            ctrl = float(qpos[qpos_addr])
            low, high = model.actuator_ctrlrange[actuator_id]
            sim.data.ctrl[actuator_id] = np.clip(ctrl, low, high)
