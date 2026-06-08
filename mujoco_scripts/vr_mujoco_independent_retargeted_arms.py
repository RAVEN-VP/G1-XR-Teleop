import json
import socket
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Allow this script to import ../common when run from the repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.teleop_logger import TeleopLogger, flatten_vec


XML_PATH = "mujoco_xml/g1_29dof_vrtest.xml"
UDP_PORT = 5005

# ---------------------------------------------------------------------
# Independent calibrated retargeting.
#
# Each controller controls only its matching robot wrist.
# The first valid controller pose becomes the human neutral pose.
# The robot neutral pose is a close, natural camera-holding pose.
# ---------------------------------------------------------------------

LEFT_ROBOT_NEUTRAL = np.array([0.34, 0.105, 0.98], dtype=float)
RIGHT_ROBOT_NEUTRAL = np.array([0.34, -0.105, 0.98], dtype=float)

# Controller delta -> robot wrist delta.
# Increase these for more range; reduce if it feels too sensitive.
RETARGET_SCALE = np.array([0.85, 0.85, 0.75], dtype=float)

# Per-arm workspace limits.
# These keep each wrist on its own side but still allow hands to come close.
LEFT_TARGET_MIN = np.array([0.10, 0.035, 0.72], dtype=float)
LEFT_TARGET_MAX = np.array([0.66, 0.34, 1.28], dtype=float)

RIGHT_TARGET_MIN = np.array([0.10, -0.34, 0.72], dtype=float)
RIGHT_TARGET_MAX = np.array([0.66, -0.035, 1.28], dtype=float)

# Optional vertical bias if the robot sits too high/low.
LEFT_TARGET_BIAS = np.array([0.0, 0.0, 0.0], dtype=float)
RIGHT_TARGET_BIAS = np.array([0.0, 0.0, 0.0], dtype=float)


RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

RIGHT_WRIST_BODY = "right_wrist_yaw_link"
RIGHT_ELBOW_BODY = "right_elbow_link"

LEFT_WRIST_BODY = "left_wrist_yaw_link"
LEFT_ELBOW_BODY = "left_elbow_link"

TELEOP_OBJECTS = [
    "teleop_cube",
    "teleop_cylinder",
    "teleop_handle",
]


# Right arm natural-ish rest pose.
RIGHT_ARM_REST_POSE = np.array([
    0.20,    # shoulder pitch
    -0.45,   # shoulder roll
    0.10,    # shoulder yaw
    0.55,    # elbow bend
    0.00,    # wrist roll
    0.00,    # wrist pitch
    0.00,    # wrist yaw
], dtype=float)

# Mirrored left arm natural-ish rest pose.
LEFT_ARM_REST_POSE = np.array([
    0.20,    # shoulder pitch
    0.45,    # shoulder roll
    -0.10,   # shoulder yaw
    0.55,    # elbow bend
    0.00,    # wrist roll
    0.00,    # wrist pitch
    0.00,    # wrist yaw
], dtype=float)


REST_BIAS_STRENGTH = 0.015

JOINT_WEIGHTS = np.array([
    0.85,  # shoulder pitch
    0.85,  # shoulder roll
    1.10,  # shoulder yaw
    0.55,  # elbow
    2.40,  # wrist roll
    2.40,  # wrist pitch
    2.40,  # wrist yaw
], dtype=float)

IK_MAX_ITERATIONS = 18
IK_DAMPING = 0.10
IK_STEP_SCALE = 0.28
TARGET_SMOOTHING = 0.12
ELBOW_WEIGHT = 0.40


# Workspace clamps.
RIGHT_TARGET_MIN = np.array([-0.05, -0.45, 0.55], dtype=float)
RIGHT_TARGET_MAX = np.array([0.75, 0.15, 1.45], dtype=float)

LEFT_TARGET_MIN = np.array([-0.05, -0.15, 0.55], dtype=float)
LEFT_TARGET_MAX = np.array([0.75, 0.45, 1.45], dtype=float)


# Elbow base guides.
RIGHT_ELBOW_BASE_GUIDE = np.array([0.00, -0.30, 0.90], dtype=float)
LEFT_ELBOW_BASE_GUIDE = np.array([0.00, 0.30, 0.90], dtype=float)


def get_mocap_id(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Could not find body: {body_name}")

    mocap_id = model.body_mocapid[body_id]
    if mocap_id < 0:
        raise RuntimeError(f"Body is not mocap: {body_name}")

    return mocap_id


def get_body_id(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Could not find body: {body_name}")
    return body_id


def get_joint_info(model, joint_names):
    qpos_ids = []
    dof_ids = []
    ranges = []

    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Could not find joint: {name}")

        qpos_id = model.jnt_qposadr[joint_id]
        dof_id = model.jnt_dofadr[joint_id]

        qpos_ids.append(qpos_id)
        dof_ids.append(dof_id)

        if model.jnt_limited[joint_id]:
            ranges.append(model.jnt_range[joint_id].copy())
        else:
            ranges.append(np.array([-np.pi, np.pi]))

        print(f"{name}: qpos={qpos_id}, dof={dof_id}, range={ranges[-1]}")

    return np.array(qpos_ids), np.array(dof_ids), np.array(ranges)


def joint_qpos_size(joint_type):
    # MuJoCo joint qpos sizes.
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def joint_dof_size(joint_type):
    # MuJoCo joint velocity/dof sizes.
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    return 1


def build_robot_lock_indices(model, allowed_joint_names, dynamic_object_prefixes=("teleop_",)):
    """
    Lock every robot joint except the upper-body arm joints.

    Dynamic teleop objects are excluded because they need to remain free bodies.
    This prevents MuJoCo physics from making the G1 legs/base move when mj_step()
    is used for object contacts.
    """
    allowed_joint_names = set(allowed_joint_names)

    qpos_indices = []
    dof_indices = []

    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""

        if joint_name in allowed_joint_names:
            continue

        if any(joint_name.startswith(prefix) for prefix in dynamic_object_prefixes):
            continue

        joint_type = model.jnt_type[joint_id]

        qpos_start = model.jnt_qposadr[joint_id]
        qpos_count = joint_qpos_size(joint_type)

        dof_start = model.jnt_dofadr[joint_id]
        dof_count = joint_dof_size(joint_type)

        qpos_indices.extend(range(qpos_start, qpos_start + qpos_count))
        dof_indices.extend(range(dof_start, dof_start + dof_count))

    return np.array(qpos_indices, dtype=int), np.array(dof_indices, dtype=int)


def lock_robot_lower_body_and_base(data, locked_qpos_ids, locked_dof_ids, locked_qpos_values):
    # Keep non-arm robot joints fixed at their startup values.
    data.qpos[locked_qpos_ids] = locked_qpos_values
    data.qvel[locked_dof_ids] = 0.0


def compute_elbow_guide(wrist_target, side):
    if side == "right":
        guide = RIGHT_ELBOW_BASE_GUIDE.copy()

        # Let elbow move slightly with the wrist target.
        guide[0] += 0.20 * (wrist_target[0] - 0.15)
        guide[1] += 0.15 * (wrist_target[1] + 0.28)
        guide[2] += 0.25 * (wrist_target[2] - 0.88)
        guide[2] = min(guide[2], wrist_target[2] - 0.08)

    elif side == "left":
        guide = LEFT_ELBOW_BASE_GUIDE.copy()

        # Mirrored version for left side.
        guide[0] += 0.20 * (wrist_target[0] - 0.15)
        guide[1] += 0.15 * (wrist_target[1] - 0.28)
        guide[2] += 0.25 * (wrist_target[2] - 0.88)
        guide[2] = min(guide[2], wrist_target[2] - 0.08)

    else:
        raise ValueError(f"Unknown side: {side}")

    return guide


def clamp_target(pos, side):
    if side == "right":
        return np.clip(pos, RIGHT_TARGET_MIN, RIGHT_TARGET_MAX)
    if side == "left":
        return np.clip(pos, LEFT_TARGET_MIN, LEFT_TARGET_MAX)

    raise ValueError(f"Unknown side: {side}")


def solve_arm_ik(
    model,
    data,
    wrist_body_id,
    elbow_body_id,
    qpos_ids,
    dof_ids,
    joint_ranges,
    rest_pose,
    wrist_target,
    side,
):
    wrist_target = clamp_target(wrist_target, side)
    elbow_target = compute_elbow_guide(wrist_target, side)

    for _ in range(IK_MAX_ITERATIONS):
        mujoco.mj_forward(model, data)

        wrist_pos = data.xpos[wrist_body_id].copy()
        elbow_pos = data.xpos[elbow_body_id].copy()

        wrist_error = wrist_target - wrist_pos
        elbow_error = elbow_target - elbow_pos

        if np.linalg.norm(wrist_error) < 0.012 and np.linalg.norm(elbow_error) < 0.035:
            break

        wrist_jacp = np.zeros((3, model.nv))
        wrist_jacr = np.zeros((3, model.nv))
        elbow_jacp = np.zeros((3, model.nv))
        elbow_jacr = np.zeros((3, model.nv))

        mujoco.mj_jacBody(model, data, wrist_jacp, wrist_jacr, wrist_body_id)
        mujoco.mj_jacBody(model, data, elbow_jacp, elbow_jacr, elbow_body_id)

        J_wrist = wrist_jacp[:, dof_ids]
        J_elbow = elbow_jacp[:, dof_ids] * ELBOW_WEIGHT

        J = np.vstack([J_wrist, J_elbow])
        error = np.concatenate([wrist_error, elbow_error * ELBOW_WEIGHT])

        # Joint weighting to avoid wrist joints doing all the solving.
        weighted_J = J / JOINT_WEIGHTS[None, :]

        A = weighted_J @ weighted_J.T + IK_DAMPING * IK_DAMPING * np.eye(J.shape[0])
        dq_weighted = weighted_J.T @ np.linalg.solve(A, error)
        dq = dq_weighted / JOINT_WEIGHTS

        # Keep close to rest pose when possible.
        current_q = data.qpos[qpos_ids].copy()
        rest_error = rest_pose - current_q
        dq += REST_BIAS_STRENGTH * rest_error

        data.qpos[qpos_ids] += IK_STEP_SCALE * dq

        # Clamp joint limits.
        for i, qid in enumerate(qpos_ids):
            lo, hi = joint_ranges[i]
            data.qpos[qid] = np.clip(data.qpos[qid], lo, hi)

    mujoco.mj_forward(model, data)
    return elbow_target


def get_object_body_ids(model, object_names):
    body_ids = {}

    for name in object_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise RuntimeError(f"Could not find teleop object body: {name}")

        body_ids[name] = body_id
        print(f"Teleop object {name}: body_id={body_id}")

    return body_ids


def add_object_fields(fieldnames, object_names):
    for name in object_names:
        fieldnames.extend([
            f"object_{name}_pos_x",
            f"object_{name}_pos_y",
            f"object_{name}_pos_z",
            f"object_{name}_quat_w",
            f"object_{name}_quat_x",
            f"object_{name}_quat_y",
            f"object_{name}_quat_z",
            f"object_{name}_linvel_x",
            f"object_{name}_linvel_y",
            f"object_{name}_linvel_z",
            f"object_{name}_angvel_x",
            f"object_{name}_angvel_y",
            f"object_{name}_angvel_z",
        ])


def add_object_state(row, model, data, object_body_ids):
    for name, body_id in object_body_ids.items():
        xpos = data.xpos[body_id].copy()
        xquat = data.xquat[body_id].copy()

        linear_vel = np.zeros(3)
        angular_vel = np.zeros(3)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            np.concatenate([angular_vel, linear_vel]),
            0,
        )

        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )

        angular_vel = velocity[0:3]
        linear_vel = velocity[3:6]

        row[f"object_{name}_pos_x"] = float(xpos[0])
        row[f"object_{name}_pos_y"] = float(xpos[1])
        row[f"object_{name}_pos_z"] = float(xpos[2])

        row[f"object_{name}_quat_w"] = float(xquat[0])
        row[f"object_{name}_quat_x"] = float(xquat[1])
        row[f"object_{name}_quat_y"] = float(xquat[2])
        row[f"object_{name}_quat_z"] = float(xquat[3])

        row[f"object_{name}_linvel_x"] = float(linear_vel[0])
        row[f"object_{name}_linvel_y"] = float(linear_vel[1])
        row[f"object_{name}_linvel_z"] = float(linear_vel[2])

        row[f"object_{name}_angvel_x"] = float(angular_vel[0])
        row[f"object_{name}_angvel_y"] = float(angular_vel[1])
        row[f"object_{name}_angvel_z"] = float(angular_vel[2])


def compute_independent_retargeted_targets(
    left_raw,
    right_raw,
    neutral_left,
    neutral_right,
):
    """
    Retarget VR controller movement to G1 wrist targets.

    Important:
    - left controller affects only left wrist
    - right controller affects only right wrist
    - no midpoint
    - no shared camera-handle target
    - no controller can move both arms
    """
    left_raw = np.array(left_raw, dtype=float)
    right_raw = np.array(right_raw, dtype=float)

    neutral_left = np.array(neutral_left, dtype=float)
    neutral_right = np.array(neutral_right, dtype=float)

    left_delta = left_raw - neutral_left
    right_delta = right_raw - neutral_right

    left_target = LEFT_ROBOT_NEUTRAL + LEFT_TARGET_BIAS + RETARGET_SCALE * left_delta
    right_target = RIGHT_ROBOT_NEUTRAL + RIGHT_TARGET_BIAS + RETARGET_SCALE * right_delta

    left_target = np.clip(left_target, LEFT_TARGET_MIN, LEFT_TARGET_MAX)
    right_target = np.clip(right_target, RIGHT_TARGET_MIN, RIGHT_TARGET_MAX)

    left_target = clamp_target(left_target, "left")
    right_target = clamp_target(right_target, "right")

    return left_target, right_target


def make_logger():
    joint_fields = []

    for name in LEFT_ARM_JOINTS:
        joint_fields.append(f"cmd_left_{name}")
        joint_fields.append(f"actual_left_{name}")

    for name in RIGHT_ARM_JOINTS:
        joint_fields.append(f"cmd_right_{name}")
        joint_fields.append(f"actual_right_{name}")

    fieldnames = [
        "timestamp",
        "elapsed",
        "packet_count",
        "deadman",
        "left_x", "left_y", "left_z",
        "right_x", "right_y", "right_z",
        "head_x", "head_y", "head_z",
        "left_raw_x", "left_raw_y", "left_raw_z",
        "right_raw_x", "right_raw_y", "right_raw_z",
        "left_target_x", "left_target_y", "left_target_z",
        "right_target_x", "right_target_y", "right_target_z",
        "left_wrist_x", "left_wrist_y", "left_wrist_z",
        "right_wrist_x", "right_wrist_y", "right_wrist_z",
        "left_elbow_target_x", "left_elbow_target_y", "left_elbow_target_z",
        "right_elbow_target_x", "right_elbow_target_y", "right_elbow_target_z",
        "left_error",
        "right_error",
    ]

    fieldnames += joint_fields
    add_object_fields(fieldnames, TELEOP_OBJECTS)

    metadata = {
        "script": "vr_mujoco_independent_retargeted_arms.py",
        "pipeline": "Quest controllers -> calibrated independent arm retargeting -> MuJoCo G1 elbow-guided IK",
        "xml_path": XML_PATH,
        "udp_port": UDP_PORT,
        "left_arm_joints": LEFT_ARM_JOINTS,
        "right_arm_joints": RIGHT_ARM_JOINTS,
        "target_smoothing": TARGET_SMOOTHING,
        "ik_max_iterations": IK_MAX_ITERATIONS,
        "ik_damping": IK_DAMPING,
        "ik_step_scale": IK_STEP_SCALE,
        "elbow_weight": ELBOW_WEIGHT,
        "right_target_min": RIGHT_TARGET_MIN.tolist(),
        "right_target_max": RIGHT_TARGET_MAX.tolist(),
        "left_target_min": LEFT_TARGET_MIN.tolist(),
        "left_target_max": LEFT_TARGET_MAX.tolist(),
        "teleop_objects": TELEOP_OBJECTS,
    }

    return TeleopLogger(
        root_dir=str(REPO_ROOT / "logs"),
        metadata=metadata,
        fieldnames=fieldnames,
    )


def add_joint_values(row, prefix, joint_names, qpos_ids, data):
    for name, qid in zip(joint_names, qpos_ids):
        row[f"{prefix}_{name}"] = float(data.qpos[qid])


model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

right_target_id = get_mocap_id(model, "vr_right_target")
left_target_id = get_mocap_id(model, "vr_left_target")
head_target_id = get_mocap_id(model, "vr_head_target")

object_body_ids = get_object_body_ids(model, TELEOP_OBJECTS)

right_wrist_body_id = get_body_id(model, RIGHT_WRIST_BODY)
right_elbow_body_id = get_body_id(model, RIGHT_ELBOW_BODY)

left_wrist_body_id = get_body_id(model, LEFT_WRIST_BODY)
left_elbow_body_id = get_body_id(model, LEFT_ELBOW_BODY)

print("\nRight arm joints:")
right_qpos_ids, right_dof_ids, right_ranges = get_joint_info(model, RIGHT_ARM_JOINTS)

print("\nLeft arm joints:")
left_qpos_ids, left_dof_ids, left_ranges = get_joint_info(model, LEFT_ARM_JOINTS)

# Initialise both arms in rest pose.
data.qpos[right_qpos_ids] = np.clip(
    RIGHT_ARM_REST_POSE,
    right_ranges[:, 0],
    right_ranges[:, 1],
)

data.qpos[left_qpos_ids] = np.clip(
    LEFT_ARM_REST_POSE,
    left_ranges[:, 0],
    left_ranges[:, 1],
)

mujoco.mj_forward(model, data)

ALLOWED_UPPER_BODY_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

locked_qpos_ids, locked_dof_ids = build_robot_lock_indices(
    model,
    allowed_joint_names=ALLOWED_UPPER_BODY_JOINTS,
)

locked_qpos_values = data.qpos[locked_qpos_ids].copy()

print(f"Locked non-arm robot qpos count: {len(locked_qpos_ids)}")
print(f"Locked non-arm robot dof count: {len(locked_dof_ids)}")
print("Only upper-body arm joints are allowed to move in this MuJoCo teleop scene.")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.setblocking(False)

left_pos_raw = np.array([0.15, 0.28, 0.88], dtype=float)
right_pos_raw = np.array([0.15, -0.28, 0.88], dtype=float)

left_pos_smooth = left_pos_raw.copy()
right_pos_smooth = right_pos_raw.copy()

left_pos_commanded = left_pos_raw.copy()
right_pos_commanded = right_pos_raw.copy()

head_pos = np.array([0.0, 0.0, 1.6], dtype=float)

print(f"\nLoaded: {XML_PATH}")
print(f"Listening on UDP port {UDP_PORT}")
print("Independent calibrated arm retargeting enabled.")
print("Left controller drives left wrist only.")
print("Right controller drives right wrist only.")
print("Hold controllers naturally in camera pose before starting sender.")
print("Press Ctrl+C to stop.")

last_print = 0.0
packet_count = 0

# Neutral controller pose used for calibrated independent retargeting.
# This prevents NameError before the first UDP packet arrives.
neutral_left = left_pos_raw.copy()
neutral_right = right_pos_raw.copy()
neutral_captured = False
deadman = False

logger = make_logger()
print(f"Logging MuJoCo session to: {logger.session_dir}")
session_start_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        while True:
            try:
                packet, address = sock.recvfrom(4096)
            except BlockingIOError:
                break

            packet_count += 1
            msg = json.loads(packet.decode("utf-8"))

            if "left" in msg:
                left_pos_raw = np.array(msg["left"], dtype=float)

            if "right" in msg:
                right_pos_raw = np.array(msg["right"], dtype=float)

            if "head" in msg:
                head_pos = np.array(msg["head"], dtype=float)

            if not neutral_captured and "left" in msg and "right" in msg:
                neutral_left = left_pos_raw.copy()
                neutral_right = right_pos_raw.copy()
                neutral_captured = True
                print("Captured neutral controller pose for independent retargeting.")
                print("Left controller neutral :", np.round(neutral_left, 3))
                print("Right controller neutral:", np.round(neutral_right, 3))

            deadman = bool(msg.get("deadman", False))
            left_trigger = float(msg.get("left_trigger", 0.0))
            right_trigger = float(msg.get("right_trigger", 0.0))

        # Independent calibrated retargeting.
        # Left controller controls left arm only.
        # Right controller controls right arm only.
        left_target, right_target = compute_independent_retargeted_targets(
            left_raw=left_pos_raw,
            right_raw=right_pos_raw,
            neutral_left=neutral_left,
            neutral_right=neutral_right,
        )

        left_pos_smooth = (
            (1.0 - TARGET_SMOOTHING) * left_pos_smooth
            + TARGET_SMOOTHING * left_target
        )

        right_pos_smooth = (
            (1.0 - TARGET_SMOOTHING) * right_pos_smooth
            + TARGET_SMOOTHING * right_target
        )

        # Update visible mocap spheres.
        data.mocap_pos[left_target_id] = left_pos_smooth
        data.mocap_pos[right_target_id] = right_pos_smooth
        data.mocap_pos[head_target_id] = head_pos

        # Solve left arm first, then right arm.
        left_elbow_target = solve_arm_ik(
            model=model,
            data=data,
            wrist_body_id=left_wrist_body_id,
            elbow_body_id=left_elbow_body_id,
            qpos_ids=left_qpos_ids,
            dof_ids=left_dof_ids,
            joint_ranges=left_ranges,
            rest_pose=LEFT_ARM_REST_POSE,
            wrist_target=left_pos_smooth,
            side="left",
        )

        right_elbow_target = solve_arm_ik(
            model=model,
            data=data,
            wrist_body_id=right_wrist_body_id,
            elbow_body_id=right_elbow_body_id,
            qpos_ids=right_qpos_ids,
            dof_ids=right_dof_ids,
            joint_ranges=right_ranges,
            rest_pose=RIGHT_ARM_REST_POSE,
            wrist_target=right_pos_smooth,
            side="right",
        )

        now = time.time()
        elapsed = now - session_start_time

        left_wrist_pos = data.xpos[left_wrist_body_id].copy()
        right_wrist_pos = data.xpos[right_wrist_body_id].copy()

        left_err = float(np.linalg.norm(left_pos_smooth - left_wrist_pos))
        right_err = float(np.linalg.norm(right_pos_smooth - right_wrist_pos))

        row = {
            "timestamp": now,
            "elapsed": elapsed,
            "packet_count": packet_count,
            "deadman": int(deadman),
            "left_error": left_err,
            "right_error": right_err,
        }

        row.update(flatten_vec("left", left_pos_smooth))
        row.update(flatten_vec("right", right_pos_smooth))
        row.update(flatten_vec("head", head_pos))
        row.update(flatten_vec("left_raw", left_pos_raw))
        row.update(flatten_vec("right_raw", right_pos_raw))
        row.update(flatten_vec("left_target", left_pos_smooth))
        row.update(flatten_vec("right_target", right_pos_smooth))
        row.update(flatten_vec("left_wrist", left_wrist_pos))
        row.update(flatten_vec("right_wrist", right_wrist_pos))
        row.update(flatten_vec("left_elbow_target", left_elbow_target))
        row.update(flatten_vec("right_elbow_target", right_elbow_target))

        add_joint_values(row, "cmd_left", LEFT_ARM_JOINTS, left_qpos_ids, data)
        add_joint_values(row, "actual_left", LEFT_ARM_JOINTS, left_qpos_ids, data)
        add_joint_values(row, "cmd_right", RIGHT_ARM_JOINTS, right_qpos_ids, data)
        add_joint_values(row, "actual_right", RIGHT_ARM_JOINTS, right_qpos_ids, data)

        # Record manipulation object pose and velocity for training data.
        add_object_state(row, model, data, object_body_ids)

        logger.write_row(row)

        # Stable upper-body teleop mode.
        # Do not call mj_step here: full-body physics causes the G1 legs/base to fight the IK.
        # The robot is treated as a kinematic upper-body rig for VR teleop prototyping.
        mujoco.mj_forward(model, data)

        if now - last_print > 1.0:
            print(
                f"Packets: {packet_count} | "
                f"L_err={left_err:.3f} | "
                f"R_err={right_err:.3f} | "
                f"L_target={np.round(left_pos_smooth, 3)} | "
                f"R_target={np.round(right_pos_smooth, 3)} | "
                f"MODE=INDEPENDENT_RETARGETED | "
                f"deadman={deadman} | "
                f"L_trig={locals().get('left_trigger', 0.0):.2f} | "
                f"R_trig={locals().get('right_trigger', 0.0):.2f} | "
                f"SIM_FOLLOWS_ALWAYS=True"
            )

            last_print = now

        viewer.sync()

logger.close()
