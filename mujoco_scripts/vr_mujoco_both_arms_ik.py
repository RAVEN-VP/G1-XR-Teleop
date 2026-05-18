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


XML_PATH = "./unitree_robots/g1/g1_29dof_vrtest.xml"
UDP_PORT = 5005


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

    metadata = {
        "script": "vr_mujoco_both_arms_ik.py",
        "pipeline": "Quest controllers -> Windows UDP sender -> MuJoCo G1 elbow-guided IK",
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

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.setblocking(False)

left_pos_raw = np.array([0.15, 0.28, 0.88], dtype=float)
right_pos_raw = np.array([0.15, -0.28, 0.88], dtype=float)

left_pos_smooth = left_pos_raw.copy()
right_pos_smooth = right_pos_raw.copy()

head_pos = np.array([0.0, 0.0, 1.6], dtype=float)

print(f"\nLoaded: {XML_PATH}")
print(f"Listening on UDP port {UDP_PORT}")
print("Both-arm elbow-guided IK enabled.")
print("Left controller drives left wrist; right controller drives right wrist.")
print("Press Ctrl+C to stop.")

last_print = 0.0
packet_count = 0
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

            deadman = bool(msg.get("deadman", False))

        # Clamp and smooth targets.
        left_clamped = clamp_target(left_pos_raw, "left")
        right_clamped = clamp_target(right_pos_raw, "right")

        left_pos_smooth = (
            (1.0 - TARGET_SMOOTHING) * left_pos_smooth
            + TARGET_SMOOTHING * left_clamped
        )

        right_pos_smooth = (
            (1.0 - TARGET_SMOOTHING) * right_pos_smooth
            + TARGET_SMOOTHING * right_clamped
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

        logger.write_row(row)

        if now - last_print > 1.0:
            print(
                f"Packets: {packet_count} | "
                f"L_err={left_err:.3f} | "
                f"R_err={right_err:.3f} | "
                f"L_target={np.round(left_pos_smooth, 3)} | "
                f"R_target={np.round(right_pos_smooth, 3)} | "
                f"deadman={deadman}"
            )

            last_print = now

        viewer.sync()

logger.close()
