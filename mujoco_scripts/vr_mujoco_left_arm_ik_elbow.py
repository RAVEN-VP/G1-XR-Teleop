import json
import socket
import time

import mujoco
import mujoco.viewer
import numpy as np


XML_PATH = "./unitree_robots/g1/g1_29dof_vrtest.xml"
UDP_PORT = 5005

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

LEFT_WRIST_BODY = "left_wrist_yaw_link"
LEFT_ELBOW_BODY = "left_elbow_link"

# Mirrored version of the stable right-arm rest pose.
LEFT_ARM_REST_POSE = np.array([
    0.20,    # shoulder pitch
    0.45,    # shoulder roll
    -0.10,   # shoulder yaw
    0.55,    # elbow bend
    0.00,    # wrist roll
    0.00,    # wrist pitch
    0.00,    # wrist yaw
], dtype=float)

REST_BIAS_STRENGTH = 0.020

JOINT_WEIGHTS = np.array([
    1.0,   # shoulder pitch
    1.0,   # shoulder roll
    1.2,   # shoulder yaw
    0.45,  # elbow
    2.2,   # wrist roll
    2.2,   # wrist pitch
    2.2,   # wrist yaw
], dtype=float)

IK_MAX_ITERATIONS = 18
IK_DAMPING = 0.10
IK_STEP_SCALE = 0.32
TARGET_SMOOTHING = 0.18

# Mirrored left-arm workspace.
LEFT_TARGET_MIN = np.array([-0.05, 0.05, 0.55], dtype=float)
LEFT_TARGET_MAX = np.array([0.75, 0.75, 1.45], dtype=float)

ELBOW_WEIGHT = 0.45
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


def clamp_target(pos):
    return np.clip(pos, LEFT_TARGET_MIN, LEFT_TARGET_MAX)


def compute_elbow_guide(wrist_target):
    guide = LEFT_ELBOW_BASE_GUIDE.copy()

    # Let elbow move slightly with the hand target.
    guide[0] += 0.20 * (wrist_target[0] - 0.15)
    guide[1] += 0.15 * (wrist_target[1] - 0.28)
    guide[2] += 0.25 * (wrist_target[2] - 0.88)

    # Keep elbow below the wrist.
    guide[2] = min(guide[2], wrist_target[2] - 0.08)

    return guide


def solve_left_arm_ik(
    model,
    data,
    wrist_body_id,
    elbow_body_id,
    qpos_ids,
    dof_ids,
    joint_ranges,
    wrist_target,
):
    wrist_target = clamp_target(wrist_target)
    elbow_target = compute_elbow_guide(wrist_target)

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

        weighted_J = J / JOINT_WEIGHTS[None, :]

        A = weighted_J @ weighted_J.T + IK_DAMPING * IK_DAMPING * np.eye(J.shape[0])
        dq_weighted = weighted_J.T @ np.linalg.solve(A, error)
        dq = dq_weighted / JOINT_WEIGHTS

        current_q = data.qpos[qpos_ids].copy()
        rest_error = LEFT_ARM_REST_POSE - current_q
        dq += REST_BIAS_STRENGTH * rest_error

        data.qpos[qpos_ids] += IK_STEP_SCALE * dq

        for i, qid in enumerate(qpos_ids):
            lo, hi = joint_ranges[i]
            data.qpos[qid] = np.clip(data.qpos[qid], lo, hi)

    mujoco.mj_forward(model, data)
    return elbow_target


model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

left_target_id = get_mocap_id(model, "vr_left_target")
right_target_id = get_mocap_id(model, "vr_right_target")
head_target_id = get_mocap_id(model, "vr_head_target")

left_wrist_body_id = get_body_id(model, LEFT_WRIST_BODY)
left_elbow_body_id = get_body_id(model, LEFT_ELBOW_BODY)

qpos_ids, dof_ids, joint_ranges = get_joint_info(model, LEFT_ARM_JOINTS)

data.qpos[qpos_ids] = np.clip(
    LEFT_ARM_REST_POSE,
    joint_ranges[:, 0],
    joint_ranges[:, 1],
)
mujoco.mj_forward(model, data)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.setblocking(False)

left_pos_raw = np.array([0.15, 0.28, 0.88], dtype=float)
left_pos_smooth = left_pos_raw.copy()

right_pos = np.array([0.15, -0.28, 0.88], dtype=float)
head_pos = np.array([0.0, 0.0, 1.6], dtype=float)

print(f"Loaded: {XML_PATH}")
print(f"Listening on UDP port {UDP_PORT}")
print("Left-arm elbow-guided IK enabled.")
print("Left wrist follows the green target; elbow is guided to bend naturally.")
print("Press Ctrl+C to stop.")

last_print = 0.0
packet_count = 0
last_elbow_target = compute_elbow_guide(left_pos_smooth)

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
                right_pos = np.array(msg["right"], dtype=float)

            if "head" in msg:
                head_pos = np.array(msg["head"], dtype=float)

        left_pos_clamped = clamp_target(left_pos_raw)
        left_pos_smooth = (
            (1.0 - TARGET_SMOOTHING) * left_pos_smooth
            + TARGET_SMOOTHING * left_pos_clamped
        )

        data.mocap_pos[left_target_id] = left_pos_smooth
        data.mocap_pos[right_target_id] = right_pos
        data.mocap_pos[head_target_id] = head_pos

        last_elbow_target = solve_left_arm_ik(
            model=model,
            data=data,
            wrist_body_id=left_wrist_body_id,
            elbow_body_id=left_elbow_body_id,
            qpos_ids=qpos_ids,
            dof_ids=dof_ids,
            joint_ranges=joint_ranges,
            wrist_target=left_pos_smooth,
        )

        now = time.time()
        if now - last_print > 1.0:
            wrist_pos = data.xpos[left_wrist_body_id].copy()
            elbow_pos = data.xpos[left_elbow_body_id].copy()

            wrist_err = np.linalg.norm(left_pos_smooth - wrist_pos)
            elbow_err = np.linalg.norm(last_elbow_target - elbow_pos)

            print(
                f"Packets: {packet_count} | "
                f"wrist_err={wrist_err:.3f} | "
                f"elbow_err={elbow_err:.3f} | "
                f"target={np.round(left_pos_smooth, 3)} | "
                f"elbow={np.round(elbow_pos, 3)} | "
                f"q={np.round(data.qpos[qpos_ids], 2)}"
            )
            last_print = now

        viewer.sync()
