import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

# Allow this script to import ../common when run from the repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.teleop_logger import TeleopLogger, flatten_vec, flatten_joint_values, make_joint_fields

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


class G1JointIndex:
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14

    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

    kNotUsedJoint = 29


CONTROL_JOINTS = [
    12, 13, 14,
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28,
]


JOINT_LIMITS = {
    G1JointIndex.WaistYaw: (-0.25, 0.25),
    G1JointIndex.WaistRoll: (-0.10, 0.10),
    G1JointIndex.WaistPitch: (-0.10, 0.10),

    G1JointIndex.LeftShoulderPitch: (-1.2, 1.2),
    G1JointIndex.LeftShoulderRoll: (-0.4, 1.4),
    G1JointIndex.LeftShoulderYaw: (-1.8, 0.8),
    G1JointIndex.LeftElbow: (0.25, 1.8),
    G1JointIndex.LeftWristRoll: (-1.6, 1.6),
    G1JointIndex.LeftWristPitch: (-0.8, 0.8),
    G1JointIndex.LeftWristYaw: (-0.8, 0.8),

    G1JointIndex.RightShoulderPitch: (-1.2, 1.2),
    G1JointIndex.RightShoulderRoll: (-1.4, 0.4),
    G1JointIndex.RightShoulderYaw: (-0.8, 1.8),
    G1JointIndex.RightElbow: (0.25, 1.8),
    G1JointIndex.RightWristRoll: (-1.6, 1.6),
    G1JointIndex.RightWristPitch: (-0.8, 0.8),
    G1JointIndex.RightWristYaw: (-0.8, 0.8),
}


last_state = None


def low_state_handler(msg: LowState_):
    global last_state
    last_state = msg


def clamp_joint(joint, value):
    lo, hi = JOINT_LIMITS[joint]
    return float(np.clip(value, lo, hi))


def compute_targets(start_q, left_delta, right_delta):
    targets = dict(start_q)

    lx, ly, lz = left_delta
    rx, ry, rz = right_delta

    # Hold waist/torso fixed at the captured startup pose.
    targets[G1JointIndex.WaistYaw] = start_q[G1JointIndex.WaistYaw]
    targets[G1JointIndex.WaistRoll] = start_q[G1JointIndex.WaistRoll]
    targets[G1JointIndex.WaistPitch] = start_q[G1JointIndex.WaistPitch]

    # Left arm mapping.
    targets[G1JointIndex.LeftShoulderPitch] = start_q[G1JointIndex.LeftShoulderPitch] - 1.15 * lz - 1.35 * lx
    targets[G1JointIndex.LeftShoulderRoll] = start_q[G1JointIndex.LeftShoulderRoll] + 2.40 * ly
    targets[G1JointIndex.LeftShoulderYaw] = start_q[G1JointIndex.LeftShoulderYaw] + 0.45 * lx
    targets[G1JointIndex.LeftElbow] = start_q[G1JointIndex.LeftElbow] + 0.55 * max(0.0, lz)

    # Right arm mapping.
    targets[G1JointIndex.RightShoulderPitch] = start_q[G1JointIndex.RightShoulderPitch] - 1.15 * rz - 1.35 * rx
    targets[G1JointIndex.RightShoulderRoll] = start_q[G1JointIndex.RightShoulderRoll] + 1.80 * ry
    targets[G1JointIndex.RightShoulderYaw] = start_q[G1JointIndex.RightShoulderYaw] + 0.45 * rx
    targets[G1JointIndex.RightElbow] = start_q[G1JointIndex.RightElbow] + 0.55 * max(0.0, rz)

    for joint in CONTROL_JOINTS:
        targets[joint] = clamp_joint(joint, targets[joint])

    return targets


def rate_limit(prev_q, target_q, max_step=0.018):
    out = dict(prev_q)

    for joint in CONTROL_JOINTS:
        delta = target_q[joint] - prev_q[joint]
        delta = float(np.clip(delta, -max_step, max_step))
        out[joint] = prev_q[joint] + delta

    return out


def write_command(publisher, crc, low_cmd, cmd_q, kp=35.0, kd=1.0):
    # Enable arm SDK control.
    low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0

    for joint in CONTROL_JOINTS:
        low_cmd.motor_cmd[joint].tau = 0.0
        low_cmd.motor_cmd[joint].q = cmd_q[joint]
        low_cmd.motor_cmd[joint].dq = 0.0
        low_cmd.motor_cmd[joint].kp = kp
        low_cmd.motor_cmd[joint].kd = kd

    low_cmd.crc = crc.Crc(low_cmd)
    publisher.Write(low_cmd)


def release_arm_sdk(publisher, crc, low_cmd):
    print("Releasing arm SDK...")

    for i in range(50):
        low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0 - (i / 49.0)
        low_cmd.crc = crc.Crc(low_cmd)
        publisher.Write(low_cmd)
        time.sleep(0.02)


def make_logger():
    fieldnames = [
        "timestamp",
        "elapsed",
        "deadman",
        "udp_timeout",
        "left_x", "left_y", "left_z",
        "right_x", "right_y", "right_z",
        "left_delta_x", "left_delta_y", "left_delta_z",
        "right_delta_x", "right_delta_y", "right_delta_z",
    ]

    fieldnames += make_joint_fields("cmd", CONTROL_JOINTS)
    fieldnames += make_joint_fields("actual", CONTROL_JOINTS)

    metadata = {
        "script": "g1_vr_arm_teleop_live.py",
        "pipeline": "Quest controllers -> Windows UDP sender -> G1 robot Python bridge -> Unitree arm_sdk",
        "udp_port": 5005,
        "control_joints": CONTROL_JOINTS,
        "joint_limits": {str(k): list(v) for k, v in JOINT_LIMITS.items()},
        "rate_limit_rad_per_frame": 0.018,
        "control_dt_seconds": 0.02,
        "deadman": "Quest inside trigger; robot only updates targets while deadman is true",
    }

    return TeleopLogger(
        root_dir=str(REPO_ROOT / "logs"),
        metadata=metadata,
        fieldnames=fieldnames,
    )


def read_actual_q():
    actual_q = {}

    if last_state is None:
        return actual_q

    for joint in CONTROL_JOINTS:
        actual_q[joint] = last_state.motor_state[joint].q

    return actual_q


def main():
    print("Initialising DDS on robot default interface...")
    ChannelFactoryInitialize(0)

    publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
    publisher.Init()

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(low_state_handler, 10)

    print("Waiting for lowstate...")
    while last_state is None:
        time.sleep(0.1)

    start_q = {}
    for joint in CONTROL_JOINTS:
        start_q[joint] = last_state.motor_state[joint].q

    cmd_q = dict(start_q)

    print("Captured current waist + arm pose.")
    print("Starting UDP listener on port 5005.")
    print("Start Windows VR sender now. Hold controllers in neutral pose.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 5005))
    sock.setblocking(False)

    neutral_left = None
    neutral_right = None

    while neutral_left is None:
        try:
            packet, _ = sock.recvfrom(4096)
            msg = json.loads(packet.decode("utf-8"))

            if "left" in msg and "right" in msg:
                neutral_left = np.array(msg["left"], dtype=float)
                neutral_right = np.array(msg["right"], dtype=float)
        except BlockingIOError:
            time.sleep(0.01)

    print("Captured VR neutral positions.")
    input("Press Enter to ENABLE LIVE ARM TELEOP for 30 seconds. Hold Quest inside trigger deadman to move. Keep e-stop ready...")

    crc = CRC()
    low_cmd = unitree_hg_msg_dds__LowCmd_()

    logger = make_logger()
    print(f"Logging session to: {logger.session_dir}")

    start_time = time.time()
    last_packet_time = time.time()
    last_print = 0.0

    left = np.array(neutral_left, dtype=float)
    right = np.array(neutral_right, dtype=float)
    left_delta = np.zeros(3)
    right_delta = np.zeros(3)
    deadman = False

    try:
        while time.time() - start_time < 30.0:
            udp_timeout = False

            while True:
                try:
                    packet, _ = sock.recvfrom(4096)
                except BlockingIOError:
                    break

                msg = json.loads(packet.decode("utf-8"))

                if "left" not in msg or "right" not in msg:
                    continue

                left = np.array(msg["left"], dtype=float)
                right = np.array(msg["right"], dtype=float)

                left_delta = left - neutral_left
                right_delta = right - neutral_right

                deadman = bool(msg.get("deadman", False))

                if deadman:
                    target_q = compute_targets(start_q, left_delta, right_delta)
                    cmd_q = rate_limit(cmd_q, target_q)
                else:
                    # Deadman released: hold the last safe commanded pose.
                    target_q = dict(cmd_q)

                last_packet_time = time.time()

            # Packet timeout safety: keep holding the last safe command.
            if time.time() - last_packet_time > 0.25:
                udp_timeout = True
                print("UDP timeout. Holding last safe command.")
                last_packet_time = time.time()

            write_command(publisher, crc, low_cmd, cmd_q)

            now = time.time()
            elapsed = now - start_time

            row = {
                "timestamp": now,
                "elapsed": elapsed,
                "deadman": int(deadman),
                "udp_timeout": int(udp_timeout),
            }

            row.update(flatten_vec("left", left))
            row.update(flatten_vec("right", right))
            row.update(flatten_vec("left_delta", left_delta))
            row.update(flatten_vec("right_delta", right_delta))
            row.update(flatten_joint_values("cmd", CONTROL_JOINTS, cmd_q))
            row.update(flatten_joint_values("actual", CONTROL_JOINTS, read_actual_q()))

            logger.write_row(row)

            if now - last_print > 0.5:
                print(
                    f"L_elbow={cmd_q[G1JointIndex.LeftElbow]:.3f} "
                    f"R_elbow={cmd_q[G1JointIndex.RightElbow]:.3f} "
                    f"L_sh_pitch={cmd_q[G1JointIndex.LeftShoulderPitch]:.3f} "
                    f"R_sh_pitch={cmd_q[G1JointIndex.RightShoulderPitch]:.3f} "
                    f"L_sh_roll={cmd_q[G1JointIndex.LeftShoulderRoll]:.3f} "
                    f"R_sh_roll={cmd_q[G1JointIndex.RightShoulderRoll]:.3f} "
                    f"deadman={deadman}"
                )
                last_print = now

            time.sleep(0.02)

    finally:
        logger.close()
        release_arm_sdk(publisher, crc, low_cmd)
        print("Done.")


if __name__ == "__main__":
    main()
