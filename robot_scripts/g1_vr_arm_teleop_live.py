import json
import socket
import time
import numpy as np

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
    input("Press Enter to ENABLE LIVE ARM TELEOP for 30 seconds. Keep e-stop ready...")

    crc = CRC()
    low_cmd = unitree_hg_msg_dds__LowCmd_()

    start_time = time.time()
    last_packet_time = time.time()
    last_print = 0.0

    try:
        while time.time() - start_time < 30.0:
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

                target_q = compute_targets(start_q, left_delta, right_delta)
                cmd_q = rate_limit(cmd_q, target_q)

                last_packet_time = time.time()

            # Packet timeout safety: keep holding the last safe command.
            if time.time() - last_packet_time > 0.25:
                print("UDP timeout. Holding last safe command.")
                last_packet_time = time.time()

            write_command(publisher, crc, low_cmd, cmd_q)

            now = time.time()
            if now - last_print > 0.5:
                print(
                    f"L_elbow={cmd_q[G1JointIndex.LeftElbow]:.3f} "
                    f"R_elbow={cmd_q[G1JointIndex.RightElbow]:.3f} "
                    f"L_sh_pitch={cmd_q[G1JointIndex.LeftShoulderPitch]:.3f} "
                    f"R_sh_pitch={cmd_q[G1JointIndex.RightShoulderPitch]:.3f} "
                    f"L_sh_roll={cmd_q[G1JointIndex.LeftShoulderRoll]:.3f} "
                    f"R_sh_roll={cmd_q[G1JointIndex.RightShoulderRoll]:.3f}"
                )
                last_print = now

            time.sleep(0.02)

    finally:
        release_arm_sdk(publisher, crc, low_cmd)
        print("Done.")


if __name__ == "__main__":
    main()
