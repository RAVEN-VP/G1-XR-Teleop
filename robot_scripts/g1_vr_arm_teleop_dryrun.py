import json
import socket
import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


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


ARM_JOINTS = [
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

    targets[G1JointIndex.WaistYaw] = start_q[G1JointIndex.WaistYaw]
    targets[G1JointIndex.WaistRoll] = start_q[G1JointIndex.WaistRoll]
    targets[G1JointIndex.WaistPitch] = start_q[G1JointIndex.WaistPitch]

    targets[G1JointIndex.LeftShoulderPitch] = start_q[G1JointIndex.LeftShoulderPitch] - 1.15 * lz - 1.35 * lx
    targets[G1JointIndex.LeftShoulderRoll] = start_q[G1JointIndex.LeftShoulderRoll] + 2.40 * ly
    targets[G1JointIndex.LeftShoulderYaw] = start_q[G1JointIndex.LeftShoulderYaw] + 0.45 * lx
    targets[G1JointIndex.LeftElbow] = start_q[G1JointIndex.LeftElbow] + 0.55 * max(0.0, lz)

    targets[G1JointIndex.RightShoulderPitch] = start_q[G1JointIndex.RightShoulderPitch] - 1.15 * rz - 1.35 * rx
    targets[G1JointIndex.RightShoulderRoll] = start_q[G1JointIndex.RightShoulderRoll] + 1.80 * ry
    targets[G1JointIndex.RightShoulderYaw] = start_q[G1JointIndex.RightShoulderYaw] + 0.45 * rx
    targets[G1JointIndex.RightElbow] = start_q[G1JointIndex.RightElbow] + 0.55 * max(0.0, rz)

    for joint in ARM_JOINTS:
        targets[joint] = clamp_joint(joint, targets[joint])

    return targets


def main():
    print("Initialising DDS on robot default interface...")
    ChannelFactoryInitialize(0)

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(low_state_handler, 10)

    print("Waiting for lowstate...")
    while last_state is None:
        time.sleep(0.1)

    start_q = {}
    for joint in ARM_JOINTS:
        start_q[joint] = last_state.motor_state[joint].q

    print("Captured current waist + arm pose.")
    print("Starting UDP listener on port 5005...")
    print("This is DRY RUN ONLY. No commands are sent to the robot.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 5005))
    sock.setblocking(False)

    neutral_left = None
    neutral_right = None
    last_print = 0.0

    while True:
        try:
            packet, _ = sock.recvfrom(4096)
            msg = json.loads(packet.decode("utf-8"))

            if "left" not in msg or "right" not in msg:
                continue

            left = np.array(msg["left"], dtype=float)
            right = np.array(msg["right"], dtype=float)

            if neutral_left is None:
                neutral_left = left.copy()
                neutral_right = right.copy()
                print("Captured VR neutral controller positions.")

            left_delta = left - neutral_left
            right_delta = right - neutral_right

            targets = compute_targets(start_q, left_delta, right_delta)

            now = time.time()
            if now - last_print > 0.5:
                print("\n--- DRY RUN TARGETS ---")
                print(f"left_delta : {np.round(left_delta, 3)}")
                print(f"right_delta: {np.round(right_delta, 3)}")
                print(f"L_sh_pitch={targets[G1JointIndex.LeftShoulderPitch]: .3f}")
                print(f"L_sh_roll ={targets[G1JointIndex.LeftShoulderRoll]: .3f}")
                print(f"L_elbow   ={targets[G1JointIndex.LeftElbow]: .3f}")
                print(f"R_sh_pitch={targets[G1JointIndex.RightShoulderPitch]: .3f}")
                print(f"R_sh_roll ={targets[G1JointIndex.RightShoulderRoll]: .3f}")
                print(f"R_elbow   ={targets[G1JointIndex.RightElbow]: .3f}")
                last_print = now

        except BlockingIOError:
            pass

        time.sleep(0.01)


if __name__ == "__main__":
    main()
