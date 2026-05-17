import sys
import time

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


JOINTS = [
    ("Waist yaw", G1JointIndex.WaistYaw),
    ("Waist roll", G1JointIndex.WaistRoll),
    ("Waist pitch", G1JointIndex.WaistPitch),

    ("L shoulder pitch", G1JointIndex.LeftShoulderPitch),
    ("L shoulder roll", G1JointIndex.LeftShoulderRoll),
    ("L shoulder yaw", G1JointIndex.LeftShoulderYaw),
    ("L elbow", G1JointIndex.LeftElbow),
    ("L wrist roll", G1JointIndex.LeftWristRoll),
    ("L wrist pitch", G1JointIndex.LeftWristPitch),
    ("L wrist yaw", G1JointIndex.LeftWristYaw),

    ("R shoulder pitch", G1JointIndex.RightShoulderPitch),
    ("R shoulder roll", G1JointIndex.RightShoulderRoll),
    ("R shoulder yaw", G1JointIndex.RightShoulderYaw),
    ("R elbow", G1JointIndex.RightElbow),
    ("R wrist roll", G1JointIndex.RightWristRoll),
    ("R wrist pitch", G1JointIndex.RightWristPitch),
    ("R wrist yaw", G1JointIndex.RightWristYaw),
]


last_state = None


def low_state_handler(msg: LowState_):
    global last_state
    last_state = msg


def main():
    if len(sys.argv) > 1:
        iface = sys.argv[1]
        print(f"Initialising DDS on interface: {iface}")
        ChannelFactoryInitialize(0, iface)
    else:
        print("Initialising DDS with default interface selection")
        ChannelFactoryInitialize(0)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(low_state_handler, 10)

    print("Subscribed to rt/lowstate.")
    print("Waiting for robot state... Ctrl+C to stop.")

    while True:
        if last_state is None:
            time.sleep(0.2)
            continue

        print("\n--- G1 waist + arm joint state ---")
        for name, idx in JOINTS:
            motor = last_state.motor_state[idx]
            print(f"{idx:02d} {name:18s} q={motor.q: .4f} dq={motor.dq: .4f}")

        time.sleep(1.0)


if __name__ == "__main__":
    main()
