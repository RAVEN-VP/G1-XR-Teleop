import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


class G1JointIndex:
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


ARM_JOINTS = [
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28,
]


last_state = None


def low_state_handler(msg: LowState_):
    global last_state
    last_state = msg


def write_arm_pose(publisher, crc, low_cmd, target_q, kp=45.0, kd=1.2):
    low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0

    for joint in ARM_JOINTS:
        low_cmd.motor_cmd[joint].tau = 0.0
        low_cmd.motor_cmd[joint].q = target_q[joint]
        low_cmd.motor_cmd[joint].dq = 0.0
        low_cmd.motor_cmd[joint].kp = kp
        low_cmd.motor_cmd[joint].kd = kd

    low_cmd.crc = crc.Crc(low_cmd)
    publisher.Write(low_cmd)


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
    for joint in ARM_JOINTS:
        start_q[joint] = last_state.motor_state[joint].q

    start_elbow = start_q[G1JointIndex.RightElbow]
    elbow_delta = 0.08

    target_elbow = float(np.clip(start_elbow + elbow_delta, 0.20, 1.60))

    target_q = dict(start_q)
    target_q[G1JointIndex.RightElbow] = target_elbow

    print("Tiny right elbow test:")
    print(f"  start elbow  = {start_elbow:.4f}")
    print(f"  target elbow = {target_elbow:.4f}")
    print(f"  delta        = {target_elbow - start_elbow:.4f}")

    input("Press Enter to move RIGHT ELBOW slightly and return. Keep e-stop ready...")

    crc = CRC()
    low_cmd = unitree_hg_msg_dds__LowCmd_()
    dt = 0.02

    print("Moving to target...")
    for i in range(100):
        ratio = (i + 1) / 100.0
        cmd_q = dict(start_q)
        cmd_q[G1JointIndex.RightElbow] = (
            (1.0 - ratio) * start_elbow
            + ratio * target_elbow
        )
        write_arm_pose(publisher, crc, low_cmd, cmd_q)
        time.sleep(dt)

    print("Holding target...")
    for _ in range(50):
        write_arm_pose(publisher, crc, low_cmd, target_q)
        time.sleep(dt)

    print("Returning to original...")
    for i in range(100):
        ratio = (i + 1) / 100.0
        cmd_q = dict(start_q)
        cmd_q[G1JointIndex.RightElbow] = (
            (1.0 - ratio) * target_elbow
            + ratio * start_elbow
        )
        write_arm_pose(publisher, crc, low_cmd, cmd_q)
        time.sleep(dt)

    print("Releasing arm SDK...")
    for i in range(50):
        low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0 - (i / 49.0)
        low_cmd.crc = crc.Crc(low_cmd)
        publisher.Write(low_cmd)
        time.sleep(dt)

    print("Done.")


if __name__ == "__main__":
    main()
