import time

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


def main():
    print("Initialising DDS...")
    ChannelFactoryInitialize(0)

    publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
    publisher.Init()

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(low_state_handler, 10)

    print("Waiting for lowstate...")
    while last_state is None:
        time.sleep(0.1)

    current_q = {}
    for joint in ARM_JOINTS:
        current_q[joint] = last_state.motor_state[joint].q

    print("Captured current arm pose:")
    for joint in ARM_JOINTS:
        print(f"  joint {joint:02d}: q={current_q[joint]: .4f}")

    input("Press Enter to HOLD CURRENT ARM POSE for 3 seconds. Keep e-stop ready...")

    crc = CRC()
    low_cmd = unitree_hg_msg_dds__LowCmd_()

    dt = 0.02
    steps = int(3.0 / dt)

    print("Holding current arm pose...")

    for _ in range(steps):
        low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0

        for joint in ARM_JOINTS:
            low_cmd.motor_cmd[joint].tau = 0.0
            low_cmd.motor_cmd[joint].q = current_q[joint]
            low_cmd.motor_cmd[joint].dq = 0.0
            low_cmd.motor_cmd[joint].kp = 40.0
            low_cmd.motor_cmd[joint].kd = 1.0

        low_cmd.crc = crc.Crc(low_cmd)
        publisher.Write(low_cmd)
        time.sleep(dt)

    print("Releasing arm SDK...")

    for i in range(50):
        low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0 - (i / 49.0)
        low_cmd.crc = crc.Crc(low_cmd)
        publisher.Write(low_cmd)
        time.sleep(dt)

    print("Done. Arms should not have moved.")


if __name__ == "__main__":
    main()
