import json
import socket
import time

import openvr


WSL_IP = "192.168.123.164"
UDP_PORT = 5005

# Adjust this if movement feels too large or too small.
POSITION_SCALE = 1.0

MUJOCO_HEAD_NEUTRAL = [0.0, 0.0, 1.6]

# Hands close together in front of the robot for camera/tripod-handle testing.
MUJOCO_LEFT_NEUTRAL = [0.08, 0.16, 0.82]
MUJOCO_RIGHT_NEUTRAL = [0.08, -0.16, 0.82]


def matrix_to_position(m):
    # Extract x/y/z position from SteamVR's 3x4 tracking matrix.
    return [float(m[0][3]), float(m[1][3]), float(m[2][3])]


def subtract(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def scale(v, s):
    return [v[0] * s, v[1] * s, v[2] * s]


def steamvr_delta_to_mujoco_delta(v):
    """
    Temporary axis remap.

    SteamVR commonly behaves like:
      x = left/right
      y = up/down
      z = forward/back

    MuJoCo/world space used by the robot-side script:
      x = forward/back
      y = left/right
      z = up/down
    """
    steam_x, steam_y, steam_z = v

    mujoco_x = -steam_z
    mujoco_y = -steam_x
    mujoco_z = steam_y

    return [mujoco_x, mujoco_y, mujoco_z]


def get_current_poses(vr):
    poses = vr.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding,
        0,
        openvr.k_unMaxTrackedDeviceCount,
    )

    result = {
        "head": None,
        "left": None,
        "right": None,
        "left_index": None,
        "right_index": None,
    }

    for i, pose in enumerate(poses):
        if not pose.bPoseIsValid:
            continue

        device_class = vr.getTrackedDeviceClass(i)
        raw_pos = matrix_to_position(pose.mDeviceToAbsoluteTracking)

        if device_class == openvr.TrackedDeviceClass_HMD:
            result["head"] = raw_pos

        elif device_class == openvr.TrackedDeviceClass_Controller:
            role = vr.getControllerRoleForTrackedDeviceIndex(i)

            if role == openvr.TrackedControllerRole_LeftHand:
                result["left"] = raw_pos
                result["left_index"] = i

            elif role == openvr.TrackedControllerRole_RightHand:
                result["right"] = raw_pos
                result["right_index"] = i

    return result


def get_controller_state(vr, device_index):
    if device_index is None:
        return None

    try:
        state_result = vr.getControllerState(device_index)
    except Exception:
        return None

    # Some pyopenvr versions return (success, state), others return state directly.
    if isinstance(state_result, tuple):
        if len(state_result) >= 2 and state_result[0]:
            return state_result[1]
        return None

    return state_result


def button_pressed(vr, device_index, button_name):
    state = get_controller_state(vr, device_index)
    if state is None:
        return False

    button_id = getattr(openvr, button_name, None)
    if button_id is None:
        return False

    mask = 1 << button_id
    return (state.ulButtonPressed & mask) != 0


def get_deadman_state(vr, poses):
    # Deadman switch uses the inside index trigger on either Meta Quest controller.
    # It intentionally does not use the side grip button.
    button_name = "k_EButton_SteamVR_Trigger"

    for index_name in ("left_index", "right_index"):
        device_index = poses.get(index_name)

        if button_pressed(vr, device_index, button_name):
            return True

    return False


def wait_for_valid_tracking(vr):
    print("Waiting for valid HMD/controller tracking...")

    while True:
        poses = get_current_poses(vr)

        if poses["head"] is not None:
            print("HMD found.")

            if poses["left"] is not None:
                print("Left controller found.")
            else:
                print("Left controller not found yet.")

            if poses["right"] is not None:
                print("Right controller found.")
            else:
                print("Right controller not found yet.")

            return poses

        time.sleep(0.1)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    openvr.init(openvr.VRApplication_Other)
    vr = openvr.VRSystem()

    print("SteamVR/OpenVR started.")
    print("Make sure headset and controllers are awake/tracked.")
    print("Auto-calibrating from current HMD/controller poses...")

    try:
        origin_poses = wait_for_valid_tracking(vr)

        # Use the HMD as the shared global tracking origin.
        head_origin = origin_poses["head"]
        left_origin = head_origin
        right_origin = head_origin

        print("Calibration complete.")
        print("Move the headset/controllers and watch the robot-side receiver.")
        print("Hold either Quest inside trigger to enable robot movement.")
        print("Press Ctrl+C to stop.")

        last_print_time = 0.0

        while True:
            poses = get_current_poses(vr)

            head_pos = MUJOCO_HEAD_NEUTRAL
            left_pos = MUJOCO_LEFT_NEUTRAL
            right_pos = MUJOCO_RIGHT_NEUTRAL

            if poses["head"] is not None:
                delta = subtract(poses["head"], head_origin)
                mapped = steamvr_delta_to_mujoco_delta(delta)
                head_pos = add(MUJOCO_HEAD_NEUTRAL, scale(mapped, POSITION_SCALE))

            if poses["left"] is not None:
                delta = subtract(poses["left"], left_origin)
                mapped = steamvr_delta_to_mujoco_delta(delta)
                left_pos = add(MUJOCO_LEFT_NEUTRAL, scale(mapped, POSITION_SCALE))

            if poses["right"] is not None:
                delta = subtract(poses["right"], right_origin)
                mapped = steamvr_delta_to_mujoco_delta(delta)
                right_pos = add(MUJOCO_RIGHT_NEUTRAL, scale(mapped, POSITION_SCALE))

            deadman = get_deadman_state(vr, poses)

            packet = {
                "left": left_pos,
                "right": right_pos,
                "head": head_pos,
                "deadman": deadman,
            }

            data = json.dumps(packet).encode("utf-8")
            sock.sendto(data, (WSL_IP, UDP_PORT))

            now = time.time()
            if now - last_print_time > 1.0:
                print("Sending:")
                print("  head    :", [round(v, 3) for v in head_pos])
                print("  left    :", [round(v, 3) for v in left_pos])
                print("  right   :", [round(v, 3) for v in right_pos])
                print("  deadman :", deadman)
                last_print_time = now

            time.sleep(0.016)

    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()
