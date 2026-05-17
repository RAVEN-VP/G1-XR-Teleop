import json
import socket
import time

import mujoco
import mujoco.viewer


XML_PATH = "./unitree_robots/g1/g1_29dof_vrtest.xml"
UDP_PORT = 5005


def get_mocap_id(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = model.body_mocapid[body_id]
    print(f"{body_name}: body_id={body_id}, mocap_id={mocap_id}")
    return mocap_id


model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

left_id = get_mocap_id(model, "vr_left_target")
right_id = get_mocap_id(model, "vr_right_target")
head_id = get_mocap_id(model, "vr_head_target")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.setblocking(False)

left_pos = [0.3, 0.25, 1.1]
right_pos = [0.3, -0.25, 1.1]
head_pos = [0.0, 0.0, 1.5]

packet_count = 0
last_print = 0.0

print(f"Loaded: {XML_PATH}")
print(f"Listening on UDP port {UDP_PORT}")
print("This debug version uses mj_forward(), not mj_step().")
print("Waiting for packets...")

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
                left_pos = [float(v) for v in msg["left"]]
            if "right" in msg:
                right_pos = [float(v) for v in msg["right"]]
            if "head" in msg:
                head_pos = [float(v) for v in msg["head"]]

            now = time.time()
            if now - last_print > 0.5:
                print(f"Packet {packet_count} from {address}")
                print("  left :", left_pos)
                print("  right:", right_pos)
                print("  head :", head_pos)
                last_print = now

        data.mocap_pos[left_id] = left_pos
        data.mocap_pos[right_id] = right_pos
        data.mocap_pos[head_id] = head_pos

        # Force MuJoCo to recompute body positions from mocap data.
        mujoco.mj_forward(model, data)

        viewer.sync()
