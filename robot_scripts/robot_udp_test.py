import socket
import json

PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))

print(f"Robot listening for UDP on port {PORT}...")

while True:
    data, addr = sock.recvfrom(4096)
    text = data.decode("utf-8", errors="replace")

    print("From:", addr)
    print(text)

    try:
        msg = json.loads(text)
        print("left :", msg.get("left"))
        print("right:", msg.get("right"))
        print("head :", msg.get("head"))
    except Exception as e:
        print("JSON error:", e)
