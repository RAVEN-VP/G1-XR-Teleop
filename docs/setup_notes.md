# Setup Notes

## Laptop Network

The laptop can use Wi-Fi for internet and Ethernet directly to the G1 robot.

The robot computer was reachable at:

```text
192.168.123.164

The secondary G1 address also responded during testing:

192.168.123.161
Windows Sender

The Windows sender is:

windows_sender/g1_steamvr_sender_calibrated.py

It streams Quest controller positions to the robot over UDP.

The UDP target should be set to the robot computer IP:

WSL_IP = "192.168.123.164"

The sender is run from Windows PowerShell:

python $env:USERPROFILE\g1_steamvr_sender_calibrated.py
Robot Environment

On the robot computer, the Unitree SDK2 Python environment needs the following variables:

cd ~/unitree_sdk2_python

export PYTHONPATH=$PWD:$PYTHONPATH
export CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds
export CMAKE_PREFIX_PATH=/home/unitree/cyclonedds_ws/install/cyclonedds
export LD_LIBRARY_PATH=/home/unitree/cyclonedds_ws/install/cyclonedds/lib:$LD_LIBRARY_PATH
Robot Mode

The G1 must be placed into the correct arm/motion-control mode before rt/arm_sdk commands will move the arms.

During testing, the official Unitree arm example did not move until the correct mode was selected on the remote. Once the correct mode was active, the official example and custom teleop scripts moved the arms.

Safety Notes

Keep the remote/e-stop ready during all live testing.

Run the scripts in this order when testing:

g1_read_arm_state.py
robot_udp_test.py
g1_vr_arm_teleop_dryrun.py
g1_hold_current_arms_waist.py
g1_vr_arm_teleop_live.py

The live script currently uses joint clamps, rate limiting, UDP timeout handling, and waist joint holding.
