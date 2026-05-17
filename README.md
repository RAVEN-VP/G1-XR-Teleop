# G1 XR Teleoperation

This repository contains the VR teleoperation prototype developed for upper-body arm control on the Unitree G1 humanoid robot. The system uses Meta Quest controller input, streamed from a Windows machine, to drive safety-limited arm movement on the G1 through Unitree SDK2.

## Overview

The current system supports real-time VR-driven arm control using the following pipeline:

```text
Meta Quest controllers
→ SteamVR on Windows
→ Python UDP sender
→ G1 robot computer
→ Python teleoperation bridge
→ Unitree SDK2 arm_sdk
→ G1 upper-body arm movement
```

Development began in MuJoCo, where VR target spheres and arm IK experiments were used to prototype controller-to-arm behaviour. The system was then transferred to the physical G1, where the final robot-side scripts were tested through Unitree’s `rt/lowstate` and `rt/arm_sdk` interfaces.

## Repository Structure

```text
windows_sender/
  Windows-side SteamVR sender for reading Quest controller positions and streaming them over UDP.

robot_scripts/
  Robot-side scripts for reading G1 state, testing UDP reception, running dry-run teleop, live teleop, and safety checks.

mujoco_scripts/
  MuJoCo prototype scripts for VR target testing and arm IK development.

mujoco_xml/
  Modified G1 MuJoCo XML files with VR mocap targets.

external/
  External dependencies and related frameworks, including GR00T WholeBodyControl as a Git submodule.

docs/
  Project notes, reports, and setup documentation.
```

## Current Status

The prototype can move the real G1 arms from Meta Quest controller input.

Implemented so far:

- Quest controller tracking through SteamVR
- Windows-to-robot UDP streaming
- robot-side UDP packet reception
- Unitree SDK2 lowstate subscription
- Unitree `arm_sdk` command publishing
- live VR-driven G1 arm movement
- waist joint hold during arm control
- joint clamps and rate limits
- MuJoCo arm-control prototypes
- dry-run mode for validating controller-to-joint mappings before live robot movement

## Main Scripts

### Windows Sender

`windows_sender/g1_steamvr_sender_calibrated.py`

Reads Quest controller positions through SteamVR and streams left/right controller data to the robot over UDP.

### Robot-Side Scripts

`robot_scripts/g1_read_arm_state.py`

Reads and prints G1 waist and arm joint positions from `rt/lowstate`.

`robot_scripts/robot_udp_test.py`

Simple UDP listener used to verify that the robot computer is receiving packets from the Windows sender.

`robot_scripts/g1_vr_arm_teleop_dryrun.py`

Receives VR packets and calculates target joint commands without moving the robot.

`robot_scripts/g1_vr_arm_teleop_live.py`

Live VR teleoperation bridge. Converts controller movement into safety-limited Unitree `arm_sdk` commands.

`robot_scripts/g1_hold_current_arms_waist.py`

Safety test that enables arm SDK while holding the current waist and arm joint positions.

## Robot Runtime Environment

On the G1 robot computer, set the following environment before running robot-side scripts:

```bash
cd ~/unitree_sdk2_python

export PYTHONPATH=$PWD:$PYTHONPATH
export CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds
export CMAKE_PREFIX_PATH=/home/unitree/cyclonedds_ws/install/cyclonedds
export LD_LIBRARY_PATH=/home/unitree/cyclonedds_ws/install/cyclonedds/lib:$LD_LIBRARY_PATH
```

## Safety Notes

The robot must be in the correct Unitree motion-control / arm SDK mode before `rt/arm_sdk` commands will move the arms.

The current live teleoperation script is a research prototype. It should only be run with the robot’s arm workspace clear and the remote/e-stop ready.

The waist joints are explicitly included and held during live teleoperation. This was added after testing showed that omitting the waist joints could allow unsafe torso or hip movement.

Current safety measures include:

- waist joint hold
- joint position clamps
- command rate limiting
- UDP timeout handling
- short controlled live-test runtime
- dry-run validation before live movement

## Development Notes

The current control method maps VR controller movement directly to G1 shoulder and elbow joint offsets. This is sufficient for early teleoperation testing, but it is not yet full pose tracking or IK-based hand following.

The next stage should move toward a more structured teleoperation controller, including deadman input, continuous safe operation, data recording, and IK-based wrist or hand target control.

## Planned Work

- Add a controller deadman switch
- Replace fixed-duration tests with continuous safe runtime
- Improve left/right arm symmetry
- Add CSV logging for controller input, commanded joints, and measured lowstate
- Add clamp warnings and safety diagnostics
- Move from direct joint mapping to IK-based hand/wrist target control
- Develop a dedicated camera/tripod-handle teleoperation mode
