# Training Data Schema

## Purpose

This document defines the intermediate dataset format used to convert VR teleoperation logs into training data for later imitation learning, policy learning, and Isaac Lab integration.

The current data pipeline is:

```text
VR teleoperation session
→ logs/session_*/metadata.json
→ logs/session_*/trajectory.csv
→ tools/convert_logs_to_npz.py
→ datasets/session_*.npz
→ future Isaac Lab loader / training pipeline

The schema is designed to support both simulation and physical robot data.

Session Structure

Each recorded session should contain:

logs/session_YYYYMMDD_HHMMSS/
  metadata.json
  trajectory.csv

The converted dataset should be saved as:

datasets/session_YYYYMMDD_HHMMSS.npz
Metadata

metadata.json stores session-level information.

Expected fields include:

created_utc
session_name
script
pipeline
robot_or_sim
udp_port
control_joints
joint_limits
control_dt_seconds
rate_limit_rad_per_frame
deadman description
task_name
object_name
operator
notes

Not every field is required in the first prototype, but the schema should grow toward this structure.

Trajectory CSV

trajectory.csv stores one row per timestep.

Each row should represent one control/logging frame.

Core fields:

timestamp
elapsed
deadman
udp_timeout
VR Controller Fields

Controller positions are stored in the shared teleop coordinate frame:

left_x
left_y
left_z

right_x
right_y
right_z

Controller deltas from the neutral calibration pose:

left_delta_x
left_delta_y
left_delta_z

right_delta_x
right_delta_y
right_delta_z

Optional future fields:

left_quat_x
left_quat_y
left_quat_z
left_quat_w

right_quat_x
right_quat_y
right_quat_z
right_quat_w
Robot Joint Fields

Commanded joint target fields use the prefix:

cmd_

Physical or simulated measured joint fields use the prefix:

actual_

For robot-side logging, the current format is:

cmd_<joint_index>_q
actual_<joint_index>_q

Example:

cmd_15_q
actual_15_q

For MuJoCo logging, the current format may use named joints:

cmd_left_left_shoulder_pitch_joint
actual_left_left_shoulder_pitch_joint

This should eventually be normalised into a shared joint ordering.

Recommended Canonical Joint Ordering

For G1 upper-body training, the canonical order should be:

12  WaistYaw
13  WaistRoll
14  WaistPitch

15  LeftShoulderPitch
16  LeftShoulderRoll
17  LeftShoulderYaw
18  LeftElbow
19  LeftWristRoll
20  LeftWristPitch
21  LeftWristYaw

22  RightShoulderPitch
23  RightShoulderRoll
24  RightShoulderYaw
25  RightElbow
26  RightWristRoll
27  RightWristPitch
28  RightWristYaw

All future converters and Isaac Lab loaders should preserve this ordering unless explicitly documented.

Observation Vector

The current observation vector should include:

left controller position
right controller position
left controller delta
right controller delta
actual robot/sim joint positions
deadman state

In column-prefix terms, the current converter uses:

left_
right_
left_delta_
right_delta_
actual_

These become:

observations

in the .npz file.

Action Vector

The current action vector should include commanded joint targets:

cmd_

These become:

actions

in the .npz file.

For the current system, the action target is:

desired joint position target

Later alternatives may include:

joint position delta
joint velocity target
end-effector pose target
hand/wrist target pose

The action interpretation must be stored in metadata.

Object State Fields

For manipulation tasks, object state should be added to the logs.

Recommended fields:

object_<name>_pos_x
object_<name>_pos_y
object_<name>_pos_z

object_<name>_quat_x
object_<name>_quat_y
object_<name>_quat_z
object_<name>_quat_w

object_<name>_linvel_x
object_<name>_linvel_y
object_<name>_linvel_z

object_<name>_angvel_x
object_<name>_angvel_y
object_<name>_angvel_z

These should later be included in the observation vector.

Camera and Detection Fields

For the G1 camera/object detection pipeline, log detections separately or as additional trajectory fields.

Recommended detection fields:

camera_frame_id
camera_timestamp

detected_object_id
detected_object_class
detected_object_confidence

bbox_x
bbox_y
bbox_w
bbox_h

If 3D object pose is estimated:

detected_object_pos_x
detected_object_pos_y
detected_object_pos_z

Raw images should not be stored directly inside CSV. They should be saved separately and referenced by path or frame ID.

Converted NPZ Format

Each .npz dataset should contain:

observations
actions
timestamps
deadman
observation_columns
action_columns
metadata
source_session

Expected shapes:

observations: [T, observation_dim]
actions:      [T, action_dim]
timestamps:   [T]
deadman:      [T]

where T is the number of timesteps.

Isaac Lab Integration Plan

The .npz format is an intermediate format.

The future Isaac Lab loader should map:

observations → Isaac policy observations
actions      → Isaac action targets
metadata     → task/session configuration

Potential Isaac observation terms:

robot joint positions
robot joint velocities
target hand/controller pose
object pose
object velocity
camera/object detection state

Potential Isaac action terms:

upper-body joint position targets
joint position deltas
end-effector target poses

The first Isaac Lab integration should begin with replaying trajectories before training a policy.

Immediate Development Priorities
Keep robot and MuJoCo logs compatible with this schema.
Add object pose logging to MuJoCo.
Add object manipulation tasks in MuJoCo.
Extend the converter to include object state.
Add a dataset inspection script.
Add an Isaac Lab replay/loader script later.
