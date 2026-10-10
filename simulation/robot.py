"""Mobile telescopic painting robot.

The seven-DOF chain keeps the original joint names/order for compatibility with
existing planner/controller code, but their physical roles are now:
  joint_base_x        X translation of the base
  joint_base_y        Y translation of the base
  joint_lift           base yaw (rotation in place; legacy name)
  joint_yaw            shoulder tilt in a vertical plane (legacy name)
  joint_shoulder       telescopic arm extension (legacy name)
  joint_elbow          wrist roll (legacy name)
  joint_roller_orient  wrist pitch (legacy name)

The legacy names are intentionally retained to keep this first migration small.
"""
import os
import tempfile

import pybullet as p

URDF_TEMPLATE = r"""<?xml version="1.0"?>
<robot name="paint_robot">
  <material name="blue"><color rgba="0.08 0.38 1.0 1"/></material>
  <material name="brown"><color rgba="0.22 0.12 0.06 1"/></material>
  <material name="tube"><color rgba="0.38 0.22 0.14 1"/></material>
  <material name="piston"><color rgba="0.98 0.78 0.45 1"/></material>
  <material name="yellow"><color rgba="0.95 0.82 0.18 1"/></material>
  <material name="red"><color rgba="1 0.10 0.10 1"/></material>
  <material name="dark_grey"><color rgba="0.22 0.22 0.22 1"/></material>

  <link name="base_link">
    <inertial><mass value="50"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>

  <link name="slider_x_link">
    <visual><geometry><box size="0.04 0.04 0.04"/></geometry><material name="dark_grey"/></visual>
    <inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>
  <joint name="joint_base_x" type="prismatic">
    <parent link="base_link"/><child link="slider_x_link"/>
    <axis xyz="1 0 0"/><origin xyz="0 0 0" rpy="0 0 0"/>
    <limit lower="{x_min}" upper="{x_max}" effort="1000" velocity="2"/>
  </joint>

  <link name="slider_y_link">
    <visual><geometry><box size="0.04 0.04 0.04"/></geometry><material name="dark_grey"/></visual>
    <inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>
  <joint name="joint_base_y" type="prismatic">
    <parent link="slider_x_link"/><child link="slider_y_link"/>
    <axis xyz="0 1 0"/><origin xyz="0 0 0" rpy="0 0 0"/>
    <limit lower="{y_min}" upper="{y_max}" effort="1000" velocity="2"/>
  </joint>

  <!-- Compact chassis; base yaw rotates the chassis, wheels, and arm together. -->
  <link name="base_yaw_link">
    <visual><origin xyz="0 0 0.12"/><geometry><box size="0.55 0.64 0.24"/></geometry><material name="blue"/></visual>
    <collision><origin xyz="0 0 0.12"/><geometry><box size="0.55 0.64 0.24"/></geometry></collision>
    <visual><origin xyz="-0.20 -0.27 0.09" rpy="1.5707963 0 0"/><geometry><cylinder radius="0.09" length="0.06"/></geometry><material name="brown"/></visual>
    <visual><origin xyz="0.20 -0.27 0.09" rpy="1.5707963 0 0"/><geometry><cylinder radius="0.09" length="0.06"/></geometry><material name="brown"/></visual>
    <visual><origin xyz="-0.20 0.27 0.09" rpy="1.5707963 0 0"/><geometry><cylinder radius="0.09" length="0.06"/></geometry><material name="brown"/></visual>
    <visual><origin xyz="0.20 0.27 0.09" rpy="1.5707963 0 0"/><geometry><cylinder radius="0.09" length="0.06"/></geometry><material name="brown"/></visual>
    <visual><origin xyz="0 0 0.56"/><geometry><cylinder radius="0.05" length="0.88"/></geometry><material name="tube"/></visual>
    <inertial><mass value="15"/><inertia ixx="0.35" ixy="0" ixz="0" iyy="0.35" iyz="0" izz="0.35"/></inertial>
  </link>
  <joint name="joint_lift" type="revolute">
    <parent link="slider_y_link"/><child link="base_yaw_link"/>
    <axis xyz="0 0 1"/><origin xyz="0 0 0" rpy="0 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="1000" velocity="2"/>
  </joint>

  <!-- Shoulder tilt; long outer tube provides reach while the beige piston telescopes. -->
  <link name="shoulder_link">
    <visual><origin xyz="0 0 0" rpy="1.5707963 0 0"/><geometry><cylinder radius="0.075" length="0.16"/></geometry><material name="yellow"/></visual>
    <visual><origin xyz="0.325 0 0" rpy="0 1.5707963 0"/><geometry><cylinder radius="0.075" length="0.65"/></geometry><material name="tube"/></visual>
    <inertial><mass value="2"/><inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.04"/></inertial>
  </link>
  <joint name="joint_yaw" type="revolute">
    <parent link="base_yaw_link"/><child link="shoulder_link"/>
    <axis xyz="0 1 0"/><origin xyz="0.02 0 1.0" rpy="0 0 0"/>
    <limit lower="-1.74533" upper="1.74533" effort="500" velocity="1.75"/>
  </joint>

  <link name="extension_link">
    <visual><origin xyz="0.125 0 0" rpy="0 1.5707963 0"/><geometry><cylinder radius="0.045" length="0.85"/></geometry><material name="piston"/></visual>
    <inertial><mass value="1.5"/><inertia ixx="0.025" ixy="0" ixz="0" iyy="0.025" iyz="0" izz="0.025"/></inertial>
  </link>
  <joint name="joint_shoulder" type="prismatic">
    <parent link="shoulder_link"/><child link="extension_link"/>
    <axis xyz="1 0 0"/><origin xyz="0.10 0 0" rpy="0 0 0"/>
    <limit lower="{extension_min}" upper="{extension_max}" effort="500" velocity="0.8"/>
  </joint>

  <link name="wrist_roll_link">
    <visual><origin xyz="0 0 0" rpy="0 1.5707963 0"/><geometry><cylinder radius="0.065" length="0.14"/></geometry><material name="yellow"/></visual>
    <inertial><mass value="0.4"/><inertia ixx="0.008" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.008"/></inertial>
  </link>
  <joint name="joint_elbow" type="revolute">
    <parent link="extension_link"/><child link="wrist_roll_link"/>
    <axis xyz="1 0 0"/><origin xyz="0.55 0 0" rpy="0 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="200" velocity="2.5"/>
  </joint>

  <!-- Fixed handle reaches forward from the wrist; it does not rotate when the roller is oriented. -->
  <link name="roller_handle_link">
    <visual><origin xyz="-0.07 0 0" rpy="0 1.5707963 0"/><geometry><cylinder radius="0.025" length="0.14"/></geometry><material name="brown"/></visual>
    <inertial><mass value="0.08"/><inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.0005"/></inertial>
  </link>

  <!-- The roller link origin is the roller's true center for IK and contact detection. -->
  <link name="roller_link">
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><cylinder radius="0.06" length="0.18"/></geometry><material name="red"/></visual>
    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><cylinder radius="0.06" length="0.18"/></geometry></collision>
    <inertial><mass value="0.35"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>

  <!-- The wrist supports the handle 0.21 m forward, clear of the wrist cylinder. -->
  <joint name="joint_roller_handle_fixed" type="fixed">
    <parent link="wrist_roll_link"/><child link="roller_handle_link"/>
    <origin xyz="0.21 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="joint_roller_orient" type="revolute">
    <parent link="roller_handle_link"/><child link="roller_link"/>
    <axis xyz="0 1 0"/><origin xyz="0 0 0" rpy="0 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="200" velocity="2.5"/>
  </joint>
</robot>
"""


class Robot:
    ROOM_HALF = 5.0
    ROLLER_RADIUS = 0.06
    ROLLER_LENGTH = 0.18
    EXTENSION_MIN = 0.095
    EXTENSION_MAX = 0.85

    def __init__(self, start_xy=(0.0, 0.0), max_lift=9.0, x_range=None, y_range=None):
        # max_lift is retained for API compatibility; vertical reach now comes from shoulder tilt.
        self.max_lift = max_lift
        self.x_range = x_range if x_range is not None else (-self.ROOM_HALF, self.ROOM_HALF)
        self.y_range = y_range if y_range is not None else (-self.ROOM_HALF + 0.2, self.ROOM_HALF - 0.2)
        urdf_path = self._write_urdf()
        self.id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
        os.remove(urdf_path)

        name_to_index = {}
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            name_to_index[info[1].decode("utf-8")] = i

        # Keep this order stable because existing code consumes joint arrays by index.
        self.joint_names = ["joint_base_x", "joint_base_y", "joint_lift", "joint_yaw",
                            "joint_shoulder", "joint_elbow", "joint_roller_orient"]
        self.joint_indices = [name_to_index[n] for n in self.joint_names]
        self.roller_link_index = name_to_index["joint_roller_orient"]

        # Use explicit visual colors so the robot remains easy to read under PyBullet's Metal renderer.
        # base_yaw_link has visuals in this order: chassis, four wheels, vertical mast.
        base_link_index = name_to_index["joint_lift"]
        base_visual_colors = (
            (0.08, 0.38, 1.00, 1.0),  # bright blue chassis
            (0.32, 0.16, 0.07, 1.0),  # wheel 1
            (0.32, 0.16, 0.07, 1.0),  # wheel 2
            (0.32, 0.16, 0.07, 1.0),  # wheel 3
            (0.32, 0.16, 0.07, 1.0),  # wheel 4
            (0.42, 0.25, 0.17, 1.0),  # mast
        )
        for visual_index, color in enumerate(base_visual_colors):
            p.changeVisualShape(self.id, base_link_index, shapeIndex=visual_index, rgbaColor=color)
        p.changeVisualShape(self.id, name_to_index["joint_yaw"], rgbaColor=[0.42, 0.25, 0.17, 1.0])
        p.changeVisualShape(self.id, name_to_index["joint_shoulder"], rgbaColor=[0.98, 0.78, 0.45, 1.0])
        p.changeVisualShape(self.id, name_to_index["joint_elbow"], rgbaColor=[1.0, 0.83, 0.12, 1.0])
        p.changeVisualShape(self.id, self.roller_link_index, rgbaColor=[1.0, 0.10, 0.10, 1.0])

        self.lower_limits, self.upper_limits = [], []
        for idx in self.joint_indices:
            info = p.getJointInfo(self.id, idx)
            self.lower_limits.append(info[8])
            self.upper_limits.append(info[9])
        self.joint_ranges = [u - l for l, u in zip(self.lower_limits, self.upper_limits)]
        self.rest_poses = [0.0] * len(self.joint_indices)
        home_x = (self.x_range[0] + self.x_range[1]) / 2.0
        home_y = (self.y_range[0] + self.y_range[1]) / 2.0
        self.home_joints = [home_x, home_y, 0.0, 0.0, self.EXTENSION_MIN, 0.0, 0.0]
        start_joints = [start_xy[0], start_xy[1], 0.0, 0.0, self.EXTENSION_MIN, 0.0, 0.0]
        self.set_joint_targets(start_joints, instant=True)

    def _write_urdf(self):
        content = URDF_TEMPLATE.format(
            x_min=self.x_range[0], x_max=self.x_range[1],
            y_min=self.y_range[0], y_max=self.y_range[1],
            extension_min=self.EXTENSION_MIN, extension_max=self.EXTENSION_MAX,
        )
        fd, path = tempfile.mkstemp(suffix=".urdf")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    def set_joint_targets(self, joint_positions, instant=False):
        for idx, pos in zip(self.joint_indices, joint_positions):
            if instant:
                p.resetJointState(self.id, idx, pos)
            p.setJointMotorControl2(self.id, idx, p.POSITION_CONTROL,
                                    targetPosition=pos, force=2000, maxVelocity=2.0)

    def get_joint_states(self):
        return [s[0] for s in p.getJointStates(self.id, self.joint_indices)]

    def get_roller_position(self):
        # URDF roller_link origin is the physical center. A separate fixed handle link
        # supports it from the wrist, while this roller alone rotates for orientation control.
        state = p.getLinkState(self.id, self.roller_link_index, computeForwardKinematics=True)
        return list(state[4])

    def get_roller_axis_segment(self):
        state = p.getLinkState(self.id, self.roller_link_index, computeForwardKinematics=True)
        center, orn = state[4], state[5]
        rot = p.getMatrixFromQuaternion(orn)
        # Roller cylinder is built along its local Z axis.
        axis_dir = (rot[2], rot[5], rot[8])
        half_len = self.ROLLER_LENGTH / 2.0
        a = [center[i] + axis_dir[i] * half_len for i in range(3)]
        b = [center[i] - axis_dir[i] * half_len for i in range(3)]
        return a, b

    def set_roller_color(self, rgba):
        p.changeVisualShape(self.id, self.roller_link_index, rgbaColor=rgba)
