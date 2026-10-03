"""Paint-roller robot: a gantry-style mobile base (x/y/lift/yaw) + 2-link arm + roller end effector.

Kinematic chain (all 7 joints are movable, used directly as the IK chain):
base_link --(joint_base_x, prismatic)--> slider_x_link
--(joint_base_y, prismatic)--> slider_y_link
--(joint_lift, prismatic)--> mast_link
--(joint_yaw, revolute)--> yaw_link (faces the arm at any of the 4 walls)
--(joint_shoulder, revolute)--> upper_arm_link (1m)
--(joint_elbow, revolute)--> forearm_link (1m)
--(joint_roller_orient, revolute)--> roller_link (paint roller, see ROLLER_RADIUS/ROLLER_LENGTH)
"""
import os
import tempfile

import pybullet as p

URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="paint_robot">
<material name="grey"><color rgba="0.5 0.5 0.5 1"/></material>
<material name="dark_grey"><color rgba="0.3 0.3 0.3 1"/></material>
<material name="roller_default"><color rgba="0.9 0.9 0.9 1"/></material>

<link name="base_link">
<!-- Fixed world anchor only; the visible chassis lives on slider_y_link, which actually moves. -->
<inertial><mass value="50"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
</link>

<link name="slider_x_link">
<visual><geometry><box size="0.05 0.05 0.05"/></geometry><material name="dark_grey"/></visual>
<inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
</link>
<joint name="joint_base_x" type="prismatic">
<parent link="base_link"/><child link="slider_x_link"/>
<axis xyz="1 0 0"/><origin xyz="0 0 0" rpy="0 0 0"/>
<limit lower="{x_min}" upper="{x_max}" effort="1000" velocity="2"/>
</joint>

<link name="slider_y_link">
<!-- Mobile base chassis: this is the part that actually slides in x/y across the floor. -->
<visual><origin xyz="0 0 0.1" rpy="0 0 0"/><geometry><box size="0.6 0.6 0.2"/></geometry><material name="grey"/></visual>
<collision><origin xyz="0 0 0.1" rpy="0 0 0"/><geometry><box size="0.6 0.6 0.2"/></geometry></collision>
<!-- Rail (sized to the lift range) the carriage (mast_link) rides up and down. -->
<visual><origin xyz="0 0 {rail_origin_z}" rpy="0 0 0"/><geometry><box size="0.1 0.1 {rail_height}"/></geometry><material name="dark_grey"/></visual>
<inertial><mass value="20"/><inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.5"/></inertial>
</link>
<joint name="joint_base_y" type="prismatic">
<parent link="slider_x_link"/><child link="slider_y_link"/>
<axis xyz="0 1 0"/><origin xyz="0 0 0" rpy="0 0 0"/>
<limit lower="{y_min}" upper="{y_max}" effort="1000" velocity="2"/>
</joint>

<link name="mast_link">
<visual><origin xyz="0 0 0.1" rpy="0 0 0"/><geometry><box size="0.2 0.2 0.2"/></geometry><material name="dark_grey"/></visual>
<inertial><mass value="5"/><inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/></inertial>
</link>
<joint name="joint_lift" type="prismatic">
<parent link="slider_y_link"/><child link="mast_link"/>
<axis xyz="0 0 1"/><origin xyz="0 0 0.2" rpy="0 0 0"/>
<limit lower="{z_min}" upper="{z_max}" effort="2000" velocity="2"/>
</joint>

<link name="yaw_link">
<visual><geometry><box size="0.15 0.15 0.1"/></geometry><material name="dark_grey"/></visual>
<inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
</link>
<joint name="joint_yaw" type="revolute">
<parent link="mast_link"/><child link="yaw_link"/>
<!-- Rotates the whole arm to face whichever wall is being painted. -->
<axis xyz="0 0 1"/><origin xyz="0 0 0.15" rpy="0 0 0"/>
<limit lower="-3.15" upper="3.15" effort="500" velocity="2"/>
</joint>

<link name="upper_arm_link">
<visual><origin xyz="0 0 0.5" rpy="0 0 0"/><geometry><cylinder radius="0.05" length="1.0"/></geometry><material name="grey"/></visual>
<collision><origin xyz="0 0 0.5" rpy="0 0 0"/><geometry><cylinder radius="0.05" length="1.0"/></geometry></collision>
<inertial><mass value="2"/><inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/></inertial>
</link>
<joint name="joint_shoulder" type="revolute">
<parent link="yaw_link"/><child link="upper_arm_link"/>
<axis xyz="1 0 0"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
<!-- Restricted to +-~90 deg so IK cannot fold the arm back through the floor. -->
<limit lower="-1.6" upper="1.6" effort="500" velocity="2"/>
</joint>

<link name="forearm_link">
<visual><origin xyz="0 0 0.5" rpy="0 0 0"/><geometry><cylinder radius="0.045" length="1.0"/></geometry><material name="grey"/></visual>
<collision><origin xyz="0 0 0.5" rpy="0 0 0"/><geometry><cylinder radius="0.045" length="1.0"/></geometry></collision>
<inertial><mass value="1.5"/><inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.01"/></inertial>
</link>
<joint name="joint_elbow" type="revolute">
<parent link="upper_arm_link"/><child link="forearm_link"/>
<axis xyz="1 0 0"/><origin xyz="0 0 1.0" rpy="0 0 0"/>
<limit lower="-1.6" upper="1.6" effort="500" velocity="2"/>
</joint>

<link name="roller_link">
<visual><geometry><cylinder radius="{roller_radius}" length="{roller_length}"/></geometry><material name="roller_default"/></visual>
<collision><geometry><cylinder radius="{roller_radius}" length="{roller_length}"/></geometry></collision>
<inertial><mass value="0.5"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
</link>
<joint name="joint_roller_orient" type="revolute">
<parent link="forearm_link"/><child link="roller_link"/>
<axis xyz="0 1 0"/><origin xyz="0 0 1.0" rpy="0 0 0"/>
<limit lower="-1.6" upper="1.6" effort="200" velocity="3"/>
</joint>
</robot>
"""


class Robot:
    ROOM_HALF = 5.0
    # Paint roller (end effector) dimensions; the wall grid is sized to match this footprint.
    ROLLER_RADIUS = 0.5
    ROLLER_LENGTH = 0.5

    def __init__(self, start_xy=(0.0, 0.0), max_lift=9.0):
        self.max_lift = max_lift
        urdf_path = self._write_urdf()
        self.id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
        os.remove(urdf_path)

        name_to_index = {}
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            name_to_index[info[1].decode("utf-8")] = i

        self.joint_names = ["joint_base_x", "joint_base_y", "joint_lift", "joint_yaw",
                            "joint_shoulder", "joint_elbow", "joint_roller_orient"]
        self.joint_indices = [name_to_index[n] for n in self.joint_names]
        self.roller_link_index = name_to_index["joint_roller_orient"]

        self.lower_limits = []
        self.upper_limits = []
        for idx in self.joint_indices:
            info = p.getJointInfo(self.id, idx)
            self.lower_limits.append(info[8])
            self.upper_limits.append(info[9])
        self.joint_ranges = [u - l for l, u in zip(self.lower_limits, self.upper_limits)]
        self.rest_poses = [0.0] * len(self.joint_indices)
        # Room center (x=0, y=0), joints neutral; used to send the robot home when done.
        self.home_joints = [0.0] * len(self.joint_indices)

        start_joints = [start_xy[0], start_xy[1]] + [0.0] * (len(self.joint_indices) - 2)
        self.set_joint_targets(start_joints, instant=True)

    def _write_urdf(self):
        z_min, z_max = 0.0, self.max_lift
        platform_top = 0.2
        rail_height = (z_max - z_min) + 0.3
        rail_origin_z = platform_top + rail_height / 2.0
        content = URDF_TEMPLATE.format(
            x_min=-self.ROOM_HALF, x_max=self.ROOM_HALF,
            y_min=-self.ROOM_HALF + 0.2, y_max=self.ROOM_HALF - 0.2,
            z_min=z_min, z_max=z_max,
            rail_height=rail_height, rail_origin_z=rail_origin_z,
            roller_radius=self.ROLLER_RADIUS, roller_length=self.ROLLER_LENGTH,
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
        states = p.getJointStates(self.id, self.joint_indices)
        return [s[0] for s in states]

    def get_roller_position(self):
        state = p.getLinkState(self.id, self.roller_link_index, computeForwardKinematics=True)
        return list(state[0])

    def set_roller_color(self, rgba):
        p.changeVisualShape(self.id, self.roller_link_index, rgbaColor=rgba)