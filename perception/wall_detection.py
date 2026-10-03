"""Perception: senses the paint roller's proximity/contact with the wall."""
import pybullet as p


class WallDetector:
    def __init__(self, wall, robot, threshold=0.05):
        self.wall = wall
        self.robot = robot
        self.threshold = threshold

    def distance_to_wall(self, position):
        coordinate = position[0] if self.wall.axis == "x" else position[1]
        return abs(coordinate - self.wall.position)

    def is_touching(self):
        # Real contact/closeness between the roller link and the wall body: more robust than
        # a link-origin distance check once the roller is long/thin enough to touch at a tilt.
        points = p.getClosestPoints(self.robot.id, self.wall.body_id, distance=self.threshold,
                                     linkIndexA=self.robot.roller_link_index)
        return len(points) > 0

    def position_to_grid(self, position):
        return self.wall.world_to_grid(position)