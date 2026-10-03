"""Path-length evaluation metric: total distance travelled by the end effector."""
import math


class PathLengthTracker:
    def __init__(self):
        self.total_length = 0.0
        self.segments = 0
        self._since_dip = 0.0

    def update(self, start_position, end_position):
        distance = math.dist(start_position, end_position)
        self.total_length += distance
        self._since_dip += distance
        self.segments += 1

    def get_total_length(self):
        return self.total_length

    def distance_since_dip(self):
        return self._since_dip

    def reset_dip_counter(self):
        self._since_dip = 0.0