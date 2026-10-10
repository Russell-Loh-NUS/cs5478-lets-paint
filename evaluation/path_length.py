"""Path-length evaluation metric: total distance travelled by the end effector."""
import math


class PathLengthTracker:
    def __init__(self):
        self.total_length = 0.0
        self.segments = 0
        self._since_dip = 0.0

    def update(self, start_position, end_position):
        """Total motion of the tool; used for reporting path length."""
        distance = math.dist(start_position, end_position)
        self.total_length += distance
        self.segments += 1

    def update_paint_distance(self, start_position, end_position):
        """Distance travelled while the roller is in wall contact and painting.

        This is what should drive redip decisions: only spent paint during actual coverage
        counts against the paint budget, not travel between targets or retract/lift motions.
        """
        distance = math.dist(start_position, end_position)
        self._since_dip += distance

    def get_total_length(self):
        return self.total_length

    def distance_since_dip(self):
        return self._since_dip

    def reset_dip_counter(self):
        self._since_dip = 0.0
