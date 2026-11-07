"""Experimental coverage-aware Stage-2 expert. Does not change Stage-1 PatchSweep."""
import numpy as np
from train_sac_demonstrations import PatchSweep
from train_td3_wall import center_bounds


class Stage2PatchSweep(PatchSweep):
    """Dense sweeps with coverage-based recovery and finite progress monitoring."""
    def __init__(self, env):
        super().__init__(env)
        # More overlap than the Stage-1 0.12 m lane pitch; include edge lanes.
        width = self.u1 - self.u0
        margin = min(0.018, width / 8)
        count = max(2, int(np.ceil(max(0.0, width - 2 * margin) / 0.075)) + 1)
        # The target mask describes paintable PIXELS, not feasible roller CENTRES.
        # The old sweep could request centres outside IK/edge bounds indefinitely.
        umin, umax, vmin, vmax = center_bounds(env.wall.length, env.wall.height, 0.)
        self.u0, self.u1 = max(self.u0, umin + .003), min(self.u1, umax - .003)
        self.v0, self.v1 = max(self.v0, vmin + .003), min(self.v1, vmax - .003)
        if self.u0 >= self.u1 or self.v0 >= self.v1:
            raise RuntimeError("No feasible roller-centre rectangle for Stage 2")
        count = max(2, int(np.ceil((self.u1 - self.u0) / .075)) + 1)
        self.lanes = np.linspace(self.u0, self.u1, count)
        self.index = 0
        self.phase = 'align'
        self.recovery_count = 0
        self.last_coverage = int(np.count_nonzero(env.target & env.wall.canvas.painted))
        self.no_progress = 0
        self.recovery_cooldown = 0
        self.max_recoveries = 20
        self.recovery_interval = 100
        self.last_target = None
        # Targeted V2 fix: avoid repeatedly retrying unproductive lanes.
        self.lane_failures = np.zeros(len(self.lanes), dtype=np.int32)
        self.recovery_lane = None
        self.recovery_start_pixels = 0
        self.recovery_age = 0
        self.recovery_log = []
        self._last_pose_uv = np.asarray(env._pose()[0][:2], float)
        self._blocked_steps = 0
        self.boundary_skips = 0

    def _select_recovery_lane(self, env):
        missing = env.target & ~env.wall.canvas.painted
        rows, cols = np.nonzero(missing)
        if not len(rows):
            return None
        ppm = env.wall.canvas.ppm
        u = env.wall.length - (cols + .5) / ppm
        v = env.wall.height - (rows + .5) / ppm
        current = env._pose()[0][:2]
        # Rank lanes by uncovered pixel count within roller footprint, then travel.
        radius = max(0.025, float(env.robot.ROLLER_LENGTH) * 0.40)
        counts = np.asarray([np.count_nonzero(np.abs(u - lane) <= radius) for lane in self.lanes])
        feasible = np.flatnonzero(counts > 0)
        if not len(feasible):
            return None
        score = counts[feasible] / ((1.0 + np.abs(self.lanes[feasible] - current[0]))
                                     * (1.0 + 2.0 * self.lane_failures[feasible]))
        index = int(feasible[int(np.argmax(score))])
        local_v = v[np.abs(u - self.lanes[index]) <= radius]
        if not len(local_v):
            return None
        # Approach the nearer vertical end, then sweep across the missing pixels.
        bottom = float(np.clip(local_v.min(), self.v0, self.v1))
        top = float(np.clip(local_v.max(), self.v0, self.v1))
        start_at_bottom = abs(current[1] - bottom) <= abs(current[1] - top)
        return index, (top if start_at_bottom else bottom)

    def action(self, env):
        coverage = int(np.count_nonzero(env.target & env.wall.canvas.painted))
        if coverage > self.last_coverage:
            self.last_coverage = coverage
            self.no_progress = 0
        else:
            self.no_progress += 1
        if self.recovery_cooldown:
            self.recovery_cooldown -= 1
        # Detect a motor that has stopped at a geometric bound even while
        # the controller reports accepted commands. Do not wait for stall.
        uv = np.asarray(env._pose()[0][:2], float)
        displacement = float(np.linalg.norm(uv - self._last_pose_uv))
        self._last_pose_uv = uv.copy()
        self._blocked_steps = self._blocked_steps + 1 if displacement < .00015 else 0
        if self._blocked_steps >= 24:
            if self.phase in ('recover_align', 'recover_stroke'):
                self._end_recovery(env, coverage, timed_out=True)
            elif self.phase == 'stroke':
                self.index = (self.index + 1) % len(self.lanes)
                self.phase = 'align'
            elif self.phase == 'align':
                self.index = (self.index + 1) % len(self.lanes)
            self.boundary_skips += 1
            self._blocked_steps = 0
            self.no_progress = 0
        if self.phase in ('recover_align', 'recover_stroke'):
            self.recovery_age += 1
            # Abort a recovery that cannot reach its endpoint. Keep the V2
            # phase transitions otherwise unchanged.
            if self.recovery_age >= 240:
                self._end_recovery(env, coverage, timed_out=True)
        if (self.phase not in ('recover_align', 'recover_stroke')
                and self.no_progress >= self.recovery_interval and not self.recovery_cooldown
                and self.recovery_count < self.max_recoveries):
            recovery = self._select_recovery_lane(env)
            if recovery is not None:
                self.index, self.last_target = recovery
                self.phase = 'recover_align'
                self.recovery_lane = self.index
                self.recovery_start_pixels = coverage
                self.recovery_age = 0
                self.recovery_count += 1
                self.recovery_cooldown = 110
                self.no_progress = 0
        pos = env._pose()[0]
        u, v, n = map(float, pos)
        lane = float(self.lanes[self.index % len(self.lanes)])
        if self.phase == 'recover_align':
            # Move in-plane toward selected lane; avoid diagonal motion.
            if abs(u - lane) < .009:
                self.phase = 'recover_stroke'
            target = np.array([lane, v])
            axis = 0
        elif self.phase == 'recover_stroke':
            target_v = float(self.last_target)
            if abs(v - target_v) < .012:
                self._end_recovery(env, coverage, timed_out=False)
            target = np.array([lane, target_v])
            axis = 1
        else:
            # Use the original sweep state machine with the denser lanes.
            return super().action(env)
        planar = np.clip((target - np.array([u, v])) / (.10 * .05), -1., 1.)
        planar[1-axis] = 0.
        normal = float(np.clip((.0595 - n) / (.025 * .05), -1., 1.))
        theta = float(np.clip(-env.command[3] / (.70 * .05), -1., 1.))
        return np.array([planar[0], planar[1], normal, theta], dtype=np.float32)

    def _end_recovery(self, env, coverage, timed_out):
        if self.recovery_lane is not None:
            gain = max(0, int(coverage) - self.recovery_start_pixels)
            # 20 pixels = 2 cm^2 at 100 px/m. A weak recovery should
            # deprioritize that lane, not repeatedly pick it.
            if gain < 20:
                self.lane_failures[self.recovery_lane] += 1
            elif self.lane_failures[self.recovery_lane] > 0:
                self.lane_failures[self.recovery_lane] -= 1
            self.recovery_log.append(dict(step=int(env.steps), lane=int(self.recovery_lane),
                gain_pixels=int(gain), timed_out=bool(timed_out),
                failures=int(self.lane_failures[self.recovery_lane])))
        self.phase = 'align'
        self.index = (self.index + 1) % len(self.lanes)
        self.recovery_lane = None
        self.recovery_age = 0
        self.recovery_cooldown = 30
        self.no_progress = 0
