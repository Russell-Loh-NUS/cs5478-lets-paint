"""Stage-3 full-wall expert: contact-safe transit and precision cleanup.

V8 retains V7 transit and stroke execution, adding precision search only
when its ordinary cleanup selection has no affordable candidates.

V7 preserves the original sweep and 98% task definition. When moving to a
cleanup stroke, it first lifts the roller from the wall so that travel over
already-painted areas does not burn the 15-second idle-contact deadline.
"""
import numpy as np
from stage2_expert_recovery import Stage2PatchSweep
from train_td3_wall import center_bounds

class Stage3FullWallSweep(Stage2PatchSweep):
    def __init__(self, env):
        super().__init__(env)
        self.cleanup_mode = False
        self.cleanup_phase = 'choose'
        self.cleanup_goal = np.asarray(env._pose()[0][:2], float)
        self.cleanup_target = None
        self.cleanup_attempts = 0
        self.cleanup_log = []
        self.cleanup_total_gain = 0
        self.cleanup_failures = {}
        self.cleanup_blacklist = set()
        self.cleanup_age = 0
        self.cleanup_start_pixels = 0
        self.cleanup_last_pixels = 0
        self.cleanup_last_progress = env.steps
        self.cleanup_entry_step = None
        self._visited_last_lane = False
        # Dedicated low horizontal stroke before generic cleanup.
        self.bottom_phase = 'pending'
        self.bottom_goal = np.asarray(env._pose()[0][:2], float)
        self.bottom_log = []
        self.bottom_start_pixels = 0
        self.bottom_age = 0
        self.bottom_last_progress = 0
        self.bottom_last_u = None
        self.bottom_start = None
        self.bottom_end = None
        self.bottom_height = 0.10
        self._sweep_last_index = self.index
        self._sweep_last_phase = self.phase
        self._sweep_last_change = int(env.steps)
        self._sweep_last_pixels = 0
        self._sweep_last_gain_step = int(env.steps)
        self._cleanup_initialized = False
        self._bottom_fallback_active = False
        self.cleanup_target_start_step = None
        self.cleanup_target_last_gain_step = None
        self.cleanup_target_last_pixels = 0
        self.cleanup_selection_log = []
        self.cleanup_phase_last = None
        self.transit_normal = .105  # roller radius .060 m: ~45 mm extra gap
        self.cleanup_phase_enter_step = int(env.steps)
        self.cleanup_stroke_start_step = None
        self.cleanup_selected_approach_steps = None

    def _control(self, env, goal, axis, theta, normal=.0595):
        """One-axis planar motion with explicit roller clearance during transit."""
        pose = env._pose()[0]
        delta = np.clip((np.asarray(goal)-pose[:2])/.005, -1., 1.)
        delta[1-axis] = 0.
        n = np.clip((float(normal)-pose[2])/.00125, -1., 1.)
        rotation_error = (float(theta)-float(env.command[3])+np.pi)%(2*np.pi)-np.pi
        rotation = np.clip(rotation_error/.035, -1., 1.)
        return np.array([delta[0], delta[1], n, rotation], np.float32)

    def _phase(self, env, phase):
        if phase != self.cleanup_phase:
            self.cleanup_phase = phase
            self.cleanup_phase_enter_step = int(env.steps)

    @staticmethod
    def _transit_ik(env, point, normal, theta):
        return env._ik(np.array([float(point[0]),float(point[1]),float(normal),float(theta)])) is not None

    def _segment_variants(self, env, candidate):
        """Focus long sweeps on intervals that actually contain missing paint.

        The source candidate's feasibility grid was sampled along its entire
        stroke. Test new endpoints and the intermediate points explicitly.
        The footprint remains an optimistic area estimate, not real painting.
        """
        _, key, theta, axis, start, end, full_gain = candidate
        missing = np.asarray(env.target & ~env.wall.canvas.painted, bool)
        ppm = int(env.wall.canvas.ppm)
        L, H = env.wall.length, env.wall.height
        lo, hi = sorted((float(start[axis]), float(end[axis])))
        fixed = float(start[1-axis])
        if axis == 1:
            c0 = max(0,int(np.floor((L-fixed-.09)*ppm)))
            c1 = min(missing.shape[1],int(np.ceil((L-fixed+.09)*ppm)))
            r0 = max(0,int(np.floor((H-hi)*ppm)))
            r1 = min(missing.shape[0],int(np.ceil((H-lo)*ppm)))
            counts = missing[r0:r1,c0:c1].sum(axis=1)
            moves = H-(np.arange(r0,r1)+.5)/ppm
        else:
            r0 = max(0,int(np.floor((H-fixed-.09)*ppm)))
            r1 = min(missing.shape[0],int(np.ceil((H-fixed+.09)*ppm)))
            c0 = max(0,int(np.floor((L-hi)*ppm)))
            c1 = min(missing.shape[1],int(np.ceil((L-lo)*ppm)))
            counts = missing[r0:r1,c0:c1].sum(axis=0)
            moves = L-(np.arange(c0,c1)+.5)/ppm
        if len(counts)==0:
            return []
        active = np.sort(moves[counts>0])
        if not len(active):
            return []
        # Merge small gaps; avoid traversing large already-painted spans.
        groups = np.split(active,np.where(np.diff(active)>.12)[0]+1)
        pose = env._pose()[0][:2]
        variants = []
        for group in groups:
            low = max(lo,float(group[0])-.045)
            high = min(hi,float(group[-1])+.045)
            if high-low < .08:
                mid = .5*(low+high)
                low,high = max(lo,mid-.045),min(hi,mid+.045)
            if high-low < .055:
                continue
            # Endpoints and interior samples need IK at painting distance.
            ok = True
            for t in np.linspace(low,high,max(3,int(np.ceil((high-low)/.06))+1)):
                uv = np.array([fixed,t]) if axis==1 else np.array([t,fixed])
                if not self._transit_ik(env,uv,.0595,theta):
                    ok = False
                    break
            if not ok:
                continue
            p0 = np.array([fixed,low]) if axis==1 else np.array([low,fixed])
            p1 = np.array([fixed,high]) if axis==1 else np.array([high,fixed])
            new_start,new_end = (p0,p1) if np.linalg.norm(pose-p0)<=np.linalg.norm(pose-p1) else (p1,p0)
            predicted = int(counts[(moves>=low)&(moves<=high)].sum())
            if predicted < 30:
                continue
            newkey = tuple(key)+('segment',round(low,3),round(high,3))
            if newkey not in self.cleanup_blacklist:
                variants.append((0.,newkey,theta,axis,new_start,new_end,predicted))
        return variants

    def _candidates(self, env):
        """Rank *swept roller-axis* footprints, not distance to missing pixels.

        theta=0 has a horizontal roller and therefore paints on vertical travel
        (axis=1). theta=pi/2 has a vertical roller and paints on horizontal
        travel (axis=0). A full rectangle is an optimistic footprint estimate;
        physics/contact still determine actual coverage.
        """
        missing = np.asarray(env.target & ~env.wall.canvas.painted, dtype=bool)
        if not missing.any():
            return []
        ppm = env.wall.canvas.ppm
        length, height = env.wall.length, env.wall.height
        pose = env._pose()[0][:2]
        proposals = []
        for theta, axis in ((0., 1), (np.pi / 2, 0)):
            u0, u1, v0, v1 = center_bounds(length, height, theta)
            # Fixed coordinate is perpendicular to travel and lies along roller axis.
            fixed_lo, fixed_hi = (u0, u1) if axis == 1 else (v0, v1)
            move_lo, move_hi = (v0, v1) if axis == 1 else (u0, u1)
            for fixed in np.arange(fixed_lo + .004, fixed_hi - .003, .035):
                # Sample every 4 cm: use only contiguous IK-feasible runs.
                samples = np.arange(move_lo + .004, move_hi - .003, .04)
                if len(samples) < 2:
                    continue
                valid = []
                for t in samples:
                    command = np.array([fixed, t, .0595, theta]) if axis == 1 else np.array([t, fixed, .0595, theta])
                    valid.append(env._ik(command) is not None)
                valid = np.asarray(valid, bool)
                boundaries = np.flatnonzero(np.diff(np.r_[False, valid, False]))
                for lo_idx, hi_idx in zip(boundaries[::2], boundaries[1::2]):
                    if hi_idx - lo_idx < 3:
                        continue
                    a, b = float(samples[lo_idx]), float(samples[hi_idx - 1])
                    if b - a < .08:
                        continue
                    # Actual roller axis length = 18 cm. The painted sweep is
                    # a rectangle: 18 cm perpendicular to motion, travel span
                    # along motion. Rasterize in wall-pixel coordinates.
                    if axis == 1:
                        ul, uh = fixed - .09, fixed + .09
                        vl, vh = a, b
                    else:
                        ul, uh = a, b
                        vl, vh = fixed - .09, fixed + .09
                    c0 = max(0, int(np.floor((length - uh) * ppm)))
                    c1 = min(missing.shape[1], int(np.ceil((length - ul) * ppm)))
                    r0 = max(0, int(np.floor((height - vh) * ppm)))
                    r1 = min(missing.shape[0], int(np.ceil((height - vl) * ppm)))
                    if c1 <= c0 or r1 <= r0:
                        continue
                    predicted = int(missing[r0:r1, c0:c1].sum())
                    if predicted < 30:
                        continue
                    key = (axis, round(float(fixed), 3), round(a, 2), round(b, 2))
                    if key in self.cleanup_blacklist:
                        continue
                    p0 = np.array([fixed, a]) if axis == 1 else np.array([a, fixed])
                    p1 = np.array([fixed, b]) if axis == 1 else np.array([b, fixed])
                    start, end = (p0, p1) if np.linalg.norm(pose-p0) <= np.linalg.norm(pose-p1) else (p1, p0)
                    travel = np.linalg.norm(pose-start) + np.linalg.norm(end-start)
                    score = predicted / (.25 + travel + 3*self.cleanup_failures.get(key, 0))
                    proposals.append((score, key, theta, axis, start, end, predicted))
        return sorted(proposals, key=lambda x: x[0], reverse=True)

    def _precision_candidates(self, env):
        """Find small missed strips after ordinary V7 candidates are exhausted.

        The ordinary planner samples roller-axis lanes at 35 mm spacing and
        requires a long IK-feasible sample run. That may miss narrow residual
        patches. Precision search samples 17.5 mm lanes and short strokes,
        but validates every proposed stroke against the actual IK geometry.
        It is an additional search, not a changed success mask.
        """
        missing = np.asarray(env.target & ~env.wall.canvas.painted, bool)
        if not np.any(missing):
            return []
        ppm = float(env.wall.canvas.ppm)
        length, height = float(env.wall.length), float(env.wall.height)
        pose = np.asarray(env._pose()[0][:2], float)
        raw = []
        for theta, axis in ((0., 1), (np.pi / 2., 0)):
            u0, u1, v0, v1 = center_bounds(length, height, theta)
            fixed_lo, fixed_hi = (u0, u1) if axis == 1 else (v0, v1)
            move_lo, move_hi = (v0, v1) if axis == 1 else (u0, u1)
            if fixed_hi-fixed_lo < .009 or move_hi-move_lo < .06:
                continue
            lanes = np.unique(np.r_[np.arange(fixed_lo + .004, fixed_hi - .002, .0175),
                                     fixed_hi - .004])
            for fixed in lanes:
                if axis == 1:
                    c0 = max(0, int(np.floor((length-fixed-.09)*ppm)))
                    c1 = min(missing.shape[1], int(np.ceil((length-fixed+.09)*ppm)))
                    if c1 <= c0: continue
                    counts = missing[:,c0:c1].sum(axis=1)
                    moves = height - (np.arange(len(counts))+.5)/ppm
                else:
                    r0 = max(0, int(np.floor((height-fixed-.09)*ppm)))
                    r1 = min(missing.shape[0], int(np.ceil((height-fixed+.09)*ppm)))
                    if r1 <= r0: continue
                    counts = missing[r0:r1,:].sum(axis=0)
                    moves = length - (np.arange(len(counts))+.5)/ppm
                mask = (counts > 0) & (moves >= move_lo + .002) & (moves <= move_hi - .002)
                active = np.flatnonzero(mask)
                if not active.size: continue
                # Small groups allow targeting corner/edge remnants, without
                # travelling a full wall length across already-painted areas.
                breaks = np.where(np.abs(np.diff(moves[active])) > .085)[0] + 1
                for group in np.split(active, breaks):
                    low = max(move_lo+.002, float(np.min(moves[group]))-.035)
                    high = min(move_hi-.002, float(np.max(moves[group]))+.035)
                    if high-low < .06:
                        mid=(high+low)/2
                        low=max(move_lo+.002,mid-.035)
                        high=min(move_hi-.002,mid+.035)
                    if high-low < .035: continue
                    gain = int(counts[(moves>=low)&(moves<=high)].sum())
                    if gain < 8: continue
                    p0 = np.array([fixed,low]) if axis==1 else np.array([low,fixed])
                    p1 = np.array([fixed,high]) if axis==1 else np.array([high,fixed])
                    start,end=(p0,p1) if np.linalg.norm(pose-p0)<=np.linalg.norm(pose-p1) else (p1,p0)
                    key = (axis,round(float(fixed),4),round(low,3),round(high,3),'precision')
                    if key in self.cleanup_blacklist: continue
                    route=float(np.sum(np.abs(start-pose))) + .4*(high-low)
                    raw.append((gain/(.3+route),key,theta,axis,start,end,gain))
        # IK evaluations are the expensive part: prioritize nearby paint gain.
        raw.sort(key=lambda c: c[0],reverse=True)
        checked=[]
        for candidate in raw[:160]:
            _,key,theta,axis,start,end,_=candidate
            feasible=True
            for t in np.linspace(start[axis],end[axis],
                                 max(3,int(np.ceil(abs(end[axis]-start[axis])/.04))+1)):
                uv=np.array([start[1-axis],t]) if axis==1 else np.array([t,start[1-axis]])
                if not self._transit_ik(env,uv,.0595,theta):
                    feasible=False
                    break
            if feasible:
                checked.append(candidate)
        return checked

    def _stall_budget(self, env):
        """Conservative remaining control steps under the REAL environment clocks."""
        dt = .05
        no_gain_limit = int(env.config.stall_seconds / dt) if env.config.stall_seconds > 0 else 10**9
        idle_limit = int(env.config.idle_contact_seconds / dt) if env.config.idle_contact_seconds > 0 else 10**9
        no_gain_left = no_gain_limit - (env.steps - env.last_progress_step)
        # Idle-contact clock only runs while touching without adding paint;
        # treat it conservatively when already accumulating idle contact.
        contact_left = idle_limit - env.idle_steps if env.idle_steps else 10**9
        return max(0, min(no_gain_left, contact_left)), max(0, no_gain_left), max(0, contact_left)

    def _estimate_steps(self, env, candidate):
        _, _, theta, axis, start, end, predicted = candidate
        pos = env._pose()[0][:2]
        # Core _drive accepts at most .005m per 20Hz step at full action.
        # Include rotation, approach and a conservative tracking margin.
        rotation = abs((theta - float(env.command[3]) + np.pi) % (2*np.pi) - np.pi)
        rotation_steps = rotation / (.70 * .05)
        approach_steps = np.linalg.norm(start-pos) / .005
        stroke_steps = np.linalg.norm(end-start) / .005
        return int(np.ceil(rotation_steps + approach_steps*1.3 + stroke_steps*1.2 + 30))

    def _choose_stall_aware(self, env, candidates):
        """Rank useful strokes using the *no-paint* transit budget.

        Ordinary cleanup travel maintained wall contact, exhausting the 300-step
        idle-contact deadline. This version lifts before rotating/repositioning;
        consequently the applicable travel limit is the remaining 900-step
        no-new-paint clock. The lift itself must still fit the contact clock.
        """
        _, no_gain_left, contact_left = self._stall_budget(env)
        pose = env._pose()[0]
        rotation_now = float(env.command[3])
        lift_steps = int(np.ceil(max(0.,self.transit_normal-pose[2])/.00125*1.4+12))
        ranked=[]
        segments=0
        skipped_ik=0
        skipped_budget=0
        for candidate in candidates:
            variants=([candidate] if 'precision' in candidate[1]
                      else [candidate]+self._segment_variants(env,candidate))
            segments+=len(variants)-1
            for variant in variants:
                _,key,theta,axis,start,end,gain = variant
                if not self._transit_ik(env,start,self.transit_normal,theta):
                    skipped_ik+=1
                    continue
                # Two-axis repositioning and lower/raise with safety margins.
                rotation = abs((theta-rotation_now+np.pi)%(2*np.pi)-np.pi)
                travel_steps = 1.3*float(np.sum(np.abs(start-pose[:2])))/.005
                rotate_steps = 1.3*rotation/.035
                lower_steps = 1.4*(self.transit_normal-.0595)/.00125
                approach = int(np.ceil(lift_steps+travel_steps+rotate_steps+lower_steps+32))
                # This is a *hard constraint* for travel without newly painted pixels.
                if contact_left < lift_steps+15 or approach+35 >= no_gain_left:
                    skipped_budget+=1
                    continue
                stroke_steps = int(np.ceil(np.linalg.norm(end-start)/.005*1.3))
                # Prefer gain/total travel time, not merely footprint size.
                score=float(gain)/max(1,approach+.4*stroke_steps)
                ranked.append((score,variant,approach,stroke_steps))
        ranked.sort(key=lambda x:x[0],reverse=True)
        row=dict(step=int(env.steps),candidates=len(candidates),segment_variants=segments,
                 affordable=len(ranked),no_gain_left=int(no_gain_left),
                 contact_left=int(contact_left),rejected_budget=skipped_budget,
                 rejected_transit_ik=skipped_ik,
                 chosen_predicted_pixels=int(ranked[0][1][-1]) if ranked else 0,
                 chosen_approach_steps=int(ranked[0][2]) if ranked else None)
        self.cleanup_selection_log.append(row)
        print(f'[cleanup-budget] wall={env.wall.name} step={env.steps} '
              f'no_paint={no_gain_left} idle_contact={contact_left} '
              f'candidates={len(candidates)} segments={segments} viable={len(ranked)} '
              f'gain={row["chosen_predicted_pixels"]} '
              f'ETA={row["chosen_approach_steps"]}',flush=True)
        if ranked:
            self.cleanup_selected_approach_steps = ranked[0][2]
            return ranked[0][1]
        return None

    def _finish(self,env,coverage,reason):
        if self.cleanup_target is not None:
            _,key,theta,axis,start,end,count=self.cleanup_target
            gain=max(0,coverage-self.cleanup_start_pixels)
            self.cleanup_log.append(dict(step=int(env.steps),axis=int(axis),theta=float(theta),
                lane=key[1],predicted_pixels=int(count),gain_pixels=int(gain),reason=reason,
                target_steps=int(env.steps-self.cleanup_target_start_step) if self.cleanup_target_start_step is not None else 0,
                budget_remaining_steps=int(self._stall_budget(env)[0]),
                stroke_steps=(int(env.steps-self.cleanup_stroke_start_step)
                              if self.cleanup_stroke_start_step is not None else 0)))
            self.cleanup_blacklist.add(key)  # One pass per segment; replan from new pixels.
            self.cleanup_total_gain += gain
            if gain < max(20, int(count * .05)):
                self.cleanup_failures[key] = self.cleanup_failures.get(key, 0) + 1
        self.cleanup_phase='choose'
        self.cleanup_target=None
        self.cleanup_age=0
        self.cleanup_target_start_step = None
        self.cleanup_target_last_gain_step = None
        self.cleanup_target_last_pixels = coverage
        self.cleanup_stroke_start_step = None

    def _bottom_action(self, env, coverage):
        """One low horizontal stroke; only then enter ordinary cleanup.

        Uses the same command scaling/contact controller as the tested stroke.
        No mask changes. Failures fall back to existing cleanup.
        """
        pos = env._pose()[0]
        self.bottom_age += 1
        if coverage > self.bottom_last_progress:
            self.bottom_last_progress = coverage
        if self.bottom_phase == 'pending':
            u0, u1, v0, v1 = center_bounds(env.wall.length, env.wall.height, 0.)
            height = max(v0 + .004, self.bottom_height)
            # Choose feasible start and end poses with a modest margin.
            feasible = []
            for u in np.linspace(u0 + .015, u1 - .015, 45):
                if env._ik(np.array([u, height, .0595, 0.])) is not None:
                    feasible.append(float(u))
            if len(feasible) < 2 or feasible[-1] - feasible[0] < .25:
                self.bottom_log.append(dict(reason='no_feasible_stroke', step=int(env.steps)))
                self.bottom_phase = 'done'
                return None
            self.bottom_start = np.array([feasible[0], height])
            self.bottom_end = np.array([feasible[-1], height])
            if abs(pos[0] - self.bottom_end[0]) < abs(pos[0] - self.bottom_start[0]):
                self.bottom_start, self.bottom_end = self.bottom_end, self.bottom_start
            self.bottom_start_pixels = coverage
            self.bottom_last_progress = coverage
            self.bottom_age = 0
            self.bottom_phase = 'rotate'
        if self.bottom_age > 1400:
            self.bottom_log.append(dict(reason='timeout', step=int(env.steps), gain_pixels=int(coverage-self.bottom_start_pixels)))
            self.bottom_phase = 'done'
            return None
        # Keep the observation goal synchronized with this motion.
        self.cleanup_phase = 'rotate' if self.bottom_phase == 'rotate' else 'align' if self.bottom_phase in ('height','position') else 'stroke'
        if self.bottom_phase == 'rotate':
            self.cleanup_goal = pos[:2].copy()
            if abs(float(env.command[3])) < .025:
                self.bottom_phase = 'height'
            else:
                return self._control(env, self.cleanup_goal, 0, 0.)
        if self.bottom_phase == 'height':
            self.cleanup_goal = np.array([pos[0], self.bottom_start[1]])
            if abs(pos[1] - self.bottom_start[1]) < .012:
                self.bottom_phase = 'position'
            else:
                return self._control(env, self.cleanup_goal, 1, 0.)
        if self.bottom_phase == 'position':
            self.cleanup_goal = np.array([self.bottom_start[0], self.bottom_start[1]])
            if abs(pos[0] - self.bottom_start[0]) < .014:
                self.bottom_phase = 'stroke'
            else:
                return self._control(env, self.cleanup_goal, 0, 0.)
        if self.bottom_phase == 'stroke':
            self.cleanup_goal = self.bottom_end.copy()
            if abs(pos[0] - self.bottom_end[0]) < .014:
                self.bottom_log.append(dict(reason='completed', step=int(env.steps), gain_pixels=int(coverage-self.bottom_start_pixels)))
                self.bottom_phase = 'done'
                return None
            return self._control(env, self.cleanup_goal, 0, 0.)
        return None

    def action(self,env):
        coverage = int(np.count_nonzero(env.target & env.wall.canvas.painted))
        if not self.cleanup_mode:
            # Do not interrupt a productive sweep merely because it crossed 90%.
            # Enter recovery only after the last lane, or when the sweep's
            # controller stops advancing and the environment is nearing stall.
            if (self.index, self.phase) != (self._sweep_last_index, self._sweep_last_phase):
                self._sweep_last_index, self._sweep_last_phase = self.index, self.phase
                self._sweep_last_change = int(env.steps)
            if coverage > self._sweep_last_pixels:
                self._sweep_last_pixels = coverage
                self._sweep_last_gain_step = int(env.steps)
            ratio = coverage / max(1, int(env.target_pixels))
            idle = int(env.steps - env.last_progress_step)
            idle_limit = int(env.config.idle_contact_seconds / .05) if env.config.idle_contact_seconds else 10**9
            progress_limit = int(env.config.stall_seconds / .05) if env.config.stall_seconds else 10**9
            main_done = self.index >= len(self.lanes)-1 and self.phase == 'align'
            blocked = (env.steps-self._sweep_last_change >= 120 and
                       env.steps-self._sweep_last_gain_step >= 80)
            # A real environmental stall cannot be bypassed by resetting its
            # timers. Enter cleanup with a margin, or report that there is none.
            approaching_stall = idle >= min(160, max(30, progress_limit-120)) or env.idle_steps >= max(0, idle_limit-120)
            if env.steps >= 500 and (main_done or (ratio >= .85 and blocked and approaching_stall)):
                self.cleanup_mode = True
                self.cleanup_entry_step = int(env.steps)
                self.cleanup_last_pixels = coverage
                self.cleanup_last_progress = int(env.steps)
                self.bottom_log.append(dict(event='cleanup_entered', step=int(env.steps),
                    coverage=float(ratio), trigger='sweep_done' if main_done else 'blocked_sweep'))
                print(f'[stage3-transition] wall={env.wall.name} step={env.steps} '
                      f'coverage={ratio:.1%} trigger={self.bottom_log[-1]["trigger"]}', flush=True)
            else:
                return super().action(env)
        # Rank missing-pixel strokes first; the dedicated bottom pass is a
        # fallback rather than an unconditional, potentially redundant detour.
        if not self._cleanup_initialized:
            self._cleanup_initialized = True
            self.cleanup_phase = 'choose'
            self.bottom_phase = 'pending'
        # If cleanup candidates are exhausted, attempt the verified low stroke.
        if self._bottom_fallback_active:
            if self.bottom_phase != 'done':
                result = self._bottom_action(env, coverage)
                if result is not None:
                    return result
            self.cleanup_goal = np.asarray(env._pose()[0][:2], float)
            return self._control(env, self.cleanup_goal, 0, float(env.command[3]))
        self.cleanup_age += 1
        if coverage > self.cleanup_last_pixels:
            self.cleanup_last_pixels = coverage
            self.cleanup_last_progress = int(env.steps)
        # A stroke gets its own bounded timeout. Long, productive strokes are
        # allowed to finish; physical environment stall clocks are never reset.
        if self.cleanup_target is not None:
            if coverage > self.cleanup_target_last_pixels:
                self.cleanup_target_last_pixels = coverage
                self.cleanup_target_last_gain_step = int(env.steps)
            if self.cleanup_phase == 'stroke':
                stroke_age = int(env.steps-(self.cleanup_stroke_start_step or env.steps))
                no_gain = int(env.steps-self.cleanup_target_last_gain_step)
                if stroke_age > 850 or no_gain > 175:
                    self._finish(env,coverage,'stroke_timeout')
            else:
                elapsed=int(env.steps-self.cleanup_target_start_step)
                if elapsed > 850:  # bounded by the real 900-step no-paint deadline
                    self._finish(env,coverage,'transit_timeout')
        if self.cleanup_phase=='choose':
            candidates=self._candidates(env)
            selected=self._choose_stall_aware(env, candidates)
            if selected is None:
                precision=self._precision_candidates(env)
                print(f'[cleanup-refine] wall={env.wall.name} step={env.steps} '
                      f'precision_candidates={len(precision)} remaining={env.target_pixels-coverage}',
                      flush=True)
                if precision:
                    selected=self._choose_stall_aware(env, precision)
            if selected is None:
                self._bottom_fallback_active = True
                self.cleanup_target = None
                self.bottom_log.append(dict(event='bottom_fallback_entered', step=int(env.steps), coverage=float(coverage/max(1,env.target_pixels))))
                print(f'[bottom-transition] wall={env.wall.name} step={env.steps} trigger=no_affordable_cleanup_candidates', flush=True)
                result = self._bottom_action(env, coverage)
                if result is not None:
                    return result
                self.cleanup_goal=np.asarray(env._pose()[0][:2],float)
                return self._control(env,self.cleanup_goal,0,float(env.command[3]))
            self.cleanup_target=selected
            self.cleanup_attempts+=1
            self.cleanup_start_pixels=coverage
            self.cleanup_age=0
            self.cleanup_target_start_step = int(env.steps)
            self.cleanup_target_last_gain_step = int(env.steps)
            self.cleanup_target_last_pixels = coverage
            self.cleanup_stroke_start_step = None
            self._phase(env,'lift')
            print(f'[cleanup-transit] wall={env.wall.name} step={env.steps} '
                  f'est_approach={self.cleanup_selected_approach_steps} '
                  f'predicted={selected[-1]}',flush=True)
        if self.cleanup_target is None:
            # Defensive guard: no cleanup stroke is selected.
            self.cleanup_phase = 'choose'
            self.cleanup_goal = np.asarray(env._pose()[0][:2], float)
            return self._control(env, self.cleanup_goal, 0, float(env.command[3]))
        _,key,theta,axis,start,end,_=self.cleanup_target
        pos=env._pose()[0]
        elevated=self.transit_normal
        # Physically separate the roller from paint during transfer. Do not
        # falsify contact or reset environment timers; PyBullet handles both.
        if self.cleanup_phase=='lift':
            self.cleanup_goal=pos[:2].copy()
            if pos[2] >= elevated-.008 and env.command[2] >= elevated-.005:
                self._phase(env,'rotate')
            else:
                return self._control(env,self.cleanup_goal,axis,float(env.command[3]),elevated)
        if self.cleanup_phase=='rotate':
            self.cleanup_goal=pos[:2].copy()
            error=(float(theta)-float(env.command[3])+np.pi)%(2*np.pi)-np.pi
            if abs(error)<.025:
                self._phase(env,'align')
            else:
                return self._control(env,self.cleanup_goal,axis,theta,elevated)
        if self.cleanup_phase=='align':
            align_axis=1-axis
            self.cleanup_goal=pos[:2].copy()
            self.cleanup_goal[align_axis]=start[align_axis]
            if abs(pos[align_axis]-start[align_axis])<.012:
                self._phase(env,'approach')
            else:
                return self._control(env,self.cleanup_goal,align_axis,theta,elevated)
        if self.cleanup_phase=='approach':
            self.cleanup_goal=start.copy()
            if abs(pos[axis]-start[axis])<.014:
                self._phase(env,'lower')
            else:
                return self._control(env,self.cleanup_goal,axis,theta,elevated)
        if self.cleanup_phase=='lower':
            self.cleanup_goal=start.copy()
            if abs(pos[2]-.0595)<.003 and abs(env.command[2]-.0595)<.003:
                self._phase(env,'stroke')
                self.cleanup_stroke_start_step=int(env.steps)
                self.cleanup_target_last_gain_step=int(env.steps)
                print(f'[cleanup-stroke] wall={env.wall.name} step={env.steps} '
                      f'predicted={self.cleanup_target[-1]}',flush=True)
            else:
                return self._control(env,self.cleanup_goal,axis,theta,.0595)
        self.cleanup_goal=end.copy()
        if abs(pos[axis]-end[axis])<.014:
            self._finish(env,coverage,'completed')
            return self._control(env,pos[:2],axis,theta,.0595)
        return self._control(env,self.cleanup_goal,axis,theta,.0595)
