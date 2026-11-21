#!/usr/bin/env python3
"""Stage 4 opening-aware wall-painting DAgger.

Run from sac_project/: python -m dagger.stage_4.train --help
This fine-tunes a Stage 3 actor without overwriting its checkpoints.
"""
import argparse
import json
import math
import time
import sys
from pathlib import Path
from functools import partial

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (str(ROOT), str(HERE)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from train_stage3_goal_dagger_v4 import Stage3TrainEnv, clone, _encode
from stage3_bottom_expert import Stage3FullWallSweep
from train_td3_wall import WallScene, Config, center_bounds
from stable_baselines3 import SAC


class OpeningExpert(Stage3FullWallSweep):
    """Deterministic segmented sweep; retracts before crossing absent wall material."""
    def __init__(self, env, pitch=.075, clearance=.035):
        super().__init__(env)
        self.cleanup_mode = True
        self._cleanup_initialized = True
        self._bottom_fallback_active = False
        self.cleanup_phase = 'lift'
        self.transit_normal = .12
        self.cleanup_target = None
        self.clearance = clearance
        self.segments = self._segments(env, pitch)
        # Visit nearby segments first, avoiding long unpainted travel at reset.
        self.segments = self._order_segments(self.segments, env._pose()[0][:2])
        self.segment_index = 0
        self.phase_counts = {key: 0 for key in ('lift','rotate','align','approach','lower','stroke')}
        self.skipped_segments = 0
        self.phase_start = env.steps
        self.phase = 'align'
        self.index = 0
        self.cleanup_goal = env._pose()[0][:2].copy()
        self.cleanup_pass = 0
        self.max_cleanup_passes = 3
        self.cleanup_history = []
        self.last_progress = env.steps
        self.last_painted = int((env.wall.canvas.painted & env.target).sum())
        self.segment_last_progress = env.steps
        self.segment_last_distance = float("inf")
        self.exhausted = False

    def _segments(self, env, pitch):
        wall = env.wall
        u0, u1, v0, v1 = center_bounds(wall.length, wall.height, 0.)
        openings = wall.openings
        segments = []
        for lane in np.arange(u0 + .005, u1 - .004, pitch):
            # Avoid any opening within the full roller's horizontal footprint.
            intervals = [(v0, v1)]
            for o in openings:
                if not (o.u - self.clearance - .10 <= lane <= o.u + o.width + self.clearance + .10):
                    continue
                lo = max(v0, o.v - self.clearance - .07)
                hi = min(v1, o.v + o.height + self.clearance + .07)
                intervals = [(a, b) for x, y in intervals for a, b in
                             ((x, min(y, lo)), (max(x, hi), y)) if b-a >= .12]
            for a, b in intervals:
                # Sample IK feasibility at the endpoints and interior.
                samples = np.linspace(a, b, max(3, int((b-a)/.08)+1))
                if all(env._ik(np.array([lane, z, .0595, 0.])) is not None for z in samples):
                    segments.append((np.array([lane, a]), np.array([lane, b])))
        return segments

    @staticmethod
    def _order_segments(segments, start):
        remaining = list(segments)
        ordered = []
        current = np.asarray(start, float)
        while remaining:
            i = min(range(len(remaining)), key=lambda j: min(
                np.linalg.norm(remaining[j][0]-current),
                np.linalg.norm(remaining[j][1]-current)))
            a, b = remaining.pop(i)
            if np.linalg.norm(b-current) < np.linalg.norm(a-current):
                a, b = b, a
            ordered.append((a, b))
            current = b
        return ordered

    def _cleanup_segments(self, env):
        """Plan short, IK-feasible strokes over still-missing target pixels.

        Only use fully solid roller footprints; gaps and opening margins are not
        crossed while painting. Both roller orientations are considered.
        """
        missing = np.asarray(env.target & ~env.wall.canvas.painted, bool)
        wall = env.wall
        ppm = wall.canvas.ppm
        proposals = []
        for theta, axis in ((0., 1), (math.pi/2, 0)):
            u0,u1,v0,v1 = center_bounds(wall.length, wall.height, theta)
            fixed_lo,fixed_hi = (u0,u1) if axis == 1 else (v0,v1)
            moving_lo,moving_hi = (v0,v1) if axis == 1 else (u0,u1)
            fixed_values = np.arange(fixed_lo+.005, fixed_hi, .065)
            moving_values = np.arange(moving_lo+.005, moving_hi, .025)
            for fixed in fixed_values:
                run = []
                for moving in moving_values:
                    u,v = (fixed,moving) if axis == 1 else (moving,fixed)
                    command = np.array([u,v,.0595,theta])
                    feasible = env._ik(command) is not None
                    # Check full cylinder footprint against openings, not just center.
                    if feasible:
                        for o in wall.openings:
                            du = .09*abs(math.cos(theta))+.06*abs(math.sin(theta))+.012
                            dv = .09*abs(math.sin(theta))+.06*abs(math.cos(theta))+.012
                            if o.u-du <= u <= o.u+o.width+du and o.v-dv <= v <= o.v+o.height+dv:
                                feasible = False
                                break
                    if feasible:
                        col = int((wall.length-u)*ppm)
                        row = int((wall.height-v)*ppm)
                        r0,r1=max(0,row-11),min(missing.shape[0],row+12)
                        c0,c1=max(0,col-11),min(missing.shape[1],col+12)
                        gain = int(missing[r0:r1,c0:c1].sum())
                        feasible = gain > 0
                    if feasible:
                        run.append((u,v,gain))
                    else:
                        if len(run)>=3:
                            proposals.append((theta,axis,run))
                        run=[]
                if len(run)>=3:
                    proposals.append((theta,axis,run))
        segments=[]
        for theta,axis,run in proposals:
            start=np.array(run[0][:2],float)
            end=np.array(run[-1][:2],float)
            if np.linalg.norm(end-start)<.065: continue
            score=sum(x[2] for x in run)
            segments.append((start,end,float(theta),score))
        # Prioritize high missing-pixel yield; cap duplicate cleanup routes.
        segments.sort(key=lambda x:x[3],reverse=True)
        return segments[:160]

    def _advance(self, env):
        self.segment_index += 1
        self.cleanup_phase='lift'
        self.phase_start=env.steps
        self.segment_last_progress=env.steps
        self.segment_last_distance=float('inf')

    def action(self, env):
        pos=env._pose()[0]
        self.phase_counts[self.cleanup_phase]=self.phase_counts.get(self.cleanup_phase,0)+1
        painted=int((env.wall.canvas.painted & env.target).sum())
        if painted>self.last_painted:
            self.last_painted=painted
            self.last_progress=env.steps
        if self.segment_index>=len(self.segments):
            coverage=painted/max(1,int(env.target.sum()))
            if coverage >= .98 or self.cleanup_pass>=self.max_cleanup_passes:
                self.exhausted=True
                self.cleanup_phase='stroke'
                self.cleanup_goal=pos[:2].copy()
                return np.zeros(4,np.float32)
            candidates=self._cleanup_segments(env)
            if not candidates:
                self.exhausted=True
                self.cleanup_goal=pos[:2].copy()
                return np.zeros(4,np.float32)
            self.cleanup_pass+=1
            self.cleanup_history.append(dict(pass_index=self.cleanup_pass,step=env.steps,
                coverage=coverage,segments=len(candidates)))
            self.segments=self._order_segments([(a,b) for a,b,theta,gain in candidates],pos[:2])
            # Resolve each ordered segment's orientation from its endpoints.
            self.segment_thetas=[math.pi/2 if abs(b[0]-a[0])>abs(b[1]-a[1]) else 0.
                                 for a,b in self.segments]
            self.segment_index=0
            self.cleanup_phase='lift'
            self.phase_start=env.steps
            self.segment_last_progress=env.steps
            self.segment_last_distance=float('inf')
        start,end=self.segments[self.segment_index]
        theta=(self.segment_thetas[self.segment_index] if self.cleanup_pass else 0.)
        self.cleanup_target=(0.,('segment',self.cleanup_pass,self.segment_index),theta,
                             0 if theta else 1,start,end,0)
        # Do not spend thousands of steps in a dead stroke. A short no-progress
        # timeout skips only this segment; cleanup can revisit missing pixels.
        distance=float(np.linalg.norm(pos[:2]-end))
        if distance<self.segment_last_distance-.004:
            self.segment_last_distance=distance
            self.segment_last_progress=env.steps
        if (env.steps-self.phase_start>2200 or
            (self.cleanup_phase=='stroke' and env.steps-self.segment_last_progress>220)):
            self.skipped_segments+=1
            self._advance(env)
            return self._control(env,pos[:2],1,theta,self.transit_normal)
        if self.cleanup_phase=='lift':
            self.cleanup_goal=pos[:2].copy()
            if pos[2]>=self.transit_normal-.008 and env.command[2]>=self.transit_normal-.005:
                self.cleanup_phase='rotate';self.phase_start=env.steps
            else:
                return self._control(env,self.cleanup_goal,1,float(env.command[3]),self.transit_normal)
        if self.cleanup_phase=='rotate':
            self.cleanup_goal=pos[:2].copy()
            err=(float(env.command[3])-theta+math.pi)%(2*math.pi)-math.pi
            if abs(err)<.025:
                self.cleanup_phase='align';self.phase_start=env.steps
            else:
                return self._control(env,self.cleanup_goal,1,theta,self.transit_normal)
        axis=1 if theta==0 else 0
        if self.cleanup_phase=='align':
            self.cleanup_goal=np.array([start[0],pos[1]]) if axis==1 else np.array([pos[0],start[1]])
            if abs(pos[1-axis]-start[1-axis])<.012:
                self.cleanup_phase='approach';self.phase_start=env.steps
            else:
                return self._control(env,self.cleanup_goal,1-axis,theta,self.transit_normal)
        if self.cleanup_phase=='approach':
            self.cleanup_goal=start.copy()
            if abs(pos[axis]-start[axis])<.014:
                self.cleanup_phase='lower';self.phase_start=env.steps
            else:
                return self._control(env,self.cleanup_goal,axis,theta,self.transit_normal)
        if self.cleanup_phase=='lower':
            self.cleanup_goal=start.copy()
            if abs(pos[2]-.0595)<.003 and abs(env.command[2]-.0595)<.003:
                self.cleanup_phase='stroke';self.phase_start=env.steps
                self.segment_last_progress=env.steps
                self.segment_last_distance=float('inf')
            else:
                return self._control(env,self.cleanup_goal,axis,theta,.0595)
        self.cleanup_goal=end.copy()
        if abs(pos[axis]-end[axis])<.014:
            self._advance(env)
            return self._control(env,pos[:2],axis,theta,self.transit_normal)
        return self._control(env,self.cleanup_goal,axis,theta,.0595)


class Stage4Env(Stage3TrainEnv):
    def __init__(self, walls, episode_steps=30000, room_seed=42):
        self.room_seed = room_seed
        super().__init__(walls, episode_steps=episode_steps)
        self.scene_factory = partial(self._make_scene)

    def _make_scene(self, rng, config, client, gui=False):
        cfg = Config(walls=config.walls, layout='random', randomize_size=True,
                     max_episode_steps=config.max_episode_steps)
        return WallScene(np.random.default_rng(self.room_seed), cfg, client=client, gui=gui)

    def _prepare(self):
        if self.expert is not None and not isinstance(self.expert, OpeningExpert):
            self.expert = OpeningExpert(self)
        super()._prepare()


def run_episode(wall, room_seed, episode_seed, steps, model=None, collect=False, beta=0.15, cap=3000,
                stall_seconds=180., idle_contact_seconds=30., gui=False, policy_mode='hybrid',
                round_index=1):
    """Student-driven rollout: expert labels every visited state BEFORE the student's action."""
    env = Stage4Env([wall], episode_steps=steps, room_seed=room_seed)
    env.gui = bool(gui)
    env.render_mode = 'human' if gui else None
    env.config.stall_seconds = stall_seconds
    env.config.idle_contact_seconds = idle_contact_seconds
    rng = np.random.default_rng(episode_seed + 123)
    # Reserve half of each episode's memory for hard transitions and recovery.
    pools = {0: [], 1: [], 2: [], 3: []}
    seen = {k: 0 for k in pools}
    # Real phase buckets: painting and transitions must both be represented.
    quotas = {0: cap//2, 1: 0, 2: cap//3, 3: cap-cap//2-cap//3}
    phase_counts = {}
    student_steps = expert_steps = 0
    stroke_without_gain = 0
    last_painted_count = 0
    info = {}
    try:
        obs, info = env.reset(seed=episode_seed, options={'wall': wall})
        for i in range(steps):
            label = np.asarray(env.expert_action(), np.float32)
            phase_name = env.expert.cleanup_phase
            phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
            if collect:
                # The expert's planned phase is known, but student deviations may
                # require corrective labels. Record both nonpainting and painting.
                phase = 2 if phase_name == 'stroke' else 1
                bucket = 0 if phase_name == 'stroke' else 3
                if phase_name in ('rotate', 'align', 'approach', 'lower'):
                    bucket = 2
                item = _encode(obs)
                item.update(action=label.copy(), bucket=np.uint8(bucket), phase=np.uint8(phase))
                seen[bucket] += 1
                pool = pools[bucket]
                if len(pool) < quotas[bucket]:
                    pool.append(item)
                elif quotas[bucket]:
                    j = int(rng.integers(seen[bucket]))
                    if j < quotas[bucket]:
                        pool[j] = item
            # Stage 4 curriculum: collect full expert trajectories before exposing
            # the learner to phases it cannot yet complete. Never train exclusively
            # on failed align states. Hybrid evaluation retains the expert planner's
            # reliable safety-critical transitions while testing neural strokes.
            painted_count = int((env.wall.canvas.painted & env.target).sum())
            if painted_count > last_painted_count:
                stroke_without_gain = 0
            elif phase_name == 'stroke':
                stroke_without_gain += 1
            else:
                stroke_without_gain = 0
            last_painted_count = painted_count
            if model is None or policy_mode == 'expert':
                use_expert = True
            elif collect:
                if round_index == 1:
                    use_expert = True
                elif phase_name != 'stroke':
                    use_expert = (rng.random() < max(.90, beta))
                else:
                    use_expert = (rng.random() < beta)
            elif policy_mode == 'hybrid':
                use_expert = phase_name != 'stroke' or stroke_without_gain >= 60
            else:
                use_expert = False
            if use_expert:
                action = label
                expert_steps += 1
            else:
                action = model.predict(obs, deterministic=True)[0]
                student_steps += 1
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        report = dict(wall=wall, room_seed=room_seed, episode_seed=episode_seed,
                      steps=i+1, coverage=float(info['target_coverage']),
                      whole_wall_coverage=float(info['wall_coverage']),
                      success=bool(info['is_success']), reason=str(info['reason']),
                      student_steps=student_steps, expert_steps=expert_steps,
                      phase_counts=phase_counts, collected_by_bucket={str(k):len(v) for k,v in pools.items()},
                      segments=len(env.expert.segments), segments_reached=env.expert.segment_index,
                      final_phase=env.expert.cleanup_phase, policy_mode=policy_mode)
        print('[stage4-dagger] '+json.dumps(report), flush=True)
        return report, [x for pool in pools.values() for x in pool]
    finally:
        env.close()


def pack_rows(rows):
    return {k:np.stack([x[k] for x in rows]) for k in
            ('state','global_map','local_map','action','bucket','phase')}


def load_episodes(directory):
    paths=sorted(directory.glob('episode_*.npz'))
    if not paths:
        raise RuntimeError('No saved episodes found in '+str(directory))
    chunks=[]
    for path in paths:
        with np.load(path, allow_pickle=False) as f:
            chunks.append({k:f[k].copy() for k in ('state','global_map','local_map','action','bucket','phase')})
    return {k:np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['validate-expert','train','evaluate','fit-saved'], default='validate-expert')
    p.add_argument('--stage3-actor', type=Path, default=HERE/'runs/stage3_dagger_v4/best_actor.zip')
    p.add_argument('--model', type=Path)
    p.add_argument('--gui', action='store_true')
    p.add_argument('--policy-mode', choices=['hybrid','student','expert'], default='student',
                   help='Hybrid uses opening-aware expert transitions and neural strokes; student is fully autonomous')
    p.add_argument('--out', type=Path, default=HERE/'runs/stage4_openings')
    p.add_argument('--room-seeds', type=int, nargs='+', default=[42,43,44,45])
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[901,902,903,904])
    p.add_argument('--walls', nargs='+', choices=['south','east','north','west'], default=['south','east','north','west'])
    p.add_argument('--episode-steps', type=int, default=50000)
    p.add_argument('--stall-seconds', type=float, default=180.,
                   help='No-paint deadline for opening-aware travel (seconds)')
    p.add_argument('--idle-contact-seconds', type=float, default=30.,
                   help='Stationary contact deadline (seconds)')
    p.add_argument('--epochs-per-round', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--bc-lr', type=float, default=3e-5)
    p.add_argument('--axis-weights', type=float, nargs=4, default=[1.6,1.,1.4,2.5])
    p.add_argument('--focus-fraction', type=float, default=.35)
    p.add_argument('--recovery-weight', type=float, default=2.)
    p.add_argument('--transition-weight', type=float, default=2.5)
    p.add_argument('--grad-clip', type=float, default=5.)
    p.add_argument('--samples-per-episode', type=int, default=3000)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--start-round', type=int, default=2, help='Start at round 2 using existing round_1_actor.zip')
    p.add_argument('--target-coverage', type=float, default=.80)
    p.add_argument('--rollout-steps', type=int, default=12000, help='Hard cap per student-driven training episode')
    p.add_argument('--beta', type=float, default=.7, help='Fraction of expert actions during student-driven collection')
    p.add_argument('--seed', type=int, default=360000)
    p.add_argument('--expert-min-coverage', type=float, default=0.)
    p.add_argument('--skip-expert-validation', action='store_true', help='Legacy flag; training now skips validation by default')
    p.add_argument('--validate-before-train', action='store_true', help='Opt in to full expert validation before training')
    args=p.parse_args()
    if args.start_round < 1 or args.rounds < args.start_round:
        p.error('--rounds must be >= --start-round >= 1')
    args.out.mkdir(parents=True, exist_ok=True)
    validation_path = args.out / 'expert_validation.json'
    if args.mode == 'evaluate':
        if args.policy_mode != 'student':
            p.error('Final Stage 4 evaluation must use --policy-mode student')
        if args.model is None and not args.stage3_actor.is_file():
            p.error('Provide --model or a valid --stage3-actor')
        model = SAC.load(str(args.model or args.stage3_actor), device='cpu')
        results = [run_episode(wall, seed, args.seed + i, args.episode_steps,
                               model=model, stall_seconds=args.stall_seconds,
                               idle_contact_seconds=args.idle_contact_seconds,
                               gui=args.gui, policy_mode=args.policy_mode)[0]
                   for i, seed in enumerate(args.eval_seeds) for wall in args.walls]
        (args.out / 'student_evaluation.json').write_text(json.dumps(results, indent=2))
        rate = sum(x['coverage'] >= args.target_coverage for x in results) / len(results)
        mean = float(np.mean([x['coverage'] for x in results]))
        print(f'[student-only] mean_coverage={mean:.1%} >= {args.target_coverage:.0%}: {rate:.1%} of episodes; expert_actions={sum(x["expert_steps"] for x in results)}',flush=True)
        print('[evaluation] saved student_evaluation.json', flush=True)
        return
    if args.mode == 'validate-expert' or args.validate_before_train:
        results = [run_episode(wall, seed, args.seed + i, args.episode_steps,
                               stall_seconds=args.stall_seconds,
                               idle_contact_seconds=args.idle_contact_seconds,
                               gui=args.gui, policy_mode=args.policy_mode)[0]
                   for i, seed in enumerate(args.room_seeds) for wall in args.walls]
        validation_path.write_text(json.dumps(results, indent=2))
        minimum = min(x['coverage'] for x in results)
        print(f'[expert-gate] minimum coverage={minimum:.1%}, required={args.expert_min_coverage:.1%}', flush=True)
        if args.mode == 'validate-expert':
            return
        if minimum < args.expert_min_coverage:
            raise SystemExit('Expert gate failed. Use --expert-min-coverage 0 to bypass.')
    else:
        # Training does NOT rerun expensive expert validation.
        print('[stage4] Skipping expert validation; collecting DAgger labels directly.', flush=True)
    model=SAC.load(str(args.stage3_actor),device='cpu')
    if model.observation_space['state'].shape != (56,):
        raise RuntimeError('Stage 3 V4 56D checkpoint required')
    if not 0 <= args.beta <= 1 or args.rollout_steps < 1 or args.samples_per_episode < 4:
        p.error('Invalid --beta, --rollout-steps or --samples-per-episode')
    episode_dir=args.out/'dagger_episodes'
    episode_dir.mkdir(parents=True,exist_ok=True)
    if args.mode == 'fit-saved':
        dataset=load_episodes(episode_dir)
        print(f'[stage4] fitting {len(dataset["action"])} saved samples',flush=True)
        clone(model,dataset,args,1)
        model.save(str(args.out/'round_1_actor.zip'))
        print('[checkpoint] '+str(args.out/'round_1_actor.zip'),flush=True)
    else:
        for r in range(args.start_round,args.rounds+1):
            if r > 1:
                prior=args.out/f'round_{r-1}_actor.zip'
                if not prior.is_file():
                    raise FileNotFoundError(f'Missing prerequisite checkpoint: {prior}')
                model=SAC.load(str(prior),device='cpu')
                print(f'[resume] round={r} loaded {prior}',flush=True)
            # Each episode is saved atomically before the next simulation begins.
            # Restarting the command reuses already collected episodes.
            for i,s in enumerate(args.room_seeds):
                for wall in args.walls:
                    path=episode_dir/f'episode_{r}_{s}_{wall}.npz'
                    if path.exists():
                        print('[reuse] '+str(path),flush=True)
                        continue
                    report,rows=run_episode(wall,s,args.seed+r*10000+i,args.rollout_steps,
                        model=model,collect=True,beta=args.beta,cap=args.samples_per_episode,
                        stall_seconds=args.stall_seconds,idle_contact_seconds=args.idle_contact_seconds,
                        round_index=r)
                    if not rows:
                        raise RuntimeError('No labelled student states')
                    temp=path.with_suffix('.tmp.npz')
                    np.savez_compressed(temp,**pack_rows(rows))
                    temp.replace(path)
                    (episode_dir/f'episode_{r}_{s}_{wall}.json').write_text(json.dumps(report,indent=2))
                    print('[saved-data] '+str(path),flush=True)
            dataset=load_episodes(episode_dir)
            if not np.any(dataset['bucket'] == 0):
                raise RuntimeError('No painting samples: refuse to train an align-only actor')
            print(f'[stage4] round={r} collected_samples={len(dataset["action"])}; fitting actor',flush=True)
            clone(model,dataset,args,r)
            path=args.out/f'round_{r}_actor.zip'
            model.save(str(path))
            print('[checkpoint] '+str(path),flush=True)
    model.save(str(args.out/'candidate_actor.zip'))
    print('Candidate only: evaluate on unseen --eval-seeds before deployment.',flush=True)

if __name__=='__main__':
    main()
