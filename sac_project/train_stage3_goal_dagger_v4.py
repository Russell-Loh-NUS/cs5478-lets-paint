#!/usr/bin/env python3
"""Stage-3 V4 full-wall DAgger: phase-focused control imitation + protected baseline.

Run beside validate_stage3_bottom.py and stage3_bottom_expert.py in sac_project.
The V8 planner remains active during learned-policy evaluation: the policy
learns low-level motion commands, not independent coverage planning. No SAC RL update.
Uses separately labelled STUDENT-driven probes to expose states missed by
expert-led DAgger. Preserves the V8 deterministic goal planner, while keeping
independent evaluation student-controlled. The probes are training data only. V4 preserves the initial policy as a
baseline, retains rare turn/lift/lower/phase-switch labels, and trains with
weighted per-axis loss. The 98% task threshold remains unchanged.

The Stage-2 checkpoint is transferred into the Stage-3 policy, including its
CNNs and actor layers. The 48->56 input expansion copies existing feature
weights and zero-initializes weights for eight new V8 transit-phase inputs.
"""
import argparse
import json
import multiprocessing as mp
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import SAC

from validate_stage3_bottom import GoalEnv as _GoalEnv, new_model

WALLS = ('south', 'east', 'north', 'west')
OBS_KEYS = ('state', 'global_map', 'local_map')
DATA_KEYS = OBS_KEYS + ('action', 'bucket', 'phase')
BUCKET_NAMES = ('main', 'cleanup', 'recovery', 'transition')
PHASE_NAMES = ('main', 'cleanup_transit', 'cleanup_stroke', 'bottom_fallback')
STATE_DIM = 56


class Stage3TrainEnv(_GoalEnv):
    """V8 planner goals with explicit lift/turn/travel/lower/paint signals.

    The first 48 state inputs retain the Stage-2 feature layout so that its
    pretrained model can be reused instead of restarting the CNN from scratch.
    """
    def __init__(self, walls, episode_steps=30000):
        super().__init__(walls, episode_steps=episode_steps)
        old_dim = self.observation_space['state'].shape[0]
        if old_dim != 48:
            raise RuntimeError(f'Stage 3 original GoalEnv state changed: {old_dim}, expected 48')
        self.observation_space.spaces['state'] = spaces.Box(
            -np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32)
        self._goal_features = np.zeros(16, np.float32)

    def _prepare(self):
        super()._prepare()  # updates the first 8 planner inputs and cached label
        if not hasattr(self.expert, 'transit_normal') or not hasattr(self.expert, '_precision_candidates'):
            raise RuntimeError('Stage-3 V8 expert is required: lift transit and precision cleanup must both be available')
        base8 = self._goal_features
        pose = self._pose()[0]
        exp = self.expert
        cleanup = bool(exp.cleanup_mode)
        phase = exp.cleanup_phase if cleanup else ''
        # Nonpainting transfer is conducted away from the wall.
        travel = phase in ('lift', 'rotate', 'align', 'approach') and not exp._bottom_fallback_active
        normal_goal = exp.transit_normal if travel else .0595
        if cleanup and not exp._bottom_fallback_active and exp.cleanup_target is not None:
            theta = float(exp.cleanup_target[2])
        elif cleanup and exp._bottom_fallback_active:
            theta = 0.
        else:
            theta = float(self.command[3])
        extra = np.array([
            (float(normal_goal)-float(pose[2]))/.06,
            np.sin(2*theta), np.cos(2*theta),
            float(cleanup and phase == 'lift'),
            float(cleanup and phase == 'rotate'),
            float(cleanup and phase in ('align', 'approach')),
            float(cleanup and phase == 'lower'),
            float(cleanup and phase == 'stroke'),
        ], np.float32)
        self._goal_features = np.concatenate((base8, extra)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        # Base reset calls _obs() before the expert is initialized.
        self._goal_features = np.zeros(16, np.float32)
        return super().reset(seed=seed, options=options)


def _encode(obs):
    # Images are bounded [0,1] and can be stored as uint8 to avoid multi-GB
    # DAgger datasets. The feature encoder receives float32 [0,1] on training.
    return dict(state=np.asarray(obs['state'], np.float32).copy(),
                global_map=np.clip(np.rint(obs['global_map']*255), 0, 255).astype(np.uint8),
                local_map=np.clip(np.rint(obs['local_map']*255), 0, 255).astype(np.uint8))


class Reservoir:
    """Uniform streaming sampling with bounded image-map memory."""
    def __init__(self, cap, rng):
        self.cap, self.rng, self.seen = int(cap), rng, 0
        self.rows = []

    def append(self, obs, action, bucket, phase):
        self.seen += 1
        if len(self.rows) < self.cap:
            idx = len(self.rows)
        else:
            idx = int(self.rng.integers(self.seen))
            if idx >= self.cap:
                return
        entry = _encode(obs)
        entry['action'] = np.asarray(action, np.float32).copy()
        entry['bucket'] = np.uint8(bucket)
        entry['phase'] = np.uint8(phase)
        if idx == len(self.rows):
            self.rows.append(entry)
        else:
            self.rows[idx] = entry


def _packed(rows):
    if not rows:
        raise RuntimeError('No samples collected from episode')
    return {k: np.stack([r[k] for r in rows]) for k in DATA_KEYS}


def _expert_phase(env):
    exp = env.expert
    if not exp.cleanup_mode:
        return 0
    if exp._bottom_fallback_active:
        return 3
    if exp.cleanup_phase == 'stroke':
        return 2
    return 1


def expert_beta(round_index, args):
    """Expert/mixed rollout schedule; each round also has student-heavy probes."""
    if round_index == 1:
        return 1.
    return max(args.beta_min, args.beta - (round_index - 2) * args.beta_step)


def rollout(task):
    """Collect labelled expert, mixed, or student-heavy on-policy episodes.

    In a probe, the student controls consecutive bursts of actions. A short
    expert correction follows each burst, or occurs sooner if an actual
    simulator safety deadline is approaching. We NEVER use an expert action
    to calculate student evaluation coverage.
    """
    (model_path, wall, seed, beta, limit, sample_cap, mode,
     recovery_steps, disagreement_threshold, no_paint_guard, contact_guard,
     student_burst_steps, args_transition_window) = task
    if mode not in ('expert', 'mixed', 'probe'):
        raise ValueError(f'Unsupported rollout mode: {mode}')
    th.set_num_threads(1)
    rng = np.random.default_rng(seed + 8731)
    env = Stage3TrainEnv([wall], episode_steps=limit)
    model = None if mode == 'expert' else SAC.load(str(model_path), device='cpu')
    # Keep rare orientation/contact transitions visible in a *separate* bin.
    # This is a fresh V4 dataset; V3 dataset metadata is not reused.
    if mode == 'expert':
        caps = [sample_cap*65//100, sample_cap*20//100, 0, 0]
    elif mode == 'probe':
        caps = [sample_cap*45//100, sample_cap*10//100, sample_cap*30//100, 0]
    else:
        caps = [sample_cap*50//100, sample_cap*20//100, sample_cap*15//100, 0]
    caps[3] = sample_cap-sum(caps[:3])
    bins = [Reservoir(n, rng) for n in caps]
    expert_actions = 0
    student_actions = 0
    rescue_events = 0
    safety_overrides = 0
    recovery_remaining = 0
    burst_remaining = student_burst_steps
    steps = 0
    observed_phase_steps = [0, 0, 0, 0]
    action_error_sum = np.zeros(4, np.float64)
    action_error_count = 0
    previous_microphase = None
    transition_hold = 0
    info = None
    try:
        obs, _ = env.reset(seed=seed, options={'wall': wall})
        for step in range(limit):
            # The cached label and obs are both prepared from the *same*
            # simulator state in GoalEnv.step/reset.
            label = np.asarray(env.expert_action(), np.float32)
            phase = _expert_phase(env)
            observed_phase_steps[phase] += 1
            no_paint_elapsed = int(env.steps - env.last_progress_step)
            no_paint_limit = int(env.config.stall_seconds/.05) if env.config.stall_seconds > 0 else 10**9
            contact_limit = int(env.config.idle_contact_seconds/.05) if env.config.idle_contact_seconds > 0 else 10**9
            danger = (no_paint_limit-no_paint_elapsed <= no_paint_guard or
                      (env.idle_steps > 0 and contact_limit-env.idle_steps <= contact_guard))

            student_action = None
            diff = 0.
            if model is not None:
                student_action = np.asarray(model.predict(obs, deterministic=True)[0], np.float32)
                err = np.abs(student_action-label)
                action_error_sum += err
                action_error_count += 1
                diff = float(err.mean())
            hard_state = model is not None and diff >= disagreement_threshold

            if mode == 'expert':
                using_expert = True
            elif mode == 'probe':
                # Consecutive student actions are crucial: a Bernoulli 5% student
                # fraction almost never leaves the expert's trajectory.
                using_expert = bool(danger or recovery_remaining > 0)
                if danger and recovery_remaining == 0:
                    recovery_remaining = recovery_steps
                    rescue_events += 1
                    safety_overrides += 1
                if using_expert:
                    if recovery_remaining > 0:
                        recovery_remaining -= 1
                else:
                    burst_remaining -= 1
                    if burst_remaining <= 0:
                        recovery_remaining = recovery_steps
                        burst_remaining = student_burst_steps
            else:
                using_expert = bool(danger or recovery_remaining > 0 or rng.random() < beta)
                if danger and recovery_remaining == 0:
                    recovery_remaining = recovery_steps
                    rescue_events += 1
                    safety_overrides += 1
                if recovery_remaining > 0:
                    recovery_remaining -= 1
                if not using_expert and hard_state and recovery_remaining == 0:
                    recovery_remaining = recovery_steps
                    rescue_events += 1

            # An explicit motion-phase switch is rare in long 20-Hz sweeps.
            # Keep a short window AFTER the switch and any meaningful roller
            # rotation, lift/lower command, or transit phase. The expert still
            # supplies the label; student-only evaluation is unchanged.
            exp = env.expert
            microphase = ('bottom:' + str(exp.bottom_phase) if exp._bottom_fallback_active
                          else 'cleanup:' + str(exp.cleanup_phase) if exp.cleanup_mode
                          else 'main:' + str(exp.phase))
            if previous_microphase is not None and microphase != previous_microphase:
                transition_hold = max(transition_hold, args_transition_window)
            previous_microphase = microphase
            special_motion = (transition_hold > 0 or abs(float(label[3])) >= 0.12
                              or abs(float(label[2])) >= 0.35
                              or microphase in ('cleanup:lift', 'cleanup:rotate', 'cleanup:lower'))
            if transition_hold:
                transition_hold -= 1
            normal_bucket = 1 if phase != 0 else 0
            bins[normal_bucket].append(obs, label, normal_bucket, phase)
            if special_motion:
                bins[3].append(obs, label, 3, phase)
            # Probe samples are sourced from states that STUDENT actions
            # actually generated; make these observations a substantial
            # training subset, instead of a few isolated hard examples.
            corrective = (mode == 'probe' or
                          (mode == 'mixed' and (hard_state or danger or recovery_remaining > 0)))
            if corrective:
                bins[2].append(obs, label, 2, phase)

            action = label if using_expert else student_action
            expert_actions += int(using_expert)
            student_actions += int(not using_expert)
            obs, _, ended, truncated, info = env.step(action)
            steps = step+1
            if ended or truncated:
                break
        rows = [entry for buf in bins for entry in buf.rows]
        data = _packed(rows)
        summary = dict(mode=mode, wall=wall, seed=int(seed), steps=steps,
                       coverage=float(info['target_coverage']),
                       success=bool(info['is_success']), reason=str(info['reason']),
                       expert_fraction=expert_actions/max(1,steps),
                       student_fraction=student_actions/max(1,steps),
                       action_mae=float(action_error_sum.sum()/max(1,4*action_error_count)),
                       action_mae_by_axis=(action_error_sum/max(1,action_error_count)).round(4).tolist(),
                       labeled_samples=len(rows),
                       main_samples=len(bins[0].rows),
                       cleanup_samples=len(bins[1].rows),
                       recovery_samples=len(bins[2].rows),
                       transition_samples=len(bins[3].rows),
                       rescue_events=rescue_events,
                       safety_overrides=safety_overrides,
                       phase_steps={PHASE_NAMES[i]:observed_phase_steps[i] for i in range(4)},
                       cleanup_attempts=int(env.expert.cleanup_attempts))
        return data, summary
    finally:
        env.close()


def collect(tasks, workers):
    if workers == 1:
        return [rollout(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        return list(pool.map(rollout, tasks))


def merge(old, new, cap, seed):
    if old is None:
        result = new
    else:
        result = {k: np.concatenate((old[k], new[k])) for k in DATA_KEYS}
    n = len(result['action'])
    if n > cap:
        # Fair sampling across all rounds; bound CPU and RAM during cloning.
        ix = np.sort(np.random.default_rng(seed).choice(n, size=cap, replace=False))
        result = {k: v[ix] for k, v in result.items()}
    return result


def load_dataset(path):
    with np.load(path) as f:
        data = {k: f[k].copy() for k in DATA_KEYS}
    n = len(data['action'])
    if n == 0 or any(len(v) != n for v in data.values()):
        raise ValueError('Invalid or empty DAgger dataset')
    if data['state'].shape[1] != STATE_DIM:
        raise ValueError(f'Expected {STATE_DIM}-D Stage-3 observations, got {data["state"].shape}')
    if np.any(data['bucket'] > 3) or np.any(data['phase'] > 3):
        raise ValueError('Invalid bucket/phase metadata')
    return data


def warm_start(model, stage2_path):
    old = SAC.load(str(stage2_path), device='cpu')
    old_dim = old.observation_space['state'].shape[0]
    if old_dim != 48 or old.action_space.shape != (4,):
        raise ValueError(f'Stage-2 actor incompatible (state={old_dim}, action={old.action_space.shape})')
    incoming = old.actor.state_dict()
    current = model.actor.state_dict()
    copied = 0
    expanded = []
    for name, dst in current.items():
        if name not in incoming:
            raise ValueError(f'Unmatched Stage-2 actor weight: {name}')
        src = incoming[name]
        if src.shape == dst.shape:
            current[name] = src.clone()
            copied += 1
        elif (src.ndim == 2 and dst.ndim == 2 and
              dst.shape[0] == src.shape[0] and
              dst.shape[1] == src.shape[1] + 8 and
              name.endswith('s.0.weight')):
            tmp = th.zeros_like(dst)
            tmp[:, :src.shape[1]] = src
            current[name] = tmp
            expanded.append(name)
        else:
            raise ValueError(f'Cannot transfer {name}: Stage2={tuple(src.shape)}, Stage3={tuple(dst.shape)}')
    if len(expanded) != 1:
        raise RuntimeError(f'Expected exactly one expanded state-input weight, got {expanded}')
    model.actor.load_state_dict(current, strict=True)
    print(f'[warm-start] copied {copied} actor tensors, expanded {expanded[0]} 48->56; '
          f'critics/replay/optimizer NOT transferred', flush=True)


def _sample_epoch_indices(data, rng, focus_fraction):
    """Combine uniform exposure and extra rare motion-phase samples.

    Each training epoch has exactly N examples. A uniform part retains main
    sweep variety; the focused part samples with replacement from rarer
    transitions and large expert roller-rotation commands.
    """
    n = len(data['action'])
    n_focus = int(round(n*focus_fraction))
    n_uniform = n-n_focus
    uniform = rng.permutation(n)[:n_uniform]
    if n_focus:
        weight = np.ones(n, np.float64)
        weight[data['bucket'] == 3] *= 4.0
        weight[data['phase'] == 1] *= 1.8
        weight[np.abs(data['action'][:, 3]) >= .12] *= 2.8
        weight[np.abs(data['action'][:, 2]) >= .35] *= 1.7
        weight /= weight.sum()
        focus = rng.choice(n, size=n_focus, replace=True, p=weight)
        indices = np.concatenate((uniform, focus))
    else:
        indices = uniform
    rng.shuffle(indices)
    return indices


def clone(model, data, args, round_index):
    """Phase-focused DAgger behavior cloning with larger rotation/planar gradients.

    The action is still the four normalized PyBullet controls. We never
    substitute expert actions during independent evaluation.
    """
    actor = model.actor
    actor.train()
    optimizer = th.optim.Adam(actor.parameters(), lr=args.bc_lr)
    rng = np.random.default_rng(args.seed + round_index)
    n = len(data['action'])
    axis_weight = th.as_tensor(args.axis_weights, dtype=th.float32, device=model.device)
    if axis_weight.shape != (4,) or not th.isfinite(axis_weight).all() or (axis_weight <= 0).any():
        raise ValueError('--axis-weights must provide four finite positive numbers')
    for epoch in range(1, args.epochs_per_round + 1):
        indices = _sample_epoch_indices(data, rng, args.focus_fraction)
        mse_total = 0.
        action_abs_sum = np.zeros(4, np.float64)
        by_bucket_sum = np.zeros(len(BUCKET_NAMES), np.float64)
        by_bucket_n = np.zeros(len(BUCKET_NAMES), np.int64)
        by_phase_sum = np.zeros(len(PHASE_NAMES), np.float64)
        by_phase_n = np.zeros(len(PHASE_NAMES), np.int64)
        for start in range(0, len(indices), args.batch_size):
            ix = indices[start:start+args.batch_size]
            obs = {'state': th.as_tensor(data['state'][ix], dtype=th.float32, device=model.device),
                   'global_map': th.as_tensor(data['global_map'][ix], device=model.device).float()/255.,
                   'local_map': th.as_tensor(data['local_map'][ix], device=model.device).float()/255.}
            label = th.as_tensor(data['action'][ix], dtype=th.float32, device=model.device)
            prediction = actor(obs, deterministic=True)
            squared = (prediction-label).square()
            # Increase the influence of horizontal movement and roller rotation.
            # Normalized by sum(weights) so loss magnitude is stable.
            per_sample = (squared*axis_weight).sum(dim=1)/axis_weight.sum()
            bucket = th.as_tensor(data['bucket'][ix].astype(np.int64), device=model.device)
            sample_weight = th.ones_like(per_sample)
            sample_weight = th.where(bucket == 2, args.recovery_weight, sample_weight)
            sample_weight = th.where(bucket == 3, args.transition_weight, sample_weight)
            loss = (per_sample*sample_weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
            optimizer.step()
            with th.no_grad():
                errs = squared.detach().cpu().numpy()
            unweighted = errs.mean(axis=1)
            mse_total += float(unweighted.sum())
            action_abs_sum += np.sqrt(errs).sum(axis=0)
            bucket_numpy = data['bucket'][ix]
            for j in range(len(BUCKET_NAMES)):
                mask = bucket_numpy == j
                by_bucket_sum[j] += float(unweighted[mask].sum())
                by_bucket_n[j] += int(mask.sum())
            phase_numpy = data['phase'][ix]
            for j in range(len(PHASE_NAMES)):
                mask = phase_numpy == j
                by_phase_sum[j] += float(unweighted[mask].sum())
                by_phase_n[j] += int(mask.sum())
        parts = ' '.join(f'{BUCKET_NAMES[j]}={by_bucket_sum[j]/by_bucket_n[j]:.5f}'
                         if by_bucket_n[j] else f'{BUCKET_NAMES[j]}=n/a'
                         for j in range(len(BUCKET_NAMES)))
        phase_parts = ' '.join(f'{PHASE_NAMES[j]}={by_phase_sum[j]/by_phase_n[j]:.4f}'
                               if by_phase_n[j] else f'{PHASE_NAMES[j]}=n/a'
                               for j in range(len(PHASE_NAMES)))
        axis_mae = np.round(action_abs_sum/max(1, len(indices)), 3).tolist()
        print(f'[bc] round={round_index} epoch={epoch}/{args.epochs_per_round} '
              f'samples={n} sampled_mse={mse_total/max(1,len(indices)):.6f} '
              f'axis_mae={axis_mae} {parts} phases={phase_parts}', flush=True)
    actor.eval()

def evaluate(model, args, seed):
    """Student-only control; expert label used solely for diagnostics."""
    env = Stage3TrainEnv(args.walls, episode_steps=args.episode_steps)
    episodes = []
    try:
        for i in range(args.eval_episodes):
            wall = args.walls[i%len(args.walls)]
            obs, _ = env.reset(seed=seed+i, options={'wall':wall})
            phase_abs = np.zeros((4,4),np.float64)
            phase_count = np.zeros(4,np.int64)
            for step in range(args.episode_steps):
                label = np.asarray(env.expert_action(),np.float32)
                phase = _expert_phase(env)
                action, _ = model.predict(obs, deterministic=True)
                phase_abs[phase] += np.abs(np.asarray(action)-label)
                phase_count[phase] += 1
                # CRITICAL: only the student's action is applied to PyBullet.
                obs, _, done, trunc, info = env.step(action)
                if done or trunc:
                    break
            phase_mae = {PHASE_NAMES[j]:round(float(phase_abs[j].mean()/phase_count[j]),4)
                         for j in range(4) if phase_count[j]}
            axis_mae = (phase_abs.sum(axis=0)/max(1,phase_count.sum())).round(4).tolist()
            episode = dict(wall=wall, seed=seed+i, steps=step+1,
                           coverage=float(info['target_coverage']),
                           success=bool(info['is_success']),reason=str(info['reason']),
                           phase_action_mae=phase_mae, action_mae_by_axis=axis_mae)
            episodes.append(episode)
            print(f'[eval] wall={wall} seed={seed+i} coverage={episode["coverage"]:.2%} '
                  f'reason={episode["reason"]} action_mae={np.mean(axis_mae):.3f} '
                  f'axis_mae={axis_mae} phase_mae={phase_mae}',flush=True)
    finally:
        env.close()
    return dict(success_rate=float(np.mean([r['success'] for r in episodes])),
                mean_coverage=float(np.mean([r['coverage'] for r in episodes])),
                episodes=episodes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('train','evaluate'), default='train')
    p.add_argument('--stage2-actor', type=Path, default=Path('runs/stage2_recovery_dagger_v3/best_actor.zip'))
    p.add_argument('--model', type=Path, help='Checkpoint for --mode evaluate')
    p.add_argument('--init-actor', type=Path, help='Compatible 56D Stage-3 actor to initialize V4; fresh V4 dataset')
    p.add_argument('--out', type=Path, default=Path('runs/stage3_dagger_v4'))
    p.add_argument('--resume', action='store_true', help='Resume completed rounds in --out')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--episodes-per-round', type=int, default=4)
    p.add_argument('--episode-steps', type=int, default=30000)
    p.add_argument('--samples-per-episode', type=int, default=4200)
    p.add_argument('--probe-steps', type=int, default=7000, help='Maximum steps in each student-driven probe')
    p.add_argument('--student-burst-steps', type=int, default=60, help='Student actions in a row between brief expert corrections')
    p.add_argument('--max-samples', type=int, default=84000)
    p.add_argument('--epochs-per-round', type=int, default=6)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--bc-lr', type=float, default=3e-5)
    p.add_argument('--beta', type=float, default=.85, help='Requested expert fraction in MIXED rollouts from round 2')
    p.add_argument('--beta-step', type=float, default=.05, help='Linear expert-fraction decrease per later round')
    p.add_argument('--beta-min', type=float, default=.50)
    p.add_argument('--recovery-steps', type=int, default=12, help='Expert correction steps after a high-disagreement student action')
    p.add_argument('--disagreement-threshold', type=float, default=.40, help='Mean absolute action disagreement that triggers correction')
    p.add_argument('--no-paint-guard', type=int, default=140, help='Expert override when fewer steps remain to no-paint stall')
    p.add_argument('--contact-guard', type=int, default=65, help='Expert override when fewer steps remain to idle-contact stall')
    p.add_argument('--recovery-weight', type=float, default=2.0,
                   help='BC importance multiplier for student-visited/recovery samples')
    p.add_argument('--transition-weight', type=float, default=2.5,
                   help='BC importance of rare rotate/lift/lower/phase-switch samples')
    p.add_argument('--transition-window', type=int, default=12,
                   help='Additional steps to preserve after a phase switch')
    p.add_argument('--focus-fraction', type=float, default=.35,
                   help='Fraction of training epoch oversampled from rare motion phases')
    p.add_argument('--axis-weights', nargs=4, type=float, default=[1.6, 1.0, 1.4, 2.5],
                   metavar=('U', 'V', 'NORMAL', 'ROTATE'),
                   help='Per-action BC loss weights')
    p.add_argument('--grad-clip', type=float, default=5.,
                   help='BC gradient norm clipping')
    p.add_argument('--skip-baseline-eval', action='store_true',
                   help='Do not independently evaluate/preserve initial checkpoint before Round 1')
    p.add_argument('--eval-episodes', type=int, default=8)
    p.add_argument('--pass-rate', type=float, default=.75)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--seed', type=int, default=360000)
    p.add_argument('--eval-seed', type=int, default=860000)
    p.add_argument('--walls', nargs='+', choices=WALLS, default=list(WALLS))
    args = p.parse_args()
    counts = (args.workers,args.rounds,args.episodes_per_round,args.episode_steps,
              args.samples_per_episode,args.max_samples,args.epochs_per_round,
              args.batch_size,args.eval_episodes,args.patience,args.recovery_steps,
              args.probe_steps,args.student_burst_steps)
    if min(counts) <= 0:
        p.error('All counts must be positive')
    if args.samples_per_episode > args.max_samples:
        p.error('--samples-per-episode cannot exceed --max-samples')
    if not all(0 <= x <= 1 for x in (args.beta,args.beta_step,args.beta_min,args.pass_rate)):
        p.error('Fractions must be between 0 and 1')
    if args.beta_min > args.beta:
        p.error('--beta-min cannot exceed --beta')
    if not 0 < args.disagreement_threshold <= 2:
        p.error('--disagreement-threshold must be in (0, 2]')
    if args.no_paint_guard < 0 or args.contact_guard < 0 or args.recovery_weight <= 0:
        p.error('Guards must be non-negative and --recovery-weight positive')
    if not (0 <= args.focus_fraction <= 1) or args.transition_window < 0:
        p.error('--focus-fraction must be in [0,1] and --transition-window >= 0')
    if (args.transition_weight <= 0 or args.grad_clip <= 0 or
        not np.all(np.isfinite(args.axis_weights)) or min(args.axis_weights) <= 0):
        p.error('Loss weights and --grad-clip must be finite positive values')
    if args.mode == 'evaluate':
        if not args.model or not args.model.is_file():
            p.error('--mode evaluate requires a valid --model ZIP')
        model = SAC.load(str(args.model), device='cpu')
        if model.observation_space['state'].shape != (STATE_DIM,):
            p.error('Checkpoint is not a Stage-3 V8 goal-conditioned actor')
        args.out.mkdir(parents=True, exist_ok=True)
        result = evaluate(model, args, args.eval_seed)
        (args.out/'independent_evaluation.json').write_text(json.dumps(result, indent=2))
        return
    if args.resume and args.init_actor is not None:
        p.error('--init-actor cannot be combined with --resume')
    if not args.resume and args.init_actor is None and not args.stage2_actor.is_file():
        p.error(f'Stage-2 checkpoint missing: {args.stage2_actor}')
    if not args.resume and args.init_actor is not None and not args.init_actor.is_file():
        p.error(f'Stage-3 initialization checkpoint missing: {args.init_actor}')
    results_path = args.out/'dagger_results.json'
    data_path = args.out/'aggregated_demonstrations.npz'
    best_path = args.out/'best_actor.zip'
    if args.resume:
        if not all(path.is_file() for path in (results_path,data_path,best_path)):
            p.error('--resume requires dagger_results.json, aggregated_demonstrations.npz and best_actor.zip')
        history = json.loads(results_path.read_text())
        data = load_dataset(data_path)
        model = SAC.load(str(best_path), device='cpu')
        first_round = len(history)+1
        best = (float(history[-1]['best_success']), float(history[-1]['best_coverage']))
        stale = int(history[-1].get('stale', 0))
        print(f'[resume] start round {first_round} with {len(data["action"])} samples',flush=True)
    else:
        if any(path.exists() for path in (results_path,data_path,best_path)):
            p.error('Output already has training data. Use --resume or choose a new --out')
        args.out.mkdir(parents=True,exist_ok=True)
        env = Stage3TrainEnv(args.walls, args.episode_steps)
        try:
            model = new_model(env, args.seed)
            if args.init_actor is None:
                warm_start(model, args.stage2_actor)
            else:
                old = SAC.load(str(args.init_actor), device='cpu')
                if old.observation_space['state'].shape != (STATE_DIM,) or old.action_space.shape != (4,):
                    p.error('--init-actor must be a compatible 56D Stage-3 actor')
                model.actor.load_state_dict(old.actor.state_dict(), strict=True)
                print(f'[warm-start] copied Stage-3 actor from {args.init_actor}; '
                      f'critics/replay/optimizer NOT transferred', flush=True)
            obs, _ = env.reset(seed=args.seed, options={'wall':args.walls[0]})
            model.predict(obs, deterministic=True)
        finally:
            env.close()
        init_path = args.out/'warm_started_actor.zip'
        model.save(str(init_path))
        history = []
        data = None
        best = (-1., -1.)
        stale = 0
        first_round = 1
        if not args.skip_baseline_eval:
            # Do not silently replace a stronger initialization with a worse
            # Round-1 student, which V3 could do when best began at (-1,-1).
            baseline = evaluate(model, args, args.eval_seed)
            best = (baseline['success_rate'], baseline['mean_coverage'])
            shutil.copy2(init_path, best_path)
            (args.out/'baseline_evaluation.json').write_text(json.dumps(baseline, indent=2))
            (args.out/'best_evaluation.json').write_text(json.dumps(baseline, indent=2))
            print(f'[baseline] student={best[0]:.1%}/{best[1]:.1%} '
                  f'preserved={best_path}', flush=True)
    th.set_num_threads(1)
    for r in range(first_round, args.rounds+1):
        start_time=time.perf_counter()
        # Start each round from the best saved student, not a failed update.
        snap=args.out/'rollout_actor.zip'
        model.save(str(snap))
        beta = expert_beta(r, args)
        # Each wall gets a full expert/mixed trajectory AND an independent
        # student-visited training probe. The probe's teacher actions are ONLY
        # labels/corrections, never counted as independent evaluation.
        tasks = []
        for i in range(args.episodes_per_round):
            wall = args.walls[i%len(args.walls)]
            tasks.append((snap, wall, args.seed+r*10000+i, beta,
                          args.episode_steps, args.samples_per_episode,
                          'expert' if r == 1 and not args.resume else 'mixed',
                          args.recovery_steps, args.disagreement_threshold,
                          args.no_paint_guard, args.contact_guard,
                          args.student_burst_steps,args.transition_window))
            tasks.append((snap, wall, args.seed+r*10000+5000+i, 0.,
                          min(args.probe_steps,args.episode_steps),
                          args.samples_per_episode, 'probe',
                          args.recovery_steps, args.disagreement_threshold,
                          args.no_paint_guard, args.contact_guard,
                          args.student_burst_steps,args.transition_window))
        entries = collect(tasks, args.workers)
        for _, row in entries:
            print(f'[rollout] round={r} mode={row["mode"]} wall={row["wall"]} coverage={row["coverage"]:.2%} '
                  f'reason={row["reason"]} beta={row["expert_fraction"]:.0%} '
                  f'samples={row["labeled_samples"]} main={row["main_samples"]} '
                  f'cleanup={row["cleanup_samples"]} recovery={row["recovery_samples"]} '
                  f'transition={row["transition_samples"]} rescues={row["rescue_events"]} student={row["student_fraction"]:.0%} '
                   f'action_mae={row["action_mae"]:.3f}',flush=True)
        new={k:np.concatenate([d[k] for d,_ in entries],axis=0) for k in DATA_KEYS}
        data=merge(data,new,args.max_samples,args.seed+r)
        clone(model,data,args,r)
        ckpt=args.out/f'round_{r}_actor.zip'
        model.save(str(ckpt))
        val=evaluate(model,args,args.eval_seed)
        score=(val['success_rate'],val['mean_coverage'])
        better=score>best
        if better:
            best=score
            stale=0
            shutil.copy2(ckpt,best_path)
            (args.out/'best_evaluation.json').write_text(json.dumps(val,indent=2))
        else:
            stale+=1
            model=SAC.load(str(best_path),device='cpu')
        np.savez_compressed(data_path,**data)
        entry=dict(round=r,beta=float(beta),samples=len(data['action']),
                   rollouts=[row for _,row in entries],evaluation=val,
                   improved=better,stale=stale,best_success=float(best[0]),
                   best_coverage=float(best[1]),
                   elapsed_s=time.perf_counter()-start_time)
        history.append(entry)
        results_path.write_text(json.dumps(history,indent=2))
        print(f'[gate] round={r} eval={score[0]:.1%}/{score[1]:.1%} '
              f'best={best[0]:.1%}/{best[1]:.1%} saved={best_path}',flush=True)
        if best[0]>=args.pass_rate:
            print('[gate] Stage-3 DAgger gate met; verify using unseen seeds.',flush=True)
            break
        if stale>=args.patience:
            print('[gate] No improvement for the patience limit; best actor preserved.',flush=True)
            break
    print('[note] V4 is imitation/DAgger; SAC critics were NOT trained. '
          'The deterministic V8 planner remains required at deployment.',flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
