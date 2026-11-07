#!/usr/bin/env python3
"""Learn local dipping with expert BC followed by DAgger, using the V1 physics.

Place beside dip_v2_environment.py in sac_project/. Saves SAC-compatible ZIPs,
but ONLY its deterministic actor is trained by imitation: no SAC/RL updates.
Teacher actions execute during labeled training collection only. Validation
and final test use 100% student actions and the unchanged V1 success criteria.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from dip_v2_environment import make_env, evaluate, PHASES, OBS_DIM


def save_json(path, value):
    Path(path).write_text(json.dumps(value, default=str, indent=2))


def sampling_tables(data):
    """Uniform phase sampling, weighted toward corrections and settling.

    origin: 0=reset/teacher, 1=student action, 2=perturbed teacher action.
    Origin describes how the CURRENT observation was reached, not its label.
    """
    pools = [np.flatnonzero(data['phase'] == k) for k in range(4)]
    present = [k for k, pool in enumerate(pools) if len(pool)]
    if not present:
        raise ValueError('No training rows')
    probabilities = {}
    for k in present:
        pool = pools[k]
        near = np.max(np.abs(data['action'][pool]), axis=1) < .9
        weights = 1. + 2.*(data['origin'][pool] == 1) + 2.*near
        probabilities[k] = weights/weights.sum()
    return pools, present, probabilities


def balanced_indices(data, batch_size, rng, tables=None):
    pools, present, probabilities = sampling_tables(data) if tables is None else tables
    phases = rng.choice(present, size=batch_size)
    indices = np.empty(batch_size, dtype=np.int64)
    for k in present:
        mask = phases == k
        indices[mask] = rng.choice(pools[k], size=int(mask.sum()), p=probabilities[k])
    return indices


def collect(env, model, episodes, seed, beta, noise=0.):
    """Labels are queried before stepping; never label with the executed mix."""
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ('obs', 'action', 'phase', 'origin')}
    reports = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed+episode)
        origin = 0
        teacher_actions = student_actions = 0
        while True:
            label = np.asarray(env.teacher(), np.float32)
            if label.shape != (4,) or not np.isfinite(label).all():
                raise ValueError('Invalid teacher action')
            rows['obs'].append(obs.copy())
            rows['action'].append(label.copy())
            rows['phase'].append(env.phase)
            rows['origin'].append(origin)
            use_teacher = model is None or rng.random() < beta
            if use_teacher:
                action = np.clip(label+rng.normal(0., noise, 4), -1., 1.)
                origin = 2 if noise else 0
                teacher_actions += 1
            else:
                action = model.predict(obs, deterministic=True)[0]
                origin = 1
                student_actions += 1
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        report = dict(seed=seed+episode, collection='training_only', beta=beta,
                      teacher_actions=teacher_actions, student_actions=student_actions, **info)
        reports.append(report)
        print('[collect] '+json.dumps(report), flush=True)
    data = {k: np.asarray(v, dtype=np.float32 if k in ('obs', 'action') else np.uint8)
            for k, v in rows.items()}
    return data, reports


def merge(old, new):
    if old is None:
        return new
    return {k: np.concatenate((old[k], new[k]), axis=0) for k in old}


def fit(model, data, updates, batch_size, seed):
    import torch
    rng = np.random.default_rng(seed)
    actor = model.actor
    actor.set_training_mode(True)
    # Fresh optimizer each round is intentional; no unsupported resume claim.
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    losses = []
    # Fixed weights for the whole fit pass; sampling remains phase-balanced.
    tables = sampling_tables(data)
    for update in range(updates):
        indices = balanced_indices(data, batch_size, rng, tables)
        x = torch.as_tensor(data['obs'][indices], device=model.device)
        y = torch.as_tensor(data['action'][indices], device=model.device)
        prediction = actor(x, deterministic=True)
        loss = torch.mean((prediction-y)**2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (update+1) % 250 == 0 or update == updates-1:
            print(f'[BC] updates={update+1}/{updates} mse={np.mean(losses[-250:]):.6f}', flush=True)
    actor.set_training_mode(False)
    return dict(mean_loss=float(np.mean(losses)), last_loss=float(np.mean(losses[-250:])),
                samples=len(data['obs']),
                phase_samples={PHASES[k]: int((data['phase'] == k).sum()) for k in range(4)},
                student_origin_samples=int((data['origin'] == 1).sum()))


def score(report):
    """Compare only independent student rollouts, never collection success."""
    episodes = report['episodes']
    return (report['success_rate'], -report['collision_rate'],
            float(np.mean([PHASES.index(r['phase']) for r in episodes])),
            -float(np.mean([r['target_error_m'] for r in episodes])),
            -float(np.mean([r['steps'] for r in episodes])))


def passes(report):
    return report['controller'] == 'student' and report['success_rate'] >= .95 and report['collision_rate'] == 0.


def self_test():
    from dip_v2_environment import self_test as geometry_test
    geometry_test()
    data = dict(obs=np.zeros((103, OBS_DIM), np.float32),
                action=np.zeros((103, 4), np.float32),
                phase=np.array([0]*100+[1, 2, 3], np.uint8),
                origin=np.zeros(103, np.uint8))
    ids = balanced_indices(data, 4000, np.random.default_rng(7))
    counts = np.bincount(data['phase'][ids], minlength=4)
    assert all(800 < c < 1200 for c in counts), counts
    assert not passes(dict(controller='expert', success_rate=1., collision_rate=0.))
    assert not passes(dict(controller='student', success_rate=0., collision_rate=0.))
    assert not passes(dict(controller='student', success_rate=1., collision_rate=.01))
    assert passes(dict(controller='student', success_rate=1., collision_rate=0.))
    print('Phase sampling and independent-student gate tests passed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('train', 'evaluate'), default='train')
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--out', type=Path, default=Path('sac_project/runs/stage4_dip_v2'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--demos', type=int, default=32)
    p.add_argument('--bc-updates', type=int, default=4000)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--episodes-per-round', type=int, default=16)
    p.add_argument('--updates-per-round', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--eval-episodes', type=int, default=12)
    p.add_argument('--test-episodes', type=int, default=40)
    p.add_argument('--episode-steps', type=int, default=800)
    p.add_argument('--seed', type=int, default=5478)
    p.add_argument('--eval-seed', type=int, default=970000)
    p.add_argument('--test-seed', type=int, default=980000)
    p.add_argument('--spread', type=float, default=.08)
    p.add_argument('--gui', action='store_true')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if min(args.demos, args.bc_updates, args.episodes_per_round, args.updates_per_round,
           args.batch_size, args.eval_episodes, args.test_episodes, args.episode_steps) < 1:
        p.error('Counts must be positive')
    if args.rounds < 0 or not 0 <= args.spread <= .15:
        p.error('rounds must be nonnegative; spread must be within [0, .15]')
    if args.mode == 'evaluate' and (args.checkpoint is None or not args.checkpoint.is_file()):
        p.error('Evaluation requires --checkpoint pointing to a saved ZIP')
    if args.mode == 'train' and args.checkpoint is not None:
        p.error('V2 trains a fresh actor; keep the V1 checkpoint as a baseline')
    # Prevent accidental train/validation/test overlap, including custom seeds.
    training_seeds = set(range(args.seed, args.seed+args.demos))
    for round_id in range(1, args.rounds+1):
        training_seeds.update(range(args.seed+10000*round_id,
                                    args.seed+10000*round_id+args.episodes_per_round))
    validation_seeds = set(range(args.eval_seed, args.eval_seed+args.eval_episodes))
    test_seeds = set(range(args.test_seed, args.test_seed+args.test_episodes))
    if training_seeds & validation_seeds or training_seeds & test_seeds or validation_seeds & test_seeds:
        p.error('Training, validation and final test seeds must not overlap')
    for name in ('simulation/robot.py', 'simulation/paint_can.py'):
        if not (args.project_root/name).is_file():
            p.error(f'Missing {name}; run from project root or pass --project-root')
    import torch
    from stable_baselines3 import SAC
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out/'config.json', vars(args))
    env = make_env(args.project_root, args.gui, args.episode_steps, args.spread)
    try:
        if args.mode == 'evaluate':
            model = SAC.load(args.checkpoint, device='cpu')
            if model.observation_space != env.observation_space or model.action_space != env.action_space:
                raise ValueError('Checkpoint spaces do not match V1 dip environment')
            test = evaluate(env, model, args.test_episodes, args.test_seed)
            save_json(args.out/'evaluation.json', dict(test, passed=passes(test)))
            return
        model = SAC('MlpPolicy', env, buffer_size=1, learning_starts=0,
                    policy_kwargs=dict(net_arch=[128, 128]), device='cpu', seed=args.seed)
        data, demos = collect(env, None, args.demos, args.seed, beta=1.)
        save_json(args.out/'expert_demonstrations.json', demos)
        np.savez_compressed(args.out/'dataset.npz', **data)
        if not all(r['is_success'] and not r['collision'] for r in demos):
            raise RuntimeError('Expert demonstrations failed; inspect report before fitting')
        if set(np.unique(data['phase'])) != {0, 1, 2, 3}:
            raise RuntimeError('Demonstrations must cover every phase')
        best_score = None
        best_round = None
        history = []
        for round_id in range(args.rounds+1):
            if round_id:
                # Mixed collection is training data, never a performance gate.
                # Last round always visits states under a pure student rollout.
                beta = 0. if round_id == args.rounds else .35*(.5**(round_id-1))
                new, collection = collect(env, model, args.episodes_per_round,
                                          args.seed+10000*round_id, beta, noise=.02 if beta else 0.)
                data = merge(data, new)
                save_json(args.out/f'collection_{round_id:02d}.json', collection)
            training = fit(model, data, args.bc_updates if round_id == 0 else args.updates_per_round,
                           args.batch_size, args.seed+round_id)
            model.save(args.out/f'round_{round_id:02d}_actor')
            validation = evaluate(env, model, args.eval_episodes, args.eval_seed)
            save_json(args.out/f'validation_{round_id:02d}.json', validation)
            candidate_score = score(validation)
            if best_score is None or candidate_score > best_score:
                best_score, best_round = candidate_score, round_id
                model.save(args.out/'best_actor')
            history.append(dict(round=round_id, training=training,
                                student_success_rate=validation['success_rate'],
                                student_collision_rate=validation['collision_rate'], best_round=best_round))
            save_json(args.out/'history.json', history)
            np.savez_compressed(args.out/'dataset.npz', **data)
            print('[student-validation] '+json.dumps(history[-1]), flush=True)
        model.save(args.out/'last_actor')
        best = SAC.load(args.out/'best_actor.zip', device='cpu')
        test = evaluate(env, best, args.test_episodes, args.test_seed)
        result = dict(test, passed=passes(test), selected_round=best_round,
                      selection='student validation; final test not used for model selection',
                      training_method='behavior cloning + DAgger; no RL updates')
        save_json(args.out/'evaluation.json', result)
        print('[FINAL] '+json.dumps({k: v for k, v in result.items() if k != 'episodes'}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
