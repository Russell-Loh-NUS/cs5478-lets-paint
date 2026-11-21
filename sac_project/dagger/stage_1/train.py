#!/usr/bin/env python3
"""DAgger for Stage-1 SAC Cartesian painting. Does not modify existing files.

Requires train_sac_demonstrations.py, train_sac_curriculum.py and train_sac_windows.py
beside this script. Expert labels policy-visited states; only expert-labelled
samples train the actor. Evaluation uses deterministic policy without expert help.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import SAC
from train_sac_curriculum import CurriculumEnv, evaluate
from train_sac_demonstrations import PatchSweep

KEYS = ('state', 'global_map', 'local_map', 'action')


def load_data(path):
    with np.load(path) as f:
        data = {k: np.asarray(f[k], dtype=np.float32).copy() for k in KEYS}
    n = len(data['action'])
    if any(len(data[k]) != n for k in KEYS):
        raise ValueError('Dataset observation/action length mismatch')
    return data


def merge_data(old, new, cap, seed):
    merged = {k: np.concatenate([old[k], new[k]], axis=0) for k in KEYS}
    if len(merged['action']) > cap:
        # Retain some initial expert data and sample the rest to bound memory.
        rng = np.random.default_rng(seed)
        n0 = min(len(old['action']), cap // 3)
        keep_old = rng.choice(len(old['action']), size=n0, replace=False)
        pool = np.arange(len(old['action']), len(merged['action']))
        other = rng.choice(pool, size=cap-n0, replace=len(pool) < cap-n0)
        idx = np.concatenate([keep_old, other])
        merged = {k: v[idx] for k, v in merged.items()}
    return merged


def rollout(model, args, round_idx):
    env = CurriculumEnv(0, args.walls, gui=False)
    buffer = {k: [] for k in KEYS}
    results = []
    rng = np.random.default_rng(args.seed + round_idx * 991)
    beta = args.beta_start * (args.beta_decay ** (round_idx - 1))
    beta = max(0.0, min(1.0, beta))
    try:
        for episode in range(args.episodes_per_round):
            wall = args.walls[episode % len(args.walls)]
            seed = args.seed + round_idx * 10000 + episode
            obs, _ = env.reset(seed=seed, options={'wall': wall})
            expert = PatchSweep(env)
            expert_moves = 0
            last_info = None
            for t in range(args.episode_steps):
                # Label the actual learner-visited state BEFORE taking an action.
                label = expert.action(env)
                policy_action, _ = model.predict(obs, deterministic=True)
                if not np.isfinite(policy_action).all():
                    raise RuntimeError('Nonfinite policy action')
                for k in KEYS[:-1]:
                    buffer[k].append(obs[k].copy())
                buffer['action'].append(label.copy())
                use_expert = rng.random() < beta
                if use_expert:
                    expert_moves += 1
                action = label if use_expert else policy_action
                obs, _, terminated, truncated, last_info = env.step(action)
                if terminated or truncated:
                    break
            row = dict(round=round_idx, episode=episode, wall=wall, seed=seed,
                       steps=t+1, coverage=float(last_info['target_coverage']),
                       success=bool(last_info['is_success']), reason=last_info['reason'],
                       expert_action_fraction=expert_moves/(t+1))
            results.append(row)
            print(f'[dagger rollout] round={round_idx} {episode+1}/{args.episodes_per_round} '
                  f'{wall} coverage={row["coverage"]:.1%} reason={row["reason"]} '
                  f'expert_actions={row["expert_action_fraction"]:.0%}', flush=True)
    finally:
        env.close()
    return {k: np.stack(v).astype(np.float32) for k, v in buffer.items()}, results


def clone_actor(model, data, args, round_idx):
    actor = model.actor
    optimizer = th.optim.Adam(actor.parameters(), lr=args.bc_lr)
    rng = np.random.default_rng(args.seed + round_idx)
    n = len(data['action'])
    for epoch in range(1, args.epochs_per_round + 1):
        order = rng.permutation(n)
        running = 0.0
        for start in range(0, n, args.batch_size):
            ix = order[start:start + args.batch_size]
            obs = {k: th.as_tensor(data[k][ix], device=model.device) for k in KEYS[:-1]}
            targets = th.as_tensor(data['action'][ix], device=model.device)
            predictions = actor(obs, deterministic=True)
            loss = th.nn.functional.mse_loss(predictions, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(ix)
        print(f'[dagger bc] round={round_idx} epoch={epoch}/{args.epochs_per_round} '
              f'mse={running/n:.6f}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('runs/sac_demo_v1/bc_initial_sac.zip'))
    p.add_argument('--initial-data', type=Path, default=Path('runs/sac_demo_v1/demonstrations.npz'))
    p.add_argument('--out', type=Path, default=Path('runs/sac_dagger_v1'))
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--episodes-per-round', type=int, default=12)
    p.add_argument('--episode-steps', type=int, default=900)
    p.add_argument('--epochs-per-round', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--bc-lr', type=float, default=5e-5)
    p.add_argument('--beta-start', type=float, default=0.5)
    p.add_argument('--beta-decay', type=float, default=0.5)
    p.add_argument('--max-samples', type=int, default=100000)
    p.add_argument('--eval-episodes', type=int, default=8)
    p.add_argument('--pass-rate', type=float, default=0.75,
                   help='Stage 1 gate: fraction of evaluation episodes successful; default 75%%')
    p.add_argument('--seed', type=int, default=160000)
    p.add_argument('--walls', nargs='+', choices=('south','east','north','west'),
                   default=['south','east','north','west'])
    args = p.parse_args()
    if min(args.rounds, args.episodes_per_round, args.episode_steps,
           args.epochs_per_round, args.batch_size, args.eval_episodes) <= 0:
        p.error('All count parameters must be positive')
    if not 0 <= args.pass_rate <= 1 or not 0 <= args.beta_start <= 1 or not 0 <= args.beta_decay <= 1:
        p.error('Rates must be in [0, 1]')
    if args.max_samples < 100:
        p.error('--max-samples must be >= 100')
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_data(args.initial_data)
    model = SAC.load(str(args.model), device='cpu')
    # Explicitly check that the model can act on the actual Stage-1 observation.
    env = CurriculumEnv(0, args.walls, gui=False)
    try:
        obs, _ = env.reset(seed=args.seed)
        if model.observation_space != env.observation_space or model.action_space != env.action_space:
            raise ValueError('Checkpoint and Stage-1 environment spaces differ')
        model.predict(obs, deterministic=True)
    finally:
        env.close()
    summary = []
    print(f'[dagger] initial samples={len(data["action"])}; Stage-1 success gate={args.pass_rate:.0%}', flush=True)
    for round_idx in range(1, args.rounds + 1):
        started = time.perf_counter()
        added, episodes = rollout(model, args, round_idx)
        data = merge_data(data, added, args.max_samples, args.seed + round_idx)
        clone_actor(model, data, args, round_idx)
        ckpt = args.out / f'round_{round_idx}_actor.zip'
        model.save(str(ckpt))
        np.savez_compressed(args.out / 'aggregated_demonstrations.npz', **data)
        # Evaluate on seeds disjoint from collection; NEVER mix expert actions here.
        ev = evaluate(model, 0, args.walls, args.eval_episodes,
                      args.seed + 500000 + round_idx * 1000, gui=False)
        record = dict(round=round_idx, samples=len(data['action']),
                      new_samples=len(added['action']), evaluation=ev,
                      rollout_episodes=episodes, elapsed_s=time.perf_counter()-started)
        summary.append(record)
        (args.out / 'dagger_results.json').write_text(json.dumps(summary, indent=2))
        print(f'[dagger gate] round={round_idx} success={ev["success_rate"]:.0%} '
              f'coverage={ev["mean_coverage"]:.1%} required={args.pass_rate:.0%} '
              f'checkpoint={ckpt}', flush=True)
        if ev['success_rate'] >= args.pass_rate:
            print('[dagger] Stage 1 gate PASSED. No SAC fine-tuning performed.', flush=True)
            break
    else:
        print('[dagger] Stage 1 gate not reached; inspect dagger_results.json.', flush=True)
    print('[dagger] Actor cloned only; SAC critics remain untrained. '
          'Do not assume this checkpoint is ready for SAC fine-tuning.', flush=True)


if __name__ == '__main__':
    main()
