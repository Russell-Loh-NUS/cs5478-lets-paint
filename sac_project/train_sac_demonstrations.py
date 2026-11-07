#!/usr/bin/env python3
"""Experimental SAC-compatible expert collection and actor pretraining.

Run from sac_project. No existing files are overwritten. Start with --mode collect.
This is a NEW heuristic expert for the 4-action SAC environment, not the old
imitation expert. It must be validated before using its demonstrations.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch as th
from stable_baselines3 import SAC
from train_sac_curriculum import CurriculumEnv, evaluate
from train_sac_windows import MapEncoder


def make_model(env, seed):
    return SAC('MultiInputPolicy', env, policy_kwargs=dict(
        features_extractor_class=MapEncoder,
        features_extractor_kwargs=dict(features_dim=256), net_arch=[256, 256]),
        learning_rate=3e-4, buffer_size=100000, learning_starts=1000,
        batch_size=128, train_freq=(4, 'step'), gradient_steps=1,
        seed=seed, device='cpu', verbose=0)


class PatchSweep:
    """Wall-frame, feedback-based lane sweep. Works on stage 1 SOLID patch only."""
    def __init__(self, env):
        rows, cols = np.nonzero(env.target)
        ppm = env.wall.canvas.ppm
        us = env.wall.length - (cols + .5) / ppm
        vs = env.wall.height - (rows + .5) / ppm
        self.u0, self.u1 = float(us.min()), float(us.max())
        self.v0, self.v1 = float(vs.min()), float(vs.max())
        self.lanes = np.linspace(self.u0 + .045, self.u1 - .045,
                                 max(2, int(np.ceil((self.u1-self.u0)/.12))))
        self.index = 0
        self.phase = 'align'
        self.stall = 0

    def action(self, env):
        pos = env._pose()[0]
        u, v, n = pos
        lane = float(self.lanes[min(self.index, len(self.lanes)-1)])
        target_v = self.v1 if self.index % 2 == 0 else self.v0
        if self.phase == 'align':
            if abs(u-lane) < .009:
                self.phase = 'stroke'
            target = np.array([lane, v])
        else:
            if abs(v-target_v) < .012:
                self.index += 1
                if self.index >= len(self.lanes):
                    self.index = 0  # second pass if gaps remain
                self.phase = 'align'
            target = np.array([lane, target_v])
        # The original environment maps actions to .10 m/s in u,v and .025 m/s in n.
        error = target - np.asarray([u,v])
        planar = np.clip(error / (.10 * .05), -1., 1.)
        if self.phase == 'align':
            planar[1] = 0.
        else:
            planar[0] = 0.
        normal = float(np.clip((.0595 - n)/(.025*.05), -1., 1.))
        theta = float(np.clip(-env.command[3]/(.70*.05), -1., 1.))
        return np.asarray([planar[0], planar[1], normal, theta], dtype=np.float32)


def collect(args):
    env = CurriculumEnv(0, args.walls, gui=False)
    dataset = {k: [] for k in ('state','global_map','local_map','action')}
    episodes = []
    try:
        for i in range(args.episodes):
            wall = args.walls[i % len(args.walls)]
            obs, _ = env.reset(seed=args.seed+i, options={'wall':wall})
            expert = PatchSweep(env)
            samples = []
            for step in range(args.episode_steps):
                action = expert.action(env)
                samples.append((obs, action))
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            coverage = float(info['target_coverage'])
            accepted = coverage >= args.min_coverage and info['reason'] != 'collision'
            episodes.append(dict(wall=wall, seed=args.seed+i, steps=step+1,
                                 coverage=coverage, reason=info['reason'], accepted=accepted))
            print(f'[demo] {i+1}/{args.episodes} {wall} coverage={coverage:.1%} '
                  f'reason={info["reason"]} accepted={accepted}', flush=True)
            if accepted:
                for ob, ac in samples:
                    for k in ('state','global_map','local_map'):
                        dataset[k].append(ob[k].copy())
                    dataset['action'].append(ac.copy())
    finally:
        env.close()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'demonstration_report.json').write_text(json.dumps(episodes, indent=2))
    if not dataset['action']:
        raise RuntimeError('No qualifying expert episodes. Inspect demonstration_report.json; '
                           'do not pretrain on failed trajectories. Improve expert first.')
    arrays = {k: np.stack(v).astype(np.float32) for k,v in dataset.items()}
    np.savez_compressed(args.out/'demonstrations.npz', **arrays)
    print(f'[demo] saved {len(arrays["action"])} compatible samples', flush=True)


def pretrain(args):
    path = args.out/'demonstrations.npz'
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}; run --mode collect first')
    data = np.load(path)
    env = CurriculumEnv(0, args.walls)
    try:
        model = make_model(env, args.seed)
        actor = model.actor
        optimizer = th.optim.Adam(actor.parameters(), lr=args.bc_lr)
        n = len(data['action'])
        if n < 100:
            raise RuntimeError('Too few demonstrations for behaviour cloning')
        rng = np.random.default_rng(args.seed)
        for epoch in range(1, args.epochs+1):
            indices = rng.permutation(n)
            total = 0.
            for start in range(0, n, args.batch_size):
                idx = indices[start:start+args.batch_size]
                obs = {k: th.from_numpy(data[k][idx]).to(model.device)
                       for k in ('state','global_map','local_map')}
                target = th.from_numpy(data['action'][idx]).to(model.device)
                predicted = actor(obs, deterministic=True)
                loss = th.nn.functional.mse_loss(predicted, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.item())*len(idx)
            print(f'[bc] epoch={epoch}/{args.epochs} mse={total/n:.6f}', flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        model.save(str(args.out/'bc_initial_sac'))
        print('[bc] saved', args.out/'bc_initial_sac.zip')
        print('[bc] Critics are NOT pretrained; evaluate the actor before SAC fine-tuning.')
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('collect','pretrain','evaluate'), required=True)
    p.add_argument('--out', type=Path, default=Path('runs/sac_demo_v1'))
    p.add_argument('--walls', nargs='+', default=['south','east','north','west'],
                   choices=['south','east','north','west'])
    p.add_argument('--episodes', type=int, default=12)
    p.add_argument('--episode-steps', type=int, default=900)
    p.add_argument('--min-coverage', type=float, default=.75)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--bc-lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=142000)
    p.add_argument('--model', type=Path)
    args = p.parse_args()
    if args.mode == 'collect':
        collect(args)
    elif args.mode == 'pretrain':
        pretrain(args)
    else:
        if not args.model:
            p.error('--model required for evaluate')
        model = SAC.load(str(args.model), device='cpu')
        result = evaluate(model, 0, args.walls, args.episodes, args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out/'bc_evaluation.json').write_text(json.dumps(result, indent=2))
        print(f'[eval] success={result["success_rate"]:.1%} coverage={result["mean_coverage"]:.1%}')

if __name__ == '__main__':
    main()
