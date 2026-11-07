#!/usr/bin/env python3
"""SAC baseline for known flat walls with randomized window openings.

Uses the project's real WholeWallEnv, wall-relative Cartesian actions, IK,
contact physics and three-channel coverage/paintable/target maps.
Scope: single-wall painting, unlimited paint; refill is NOT trained here.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn
import train_td3_wall as core


class MapEncoder(BaseFeaturesExtractor):
    """CNN map encoder + MLP state encoder; avoids flattening spatial pixels."""
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.global_net = self._cnn(core.WholeWallEnv.GLOBAL)
        self.local_net = self._cnn(core.WholeWallEnv.LOCAL)
        self.state_net = nn.Sequential(nn.Linear(core.WholeWallEnv.STATE, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        with th.no_grad():
            g = self.global_net(th.zeros(1, 3, core.WholeWallEnv.GLOBAL, core.WholeWallEnv.GLOBAL)).shape[1]
            l = self.local_net(th.zeros(1, 3, core.WholeWallEnv.LOCAL, core.WholeWallEnv.LOCAL)).shape[1]
        self.head = nn.Sequential(nn.Linear(g+l+64, features_dim), nn.ReLU())

    @staticmethod
    def _cnn(size):
        return nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                             nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten())

    def forward(self, obs):
        g = obs['global_map'].reshape(-1, 3, core.WholeWallEnv.GLOBAL, core.WholeWallEnv.GLOBAL)
        l = obs['local_map'].reshape(-1, 3, core.WholeWallEnv.LOCAL, core.WholeWallEnv.LOCAL)
        return self.head(th.cat([self.global_net(g), self.local_net(l), self.state_net(obs['state'])], dim=1))



class StepProgress(BaseCallback):
    """Print periodic progress even when an episode has not completed."""
    def __init__(self, every=100):
        super().__init__()
        self.every = every
        self.started = None

    def _on_training_start(self):
        self.started = time.perf_counter()
        print("[SAC] Training started", flush=True)

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            elapsed = max(time.perf_counter() - self.started, 1e-9)
            print(f"[SAC] steps={self.num_timesteps}/{self.locals.get('total_timesteps', self.model._total_timesteps)} "
                  f"elapsed={elapsed:.1f}s avg_fps={self.num_timesteps/elapsed:.2f}", flush=True)
        return True


def environment(layout, walls, max_steps, randomize=True, gui=False):
    cfg = core.Config(walls=tuple(walls), layout=layout, randomize_size=randomize,
                      max_episode_steps=max_steps, success_fraction=.98)
    return core.WholeWallEnv(cfg, gui=gui)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('train', 'evaluate'), default='train')
    ap.add_argument('--gui', action='store_true', help='Show PyBullet window during evaluation only')
    ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--log-every', type=int, default=100, help='Print progress every N environment steps')
    ap.add_argument('--train-freq', type=int, default=4, help='Environment steps per gradient update')
    ap.add_argument('--gradient-steps', type=int, default=1)
    ap.add_argument('--episode-steps', type=int, default=5000)
    ap.add_argument('--out', type=Path, default=Path('runs/sac_windows_v1'))
    ap.add_argument('--model', type=Path)
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42000)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--layout', choices=('solid', 'random', 'original'), default='random')
    ap.add_argument('--walls', nargs='+', choices=core.WALLS, default=list(core.WALLS))
    args = ap.parse_args()
    if min(args.steps, args.episode_steps, args.episodes, args.log_every, args.train_freq, args.gradient_steps) <= 0:
        ap.error('steps, episode-steps and episodes must be positive')
    if args.gui and args.mode != 'evaluate':
        ap.error('--gui is supported only in evaluate mode')
    args.out.mkdir(parents=True, exist_ok=True)
    env = environment(args.layout, args.walls, args.episode_steps, gui=args.gui)
    try:
        if args.mode == 'train':
            model = SAC('MultiInputPolicy', env, policy_kwargs=dict(
                features_extractor_class=MapEncoder,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=[256, 256]), learning_rate=3e-4, buffer_size=100000,
                learning_starts=1000, batch_size=128, train_freq=args.train_freq,
                gradient_steps=args.gradient_steps, seed=args.seed, device=args.device, verbose=1,
                tensorboard_log=None)
            (args.out/'config.json').write_text(json.dumps({'config':asdict(env.config), 'seed':args.seed,
                'scope':'single-wall, no paint-can refill', 'training_layout':args.layout}, indent=2))
            callbacks = CallbackList([
                StepProgress(args.log_every),
                CheckpointCallback(save_freq=10000, save_path=str(args.out/'checkpoints'), name_prefix='sac')
            ])
            try:
                model.learn(total_timesteps=args.steps, callback=callbacks,
                            log_interval=1, progress_bar=True)
            except KeyboardInterrupt:
                model.save(str(args.out/'interrupted'))
                print('Interrupted: saved', args.out/'interrupted.zip', flush=True)
                return
            model.save(str(args.out/'final'))
            print('Saved', args.out/'final.zip')
        else:
            if args.model is None:
                ap.error('--model required for evaluation')
            model = SAC.load(str(args.model), device=args.device)
            rows=[]
            for i in range(args.episodes):
                wall=args.walls[i % len(args.walls)]
                obs,info=env.reset(seed=args.seed+i, options={'wall':wall})
                terminated=truncated=False
                for step in range(args.episode_steps):
                    action,_=model.predict(obs, deterministic=True)
                    obs,reward,terminated,truncated,info=env.step(action)
                    if terminated or truncated: break
                row={'episode':i, 'wall':wall, 'seed':args.seed+i, 'steps':step+1,
                     'reason':info.get('reason'), 'success':bool(info.get('is_success', False)),
                     'target_coverage':float(info.get('target_coverage', 0)),
                     'wall_coverage':float(info.get('wall_coverage', 0))}
                rows.append(row)
                print(json.dumps(row), flush=True)
            path=args.out/'evaluation.json'
            path.write_text(json.dumps({'model':str(args.model),'layout':args.layout,'episodes':rows},indent=2))
            print('Saved',path)
    finally:
        env.close()

if __name__=='__main__':
    main()
