#!/usr/bin/env python3
"""Gated SAC curriculum for the existing flat-wall PyBullet environment.

Run from sac_project. Stages: local 0.45m patch, 1.0m patch,
full solid wall, randomized window wall. The same SAC actor and replay
buffer continue across stages within a run. No physical refilling yet.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from train_sac_windows import MapEncoder
import train_td3_wall as core

STAGES = (
    ('small_patch', 'solid', (0.45, 0.45), 900),
    ('large_patch', 'solid', (1.0, 0.85), 2500),
    ('solid_wall', 'solid', None, 5000),
    ('window_wall', 'random', None, 5000),
)

class CurriculumEnv(core.WholeWallEnv):
    def __init__(self, stage, walls, gui=False):
        self.stage_index = stage
        _, layout, self.patch_size, episode_steps = STAGES[stage]
        cfg = core.Config(walls=tuple(walls), layout=layout, randomize_size=True,
                          max_episode_steps=episode_steps, success_fraction=0.90 if stage < 2 else 0.98)
        super().__init__(cfg, gui=gui)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.patch_size is not None:
            # Target only the physically reachable, paintable neighborhood of
            # the roller's settled starting position. Outside remains unpainted
            # for task reward, but physical painting is never faked or disabled.
            u, v = self._pose()[0][:2]
            width, height = self.patch_size
            rows, cols = np.indices(self.target.shape)
            pixel_u = self.wall.length - (cols + 0.5) / core.PPM
            pixel_v = self.wall.height - (rows + 0.5) / core.PPM
            region = ((abs(pixel_u-u) <= width/2) & (abs(pixel_v-v) <= height/2))
            target = self.target & region
            if target.sum() < 100:
                raise RuntimeError('Curriculum target too small; check reset geometry')
            self.target = target
            self.target_pixels = int(target.sum())
            self.target_area = self.target_pixels / core.PPM**2
            self._update_goal()
            obs, info = self._obs(), self._info()
        if self.stage_index == 2:
            # Stage 3 only. Recompute from exact IK/joint limits and sampled
            # roller-axis sweep footprints, not from an arbitrary floor margin.
            original_pixels = int(self.target.sum())
            candidate, stats = core.stage3_sampled_ik_mask(self)
            self.target = candidate
            self.target_pixels = int(candidate.sum())
            self.target_area = self.target_pixels / core.PPM**2
            self.stage3_mask_diagnostics = dict(
                original_target_pixels=original_pixels,
                candidate_target_pixels=self.target_pixels,
                excluded_pixels=original_pixels-self.target_pixels,
                retained_fraction=self.target_pixels/max(1, original_pixels),
                ik_sampling=stats,
                note='IK candidate mask only; static contact and complete strokes not verified')
            self._update_goal()
            obs, info = self._obs(), self._info()
            info['stage3_mask_diagnostics'] = self.stage3_mask_diagnostics
        info['curriculum_stage'] = STAGES[self.stage_index][0]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info['curriculum_stage'] = STAGES[self.stage_index][0]
        return obs, reward, terminated, truncated, info

class Progress(BaseCallback):
    def __init__(self, every):
        super().__init__()
        self.every = every
        self.started = 0.0
        self.episodes = 0
        self.last = None
        self.episode_coverage = []
        self.episode_success = []
        self.actions_sum = 0.0
        self.actions_n = 0

    def _on_training_start(self):
        self.started = time.perf_counter()

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episodes += 1
                self.last = info
                self.episode_coverage.append(float(info.get('target_coverage', 0)))
                self.episode_success.append(float(info.get('is_success', False)))
                for key in ('target_coverage', 'wall_coverage', 'tracking_error_mm', 'rejected_fraction', 'paint_contact_fraction'):
                    if key in info:
                        self.logger.record('painting/episode_' + key, float(info[key]))
                self.logger.record('painting/episode_success', float(info.get('is_success', False)))
                self.logger.record('painting/episode_steps', float(info.get('steps', 0)))
                print(f"[episode] #{self.episodes} reason={info.get('reason')} "
                      f"coverage={100*info.get('target_coverage', 0):.2f}% "
                      f"return={info['episode']['r']:.2f}", flush=True)
        actions = self.locals.get('actions')
        if actions is not None:
            self.actions_sum += float(np.mean(np.abs(actions)))
            self.actions_n += 1
        if self.num_timesteps % self.every == 0:
            infos = self.locals.get('infos', [])
            if infos:
                info = infos[0]
                for key in ('target_coverage', 'wall_coverage', 'tracking_error_mm', 'rejected_fraction', 'paint_contact_fraction'):
                    if key in info:
                        self.logger.record('painting/current_' + key, float(info[key]))
            if self.actions_n:
                self.logger.record('control/mean_abs_action', self.actions_sum/self.actions_n)
                self.actions_sum = 0.0
                self.actions_n = 0
            if self.episode_coverage:
                self.logger.record('painting/mean_episode_coverage_recent', float(np.mean(self.episode_coverage[-20:])))
                self.logger.record('painting/success_rate_recent', float(np.mean(self.episode_success[-20:])))
            self.logger.dump(self.num_timesteps)
            elapsed = time.perf_counter()-self.started
            print(f'[train] steps={self.num_timesteps} elapsed={elapsed:.1f}s '
                  f'fps={self.num_timesteps/max(elapsed, 1e-9):.1f}', flush=True)
        return True

def evaluate(model, stage, walls, episodes, seed, gui=False):
    env = CurriculumEnv(stage, walls, gui=gui)
    results = []
    try:
        for i in range(episodes):
            wall = walls[i % len(walls)]
            obs, _ = env.reset(seed=seed+i, options={'wall': wall})
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            row = dict(wall=wall, seed=seed+i, success=bool(info['is_success']),
                       coverage=float(info['target_coverage']),
                       whole_wall_coverage=float(info['wall_coverage']),
                       steps=int(info['steps']), reason=info['reason'],
                       rejected_fraction=float(info.get('rejected_fraction', 0)))
            results.append(row)
            print(f"[eval] {wall} success={row['success']} "
                  f"target={row['coverage']:.1%} reason={row['reason']}", flush=True)
    finally:
        env.close()
    return dict(stage=STAGES[stage][0], episodes=results,
                success_rate=sum(x['success'] for x in results)/len(results),
                mean_coverage=float(np.mean([x['coverage'] for x in results])))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['train', 'evaluate'], default='train')
    p.add_argument('--stage', type=int, choices=range(1, 5), default=1, help='1..4')
    p.add_argument('--through-stage', type=int, choices=range(1, 5), default=4)
    p.add_argument('--steps-per-stage', type=int, default=50000)
    p.add_argument('--eval-episodes', type=int, default=8)
    p.add_argument('--pass-rate', type=float, default=0.75)
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--out', type=Path, default=Path('runs/sac_curriculum_v1'))
    p.add_argument('--model', type=Path, help='Resume stage from saved SAC checkpoint')
    p.add_argument('--replay-buffer', type=Path, help='Optional matching replay buffer')
    p.add_argument('--walls', nargs='+', choices=core.WALLS, default=list(core.WALLS))
    p.add_argument('--seed', type=int, default=42000)
    p.add_argument('--device', default='cpu')
    p.add_argument('--tensorboard-log', type=Path, help='TensorBoard event directory; default OUT/tensorboard')
    p.add_argument('--gui', action='store_true', help='GUI evaluation only')
    args = p.parse_args()
    if args.steps_per_stage <= 0 or args.eval_episodes <= 0 or args.log_every <= 0:
        p.error('Step and episode counts must be positive')
    if not 0 <= args.pass_rate <= 1:
        p.error('--pass-rate must be in [0,1]')
    if args.stage > args.through_stage and args.mode == 'train':
        p.error('--through-stage must be >= --stage')
    if args.gui and args.mode == 'train':
        p.error('--gui is only for evaluation')
    args.out.mkdir(parents=True, exist_ok=True)
    tb_dir = args.tensorboard_log or (args.out / 'tensorboard')
    stage = args.stage-1
    if args.mode == 'evaluate':
        if not args.model:
            p.error('--model required for evaluation')
        model = SAC.load(str(args.model), device=args.device)
        result = evaluate(model, stage, args.walls, args.eval_episodes, args.seed, args.gui)
        (args.out/'evaluation.json').write_text(json.dumps(result, indent=2))
        print('Saved', args.out/'evaluation.json')
        return
    model = None
    for stage in range(args.stage-1, args.through_stage):
        name = STAGES[stage][0]
        env = CurriculumEnv(stage, args.walls)
        try:
            if model is None:
                if args.model:
                    model = SAC.load(str(args.model), env=env, device=args.device)
                    model.tensorboard_log = str(tb_dir)
                    if args.replay_buffer:
                        model.load_replay_buffer(str(args.replay_buffer))
                else:
                    model = SAC('MultiInputPolicy', env,
                                policy_kwargs=dict(features_extractor_class=MapEncoder,
                                                   features_extractor_kwargs=dict(features_dim=256),
                                                   net_arch=[256, 256]),
                                learning_rate=3e-4, buffer_size=100000,
                                learning_starts=1000, batch_size=128,
                                train_freq=(4, 'step'), gradient_steps=1,
                                seed=args.seed, device=args.device, verbose=1,
                                tensorboard_log=str(tb_dir))
            else:
                model.set_env(env)
            print(f'\n=== Stage {stage+1}: {name} ({args.steps_per_stage} steps) ===', flush=True)
            started = time.perf_counter()
            try:
                model.learn(total_timesteps=args.steps_per_stage, reset_num_timesteps=False,
                            callback=CallbackList([Progress(args.log_every), CheckpointCallback(save_freq=10000, save_path=str(args.out / 'checkpoints'), name_prefix=name)]),
                            progress_bar=False, log_interval=1, tb_log_name=name)
            except KeyboardInterrupt:
                model.save(str(args.out/'interrupted'))
                model.save_replay_buffer(str(args.out/'interrupted_replay.pkl'))
                print('Interrupted checkpoint saved', flush=True)
                return
            model.save(str(args.out/f'{name}_final'))
            model.save_replay_buffer(str(args.out/f'{name}_replay.pkl'))
            print(f'[train] stage wall time={time.perf_counter()-started:.1f}s', flush=True)
        finally:
            env.close()
        result = evaluate(model, stage, args.walls, args.eval_episodes, args.seed+100000)
        (args.out/f'{name}_evaluation.json').write_text(json.dumps(result, indent=2))
        print(f"[gate] {name}: success={result['success_rate']:.0%} "
              f"coverage={result['mean_coverage']:.1%} required={args.pass_rate:.0%}", flush=True)
        if result['success_rate'] < args.pass_rate:
            print(f'[gate] STOP: repeat stage {stage+1} with --model '
                  f'{args.out / (name + "_final.zip")} --replay-buffer '
                  f'{args.out / (name + "_replay.pkl")}', flush=True)
            break
        print('[gate] PASS: moving to next stage', flush=True)

if __name__ == '__main__':
    main()
