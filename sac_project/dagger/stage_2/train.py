#!/usr/bin/env python3
"""Goal-observed Stage-2 DAgger. New observation space; old checkpoints/data are incompatible.

The deterministic coverage planner runs during both expert and autonomous evaluation.
DAgger learns goal-conditioned low-level Cartesian movement, not autonomous route choice.
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
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from train_sac_curriculum import CurriculumEnv
from stage2_expert_recovery import Stage2PatchSweep
from train_sac_dagger import clone_actor, merge_data
import train_td3_wall as core

STAGE = 1
KEYS = ('state', 'global_map', 'local_map', 'action')
PHASES = ('align', 'stroke', 'recover_align', 'recover_stroke')


class GoalEnv(CurriculumEnv):
    """Adds the planner's actual goal/phase to every observation.

    A single planner decision is cached per control step so logging, expert
    labels, and policy observations all refer to the same decision.
    """
    def __init__(self, walls, episode_steps=2500):
        super().__init__(STAGE, walls, gui=False)
        self.config.max_episode_steps = episode_steps
        self.expert = None
        old = self.observation_space['state']
        self.observation_space.spaces['state'] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(old.shape[0] + 8,), dtype=np.float32)
        self._cached_action = None
        self._goal_features = np.zeros(8, np.float32)

    def _prepare(self):
        self._cached_action = np.asarray(self.expert.action(self), np.float32)
        pose = self._pose()[0]
        lane = float(self.expert.lanes[self.expert.index % len(self.expert.lanes)])
        phase = self.expert.phase
        target_v = (float(self.expert.last_target) if phase == 'recover_stroke'
                    else self.expert.v1 if self.expert.index % 2 == 0 else self.expert.v0)
        target = np.array([lane, pose[1] if phase in ('align', 'recover_align') else target_v])
        onehot = np.array([float(phase == name) for name in PHASES], np.float32)
        self._goal_features = np.concatenate([
            (target - pose[:2]) / [self.wall.length, self.wall.height],
            onehot,
            [float(self.expert.index) / max(1, len(self.expert.lanes) - 1),
             min(1., float(self.expert.no_progress) / max(1, self.expert.recovery_interval))]
        ]).astype(np.float32)

    def _obs(self):
        obs = super()._obs()
        obs['state'] = np.concatenate([obs['state'], self._goal_features]).astype(np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        # Base reset invokes self._obs(), so zero features must exist first.
        self._goal_features = np.zeros(8, np.float32)
        obs, info = super().reset(seed=seed, options=options)
        self.expert = Stage2PatchSweep(self)
        self._prepare()
        return self._obs(), info

    def step(self, action):
        # Base step invokes self._obs() before returning; refresh it afterward.
        _, reward, terminated, truncated, info = super().step(action)
        if not (terminated or truncated):
            self._prepare()
        return self._obs(), reward, terminated, truncated, info

    def expert_action(self):
        return self._cached_action.copy()


class GoalMapEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.g = nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Flatten())
        self.l = nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Flatten())
        self.s = nn.Sequential(nn.Linear(observation_space['state'].shape[0],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        with th.no_grad():
            gdim = self.g(th.zeros(1,3,core.WholeWallEnv.GLOBAL,core.WholeWallEnv.GLOBAL)).shape[1]
            ldim = self.l(th.zeros(1,3,core.WholeWallEnv.LOCAL,core.WholeWallEnv.LOCAL)).shape[1]
        self.head = nn.Sequential(nn.Linear(gdim+ldim+64,features_dim),nn.ReLU())

    def forward(self, obs):
        g = obs['global_map'].reshape(-1,3,core.WholeWallEnv.GLOBAL,core.WholeWallEnv.GLOBAL)
        l = obs['local_map'].reshape(-1,3,core.WholeWallEnv.LOCAL,core.WholeWallEnv.LOCAL)
        return self.head(th.cat((self.g(g),self.l(l),self.s(obs['state'])),dim=1))


def new_model(env, seed):
    return SAC('MultiInputPolicy', env, policy_kwargs=dict(features_extractor_class=GoalMapEncoder,
        features_extractor_kwargs=dict(features_dim=256),net_arch=[256,256]),
        learning_rate=3e-4, buffer_size=100000, learning_starts=1000,
        batch_size=128, train_freq=(4,'step'), gradient_steps=1,
        seed=seed,device='cpu',verbose=0)


def job(task):
    model_path, wall, seed, beta, limit, expert_only = task
    th.set_num_threads(1)
    env = GoalEnv([wall], episode_steps=limit)
    model = None if expert_only else SAC.load(str(model_path), device='cpu')
    rng = np.random.default_rng(seed + 8731)
    data = {k: [] for k in KEYS}
    expert_count = 0
    try:
        obs, _ = env.reset(seed=seed, options={'wall':wall})
        for step in range(limit):
            label = env.expert_action()
            for k in KEYS[:-1]:
                data[k].append(obs[k].copy())
            data['action'].append(label)
            use_expert = expert_only or rng.random() < beta
            expert_count += int(use_expert)
            action = label if use_expert else model.predict(obs, deterministic=True)[0]
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        row = dict(wall=wall,seed=seed,coverage=float(info['target_coverage']),
                   success=bool(info['is_success']),reason=info['reason'],steps=step+1,
                   expert_fraction=expert_count/(step+1),recoveries=env.expert.recovery_count)
        return {k:np.stack(v).astype(np.float32) for k,v in data.items()},row
    finally:
        env.close()


def jobs(tasks, workers):
    if workers == 1:
        return [job(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        return list(pool.map(job,tasks))


def evaluation(model, args, seed, count):
    env = GoalEnv(args.walls, episode_steps=args.episode_steps)
    rows=[]
    try:
        for i in range(count):
            wall=args.walls[i % len(args.walls)]
            obs,_=env.reset(seed=seed+i,options={'wall':wall})
            while True:
                action,_=model.predict(obs,deterministic=True)
                obs,_,done,truncated,info=env.step(action)
                if done or truncated: break
            row=dict(wall=wall,seed=seed+i,coverage=float(info['target_coverage']),
                     success=bool(info['is_success']),reason=info['reason'],steps=info['steps'])
            rows.append(row)
            print(f"[eval] {wall} coverage={row['coverage']:.1%} reason={row['reason']}",flush=True)
    finally:
        env.close()
    return dict(success_rate=float(np.mean([r['success'] for r in rows])),
                mean_coverage=float(np.mean([r['coverage'] for r in rows])),episodes=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['validate-expert','train','evaluate'],required=True)
    p.add_argument('--out',type=Path,default=Path('runs/stage2_goal_dagger_v2'))
    p.add_argument('--model',type=Path)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--episodes-per-round',type=int,default=16)
    p.add_argument('--episode-steps',type=int,default=2500)
    p.add_argument('--eval-episodes',type=int,default=16)
    p.add_argument('--rounds',type=int,default=12)
    p.add_argument('--patience',type=int,default=4)
    p.add_argument('--epochs-per-round',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--bc-lr',type=float,default=5e-5)
    p.add_argument('--max-samples',type=int,default=100000)
    p.add_argument('--pass-rate',type=float,default=.75)
    p.add_argument('--beta',type=float,default=.5)
    p.add_argument('--seed',type=int,default=360000)
    p.add_argument('--walls',nargs='+',choices=('south','east','north','west'),default=['south','east','north','west'])
    args=p.parse_args()
    if min(args.workers,args.episodes_per_round,args.episode_steps,args.eval_episodes,args.rounds,args.patience) <= 0:
        p.error('counts must be positive')
    if not (0<=args.beta<=1 and 0<=args.pass_rate<=1): p.error('rates must be in [0,1]')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=='validate-expert':
        tasks=[(None,args.walls[i%len(args.walls)],args.seed+i,1.,args.episode_steps,True)
               for i in range(args.episodes_per_round)]
        result=jobs(tasks,args.workers)
        rows=[r for _,r in result]
        (args.out/'expert_validation.json').write_text(json.dumps(rows,indent=2))
        for r in rows: print(f"[expert] {r['wall']} coverage={r['coverage']:.1%} reason={r['reason']}",flush=True)
        print(f"[expert] {sum(r['success'] for r in rows)}/{len(rows)} successes",flush=True)
        if not all(r['success'] for r in rows): raise SystemExit('Expert failed: do not train')
        return
    if args.mode=='evaluate':
        if args.model is None: p.error('--model required')
        model=SAC.load(str(args.model),device='cpu')
        ev=evaluation(model,args,args.seed,args.eval_episodes)
        (args.out/'evaluation.json').write_text(json.dumps(ev,indent=2))
        return
    val_path=args.out/'expert_validation.json'
    if not val_path.exists() or not all(x['success'] for x in json.loads(val_path.read_text())):
        p.error('Validate the expert first in this output directory')
    if (args.out/'best_actor.zip').exists(): p.error('Output already trained; use a new --out')
    env=GoalEnv(args.walls,args.episode_steps)
    try: model=new_model(env,args.seed)
    finally: env.close()
    data=None
    best=(-1.,-1.)
    stale=0
    summary=[]
    # Fixed validation seeds across rounds, disjoint from collection seeds.
    for round_idx in range(1,args.rounds+1):
        started=time.perf_counter()
        if round_idx>1:
            model=SAC.load(str(args.out/'best_actor.zip'),device='cpu')
        snapshot=args.out/'rollout_actor.zip'
        model.save(str(snapshot))
        tasks=[(snapshot,args.walls[i%len(args.walls)],args.seed+10000*round_idx+i,
                1. if round_idx==1 else args.beta,args.episode_steps,round_idx==1)
               for i in range(args.episodes_per_round)]
        results=jobs(tasks,args.workers)
        for _,r in results:
            print(f"[rollout] round={round_idx} {r['wall']} coverage={r['coverage']:.1%} expert={r['expert_fraction']:.0%}",flush=True)
        new={k:np.concatenate([d[k] for d,_ in results],axis=0) for k in KEYS}
        data=new if data is None else merge_data(data,new,args.max_samples,args.seed+round_idx)
        class CloneArgs: pass
        ca=CloneArgs(); ca.bc_lr=args.bc_lr; ca.seed=args.seed; ca.epochs_per_round=args.epochs_per_round; ca.batch_size=args.batch_size
        clone_actor(model,data,ca,round_idx)
        checkpoint=args.out/f'round_{round_idx}_actor.zip'
        model.save(str(checkpoint))
        np.savez_compressed(args.out/'aggregated_demonstrations.npz',**data)
        ev=evaluation(model,args,args.seed+500000,args.eval_episodes)
        rank=(ev['success_rate'],ev['mean_coverage'])
        improved=rank>best
        if improved:
            best=rank; stale=0
            shutil.copy2(checkpoint,args.out/'best_actor.zip')
            (args.out/'best_evaluation.json').write_text(json.dumps(ev,indent=2))
        else: stale+=1
        summary.append(dict(round=round_idx,evaluation=ev,samples=len(data['action']),elapsed_s=time.perf_counter()-started,improved=improved))
        (args.out/'dagger_results.json').write_text(json.dumps(summary,indent=2))
        print(f"[gate] round={round_idx} success={rank[0]:.1%} coverage={rank[1]:.1%} best={best}",flush=True)
        if rank[0]>=args.pass_rate:
            print('[gate] Stage 2 passed; test on new seeds',flush=True);break
        if stale>=args.patience:
            print('[gate] Stopped at patience limit',flush=True);break
    print('[note] Goal-conditioned DAgger actor only; SAC critics are not trained.',flush=True)

if __name__=='__main__':
    mp.freeze_support()
    main()
