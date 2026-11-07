# Train / evaluate the base-motion RL policy (planning/motion_planner_rl.py).
#
# Curriculum (each stage continues the previous weights) -
# 1. rectangular rooms, no doors/windows
# 2. rectangular rooms with doors/windows (leaving through a door gap ends the episode)
# 3. 50% L-shaped rooms
#
# Examples -
# python train_motion_rl.py --timesteps 200000 --out models/motion_rl/my_run
# python train_motion_rl.py --eval-only models/motion_rl/motion_rl_band/motion_rl_final.zip --l-prob 0.5
#
# Trained models live under models/motion_rl/<run name>/ (older ones are in models/motion_rl/archive/,
# which use the old 38-value observation and no longer load).
import argparse
import os
from dataclasses import replace

import numpy as np

# macOS - set the OpenMP workaround before torch is imported (same settings as the coverage planner).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from planning.motion_planner_rl import MotionRLConfig, RLMotionEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

STAGES = (
    dict(l_shape_probability=0.0, keep_openings=False),
    dict(l_shape_probability=0.0, keep_openings=True),
    dict(l_shape_probability=0.5, keep_openings=True),
)


def make_env(cfg, seed):
    def _init():
        return RLMotionEnv(cfg, seed=seed)
    return _init


def evaluate(model, cfg, episodes=50, seed=1234):
    env = RLMotionEnv(cfg, seed=seed)
    stats = {0: [], 1: []}  # phase -> list of (success, collision, oob, steps, path)
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        phase = info["phase"]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        stats[phase].append((info["is_success"], info["collision"], info["out_of_bounds"],
                             info["steps"], info["path_length"], info["inset_frac"]))
    env.close()
    for phase, name in ((0, "outbound (to can)"), (1, "return (to wall)")):
        rows = np.array(stats[phase], dtype=float)
        if len(rows) == 0:
            continue
        print(f"[eval] {name:18s} n={len(rows):3d} success={rows[:, 0].mean():.2f} "
              f"collision={rows[:, 1].mean():.2f} left_room={rows[:, 2].mean():.2f} "
              f"steps={rows[:, 3].mean():.0f} path={rows[:, 4].mean():.2f} m "
              f"within_0.6m_of_wall={rows[:, 5].mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=100_000, help="per curriculum stage")
    ap.add_argument("--stages", type=int, default=3, choices=(1, 2, 3))
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/motion_rl")
    ap.add_argument("--phase", default="mixed", choices=("outbound", "return", "mixed"))
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--eval-only", metavar="MODEL", help="skip training, evaluate this model")
    ap.add_argument("--l-prob", type=float, default=None, help="override L-room probability for eval")
    ap.add_argument("--resume", metavar="MODEL", help="continue training from this model's weights")
    ap.add_argument("--first-stage", type=int, default=1, choices=(1, 2, 3),
                    help="skip the curriculum stages before this one (use with --resume)")
    args = ap.parse_args()

    base =MotionRLConfig(phase=args.phase)
    if args.eval_only:
        cfg = replace(base, keep_openings=True,
                      l_shape_probability=args.l_prob if args.l_prob is not None else 0.0)
        evaluate(PPO.load(args.eval_only, device="cpu"), cfg, args.eval_episodes)
        return

    os.makedirs(args.out, exist_ok=True)
    model = None
    for i in range(args.first_stage, args.stages + 1):
        stage = STAGES[i - 1]
        cfg = replace(base, **stage)
        fns = [make_env(cfg, args.seed + 1000 * i + k) for k in range(args.n_envs)]
        vec = SubprocVecEnv(fns) if args.n_envs > 1 else DummyVecEnv(fns)
        if model is None and args.resume:
            model = PPO.load(args.resume, env=vec, device="cpu")
        elif model is None:
            model = PPO("MlpPolicy", vec, n_steps=2048, batch_size=256, ent_coef=0.005,
                        verbose=1, device="cpu", seed=args.seed)
        else:
            model.set_env(vec)
        print(f"=== stage {i}: {stage} ===")
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)
        path = os.path.join(args.out, f"motion_rl_stage{i}.zip")
        model.save(path)
        print(f"saved {path}")
        evaluate(model, cfg, args.eval_episodes)
        vec.close()
    final = os.path.join(args.out, "motion_rl_final.zip")
    model.save(final)
    print(f"saved {final}")


if __name__ == "__main__":
    main()
