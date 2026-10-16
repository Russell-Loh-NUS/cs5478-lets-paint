import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from planning.rl_env import RLCoverageEnv


def make_env(stage, wall_name, seed=None):
    def _init():
        return RLCoverageEnv(gui=False, stage=stage, wall_name=wall_name, seed=seed, max_steps=500)
    return _init


def train_curriculum(total_timesteps_per_stage=10000, wall_name='south', out_dir='models', seed=0):
    os.makedirs(out_dir, exist_ok=True)
    model = None
    for stage in (1, 2, 3, 4):
        print(f"=== Stage {stage}: creating env ===")
        vec = DummyVecEnv([make_env(stage, wall_name, seed)])
        model_path = os.path.join(out_dir, f"ppo_stage{stage}.zip")

        if model is None:
            print("Creating new PPO model")
            # Use MultiInputPolicy because the env observation space is a Dict
            model = PPO("MultiInputPolicy", vec, verbose=1)
        else:
            print(f"Loading previous model for continued training: {prev_path}")
            model = PPO.load(prev_path, env=vec)

        print(f"Training stage {stage} for {total_timesteps_per_stage} timesteps")
        model.learn(total_timesteps=total_timesteps_per_stage)
        print(f"Saving stage {stage} model -> {model_path}")
        model.save(model_path)
        prev_path = model_path

    final_path = os.path.join(out_dir, "ppo_coverage_final.zip")
    print(f"Saving final model -> {final_path}")
    model.save(final_path)
    return final_path


def evaluate_model(model_path, stage=4, wall_name='south', episodes=3, seed=0):
    print(f"Evaluating {model_path} on stage {stage} for wall {wall_name}")
    env = RLCoverageEnv(gui=False, stage=stage, wall_name=wall_name, seed=seed, max_steps=1000)
    model = PPO.load(model_path, env=None)
    total_rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = env.step(int(action) if hasattr(action, '__int__') else action)
            ep_reward += r
        total_rewards.append(ep_reward)
        print(f" Episode {ep} reward {ep_reward:.2f} coverage {obs['coverage']:.3f}")
    env.close()
    print(f"Avg reward: {sum(total_rewards)/len(total_rewards):.2f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--timesteps', type=int, default=2000)
    p.add_argument('--wall', type=str, default='south')
    p.add_argument('--out', type=str, default='models')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval', action='store_true')
    args = p.parse_args()

    final = train_curriculum(total_timesteps_per_stage=args.timesteps, wall_name=args.wall, out_dir=args.out, seed=args.seed)
    if args.eval:
        evaluate_model(final, stage=4, wall_name=args.wall, episodes=2, seed=args.seed)
