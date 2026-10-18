import os
import time
import sys
import argparse
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from planning.rl_env import RLCoverageEnv


def make_env(stage, wall_name, seed=None):
    def _init():
        return RLCoverageEnv(gui=False, stage=stage, wall_name=wall_name, seed=seed, max_steps=500)
    return _init


def train_curriculum(total_timesteps_per_stage=10000, wall_name='south', out_dir='models', seed=0):
    os.makedirs(out_dir, exist_ok=True)
    # set up file logging for this training run
    timestamp = int(time.time())
    log_path = os.path.join(out_dir, f"training_{timestamp}.log")
    logger = logging.getLogger("curriculum_train")
    logger.setLevel(logging.INFO)
    # avoid adding multiple handlers on repeated calls
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Redirect stdout/stderr through a tee so any plain `print()` calls
        # (e.g. from third-party libs) are written to both console and log file.
        class Tee:
            def __init__(self, *writers):
                self.writers = writers

            def write(self, data):
                for w in self.writers:
                    try:
                        w.write(data)
                        w.flush()
                    except Exception:
                        pass

            def flush(self):
                for w in self.writers:
                    try:
                        w.flush()
                    except Exception:
                        pass

        # open a raw file object for print teeing
        file_obj = open(log_path, 'a')
        sys.stdout = Tee(sys.__stdout__, file_obj)
        sys.stderr = Tee(sys.__stderr__, file_obj)

    logger.info('Training logs will be written to %s', log_path)
    model = None
    for stage in (1, 2, 3, 4):
        logger.info('=== Stage %d: creating env ===', stage)
        vec = DummyVecEnv([make_env(stage, wall_name, seed)])
        model_path = os.path.join(out_dir, f"ppo_stage{stage}.zip")

        if model is None:
            logger.info('Creating new PPO model')
            model = PPO("MultiInputPolicy", vec, verbose=1)
        else:
            logger.info('Loading previous model for continued training: %s', prev_path)
            model = PPO.load(prev_path, env=vec)

        logger.info('Training stage %d for %d timesteps', stage, total_timesteps_per_stage)
        model.learn(total_timesteps=total_timesteps_per_stage)
        logger.info('Saving stage %d model -> %s', stage, model_path)
        model.save(model_path)
        prev_path = model_path

    final_path = os.path.join(out_dir, "ppo_coverage_final.zip")
    logger.info('Saving final model -> %s', final_path)
    model.save(final_path)
    logger.info('Training finished. Final model: %s', final_path)
    return final_path


def evaluate_model(model_path, stage=4, wall_name='south', episodes=3, seed=0):
    logger = logging.getLogger("curriculum_train")
    logger.info('Evaluating %s on stage %d for wall %s', model_path, stage, wall_name)
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
        logger.info(' Episode %d reward %.2f coverage %.3f', ep, ep_reward, obs['coverage'])
    env.close()
    logger.info('Avg reward: %.2f', (sum(total_rewards)/len(total_rewards)))


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
