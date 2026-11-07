#!/usr/bin/env python3
"""Four-wall Stage3 V4 coverage demonstration, with optional Stage4 neural refill.

Each wall is an independent original Stage3 environment episode. This is NOT
one persistent physical robot traversing between walls. Never claim otherwise.
"""
import argparse
import json
import traceback
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from train_stage3_goal_dagger_v4 import Stage3TrainEnv
from stage3_native_refill_demo import Refill

WALLS = ('south', 'east', 'north', 'west')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paint-actor', type=Path, required=True)
    ap.add_argument('--dip-actor', type=Path)
    ap.add_argument('--walls', nargs='+', choices=WALLS, default=list(WALLS))
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--max-actions', type=int, default=30000)
    ap.add_argument('--seed', type=int, default=860000)
    ap.add_argument('--paint-distance-m', type=float, default=12.0,
                    help='Contact travel metres per full roller (original distance-budget concept)')
    ap.add_argument('--low-fraction', type=float, default=.15)
    ap.add_argument('--max-refills', type=int, default=100,
                    help='Safety cap per wall; increase for full-wall evaluations')
    ap.add_argument('--log-every', type=int, default=500)
    ap.add_argument('--out', type=Path, default=Path('sac_project/runs/four_wall_coverage'))
    args = ap.parse_args()
    if not args.paint_actor.is_file(): ap.error('Painting actor not found')
    if args.dip_actor and not args.dip_actor.is_file(): ap.error('Dipping actor not found')
    if args.max_actions < 1 or args.paint_distance_m <= 0: ap.error('Invalid action or capacity budget')
    args.out.mkdir(parents=True, exist_ok=True)
    painter = SAC.load(str(args.paint_actor), device='cpu')
    if painter.observation_space['state'].shape != (56,) or painter.action_space.shape != (4,):
        raise RuntimeError('Expected original Stage3 V4 56D state and 4D actions')
    dipper = SAC.load(str(args.dip_actor), device='cpu') if args.dip_actor else None
    if dipper is not None and (dipper.observation_space.shape != (36,) or dipper.action_space.shape != (4,)):
        raise RuntimeError('Expected Stage4 V2 36D observation and 4D actions')
    results = []
    for index, wall in enumerate(args.walls):
        env = None
        record = {'wall':wall, 'seed':args.seed+index, 'result':'not_started',
                  'neural_painting':True, 'neural_dipping':bool(dipper),
                  'geometric_refill_travel':bool(dipper), 'independent_wall_reset':True}
        try:
            env = Stage3TrainEnv([wall], episode_steps=args.max_actions)
            env.gui = bool(args.gui)
            env.render_mode = 'human' if args.gui else None
            obs, info = env.reset(seed=args.seed+index, options={'wall':wall})
            refill = Refill(env,dipper) if dipper is not None else None
            level, contact_distance, total_contact_distance = 1.0, 0.0, 0.0
            previous_roller = np.asarray(env.robot.get_roller_position(), dtype=float)
            reason = 'action_limit'
            actions = 0
            for k in range(args.max_actions):
                if refill is not None and level <= args.low_fraction:
                    if refill.refills >= args.max_refills:
                        reason = 'refill_limit'; break
                    obs = refill.run()
                    level = 1.0
                    previous_roller = np.asarray(env.robot.get_roller_position(), dtype=float)
                    contact_distance = 0.0
                action, _ = painter.predict(obs,deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                actions += 1
                roller = np.asarray(env.robot.get_roller_position(), dtype=float)
                travel = float(np.linalg.norm(roller - previous_roller))
                # Canvas.stroke sets _prev; Canvas.lift clears it.
                painting_contact = env.wall.canvas._prev is not None
                if refill is not None and painting_contact:
                    contact_distance += travel
                    total_contact_distance += travel
                    level = max(0.0, 1.0 - contact_distance / args.paint_distance_m)
                previous_roller = roller
                if actions % args.log_every == 0:
                    print('[wall-progress] '+json.dumps({'wall':wall,'actions':actions,
                        'target_coverage':float(info['target_coverage']),
                        'whole_wall_coverage':float(info['wall_coverage']),
                        'refills':refill.refills if refill else 0,
                        'paint_level':round(level,3),
                        'contact_distance_m':round(total_contact_distance,3)}),flush=True)
                if terminated or truncated:
                    reason = str(info.get('reason','terminated')); break
            record.update(result=reason, painting_actions=actions,
                target_coverage=float(info['target_coverage']),
                whole_wall_coverage=float(info['wall_coverage']),
                refills=refill.refills if refill else 0,
                paint_level=level, contact_distance_m=total_contact_distance,
                paint_budget='contact_distance', paint_distance_capacity_m=args.paint_distance_m,
                refill_events=refill.events if refill else [])
        except Exception as exc:
            record.update(result='failed', error=str(exc), traceback=traceback.format_exc())
            print('[wall-error] '+json.dumps({'wall':wall,'error':str(exc)}),flush=True)
        finally:
            if env is not None:
                try: env.close()
                except Exception: pass
            results.append(record)
            (args.out/f'{wall}_result.json').write_text(json.dumps(record,indent=2))
            print('[wall-result] '+json.dumps({k:v for k,v in record.items() if k not in ('refill_events','traceback')}),flush=True)
    valid=[r for r in results if 'target_coverage' in r]
    summary={'wall_results':results,'walls_requested':len(args.walls),
             'walls_evaluated':len(valid),'walls_failed':sum(r['result']=='failed' for r in results),
             'mean_target_coverage':sum(r['target_coverage'] for r in valid)/len(valid) if valid else None,
             'total_refills':sum(r.get('refills',0) for r in results),
             'evaluation_type':'independent original Stage3 wall episodes',
             'continuous_four_wall_physics':False,
             'paint_budget':'roller travel while canvas stroke active',
             'paint_distance_capacity_m':args.paint_distance_m,
             'low_fraction':args.low_fraction,
             'controller_attribution':{'painting':'Stage3 V4 neural DAgger',
                'coverage_goals':'Stage3 rule-based coverage planner',
                'dipping':'Stage4 V2 neural DAgger' if dipper else 'not used',
                'refill_travel':'geometric motor controller' if dipper else 'not used',
                'wall_switch':'reset to next wall frame (not physical navigation)'}}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('[four-wall-summary] '+json.dumps({k:v for k,v in summary.items() if k!='wall_results'}),flush=True)
    return 0 if len(valid)==len(args.walls) and summary['walls_failed']==0 else 1

if __name__=='__main__': raise SystemExit(main())
