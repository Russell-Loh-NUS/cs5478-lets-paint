#!/usr/bin/env python3
"""Frozen-policy geometry evaluation. One independent wall episode per invocation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback
import time
from functools import partial


def write(path, record):
    path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--wall-type',choices=('flat','hole','protrusion'),required=True)
    ap.add_argument('--surface',choices=('front','side'),default='front')
    ap.add_argument('--side-width',type=float,default=.6)
    ap.add_argument('--wall',choices=('south','east','north','west'),default='south')
    ap.add_argument('--paint-actor',type=Path)
    ap.add_argument('--dip-actor',type=Path)
    ap.add_argument('--paint-distance-m',type=float,default=12.)
    ap.add_argument('--low-fraction',type=float,default=.15)
    ap.add_argument('--max-refills',type=int,default=100)
    ap.add_argument('--max-actions',type=int,default=30000)
    ap.add_argument('--seed',type=int,default=860000)
    ap.add_argument('--log-every',type=int,default=500)
    ap.add_argument('--gui',action='store_true')
    ap.add_argument('--fast-gui',action='store_true',help='Reduce display overhead; physics speed unchanged')
    ap.add_argument('--scene',choices=('room','standalone'),default='room')
    ap.add_argument('--painting-only',action='store_true',help='Stage3 only, no can/refill, unlimited paint')
    ap.add_argument('--out',type=Path,default=Path('sac_project/runs/wall_geometry'))
    ap.add_argument('--smoke-test',action='store_true',help='Reset and hold for 5 steps; no neural evaluation')
    args=ap.parse_args()
    if not 0<args.low_fraction<1 or args.paint_distance_m<=0 or min(args.max_actions,args.log_every)<1 or args.max_refills<0:
        ap.error('Invalid budgets or low-fraction')
    if args.surface=='side':
        if args.wall_type!='flat' or not args.painting_only:
            ap.error('Side-face mode requires --wall-type flat --painting-only')
        if not .3<=args.side_width<=2.:
            ap.error('--side-width must be between 0.3 and 2.0 metres')
        args.scene='standalone'
    args.out.mkdir(parents=True,exist_ok=True)
    record=dict(wall_type=args.wall_type,wall=args.wall,seed=args.seed,status='not_started',
        termination_reason=None,failure_reason=None,target_coverage=None,whole_valid_wall_coverage=None,
        verified_refills=0,painting_actions=0,painting_contact_distance_m=0.,collision_count=0,
        neural_painting_used=False,neural_dipping_used=False,independent_wall_episode=True,
        test_type='environment_smoke' if args.smoke_test else 'frozen_policy_evaluation',
        paint_budget=dict(capacity_m=args.paint_distance_m,low_fraction=args.low_fraction),
        controllers=dict(painting='Stage3 V4 neural actor',dipping='Stage4 V2 neural actor',
            refill_travel='original geometric motor controller',
            goals='original V8' if args.wall_type=='flat' else 'split lanes and rectangle visibility detours'),
        comparison_limitations=[] if args.wall_type=='flat' else [
            'Geometry modes add missing arm collision shapes and replace goal scheduling.',
            'This evaluates the frozen low-level policy in an adapted hierarchy, not geometry alone.',
            'Clearance bands remain in coverage denominators; a one-pass route may leave gaps.'])
    record.update(scene=args.scene,painting_only=args.painting_only,fast_gui=args.fast_gui,
                  paint_supply='unlimited' if args.painting_only else 'distance_budget')
    if args.painting_only:
        record['controllers'].update(dipping='disabled',refill_travel='disabled')
        record['paint_budget']=None
    record.update(surface=args.surface)
    if args.surface=='side':
        record['comparison_limitations']=[
            'Side-only paintable strip in a 4 m coordinate workspace; physical side width is recorded separately.',
            'Original V8 goal scheduler with side-clearance IK constraints and added arm collisions.',
            'Fixed initial centre pose; baseline random reset pose is not matched.']
        record['controllers']['goals']='original V8 with side-face feasibility constraints'
    started=time.perf_counter()
    env=refill=None
    level=1.; since_refill=total_distance=0.; actions=0
    try:
        missing=[n for n in ('numpy','PIL','pybullet','torch','gymnasium','stable_baselines3')
                 if importlib.util.find_spec(n) is None]
        if missing:
            record['missing_packages']=missing
            raise RuntimeError('dependencies_unavailable: '+', '.join(missing))
        if not args.smoke_test:
            for name,path in ([('paint_actor',args.paint_actor)] +
                              ([] if args.painting_only else [('dip_actor',args.dip_actor)])):
                if path is None or not path.is_file():
                    raise RuntimeError('checkpoint_unavailable: '+name)
                record[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        import numpy as np
        from stable_baselines3 import SAC
        from train_stage3_goal_dagger_v4 import Stage3TrainEnv
        from stage3_native_refill_demo import Refill
        from wall_geometry_env import GeometryEnv, GeometryScene, verify_scene
        if args.surface=='side':
            from side_face_geometry import SideFaceEnv
            env=SideFaceEnv(args.wall,args.max_actions,args.side_width,args.fast_gui)
        elif args.wall_type=='flat':
            env=Stage3TrainEnv([args.wall],episode_steps=args.max_actions)
            if args.scene=='standalone' or args.fast_gui:
                env.scene_factory=partial(GeometryScene,wall_name=args.wall,wall_type='flat',
                    standalone=args.scene=='standalone',fast_gui=args.fast_gui)
        else:
            env=GeometryEnv(args.wall,args.wall_type,args.max_actions,
                            standalone=args.scene=='standalone',fast_gui=args.fast_gui)
        env.gui=args.gui
        env.render_mode='human' if args.gui else None
        obs,info=env.reset(seed=args.seed,options={'wall':args.wall})
        record['physical_wall_count']=len(env.sim.walls)
        c=env.wall.canvas
        np.savez_compressed(args.out/'masks.npz',paintable=c.paintable,target=env.target,
                            painted=c.painted)
        record['geometry']=dict(getattr(env.sim,'geometry',dict(type='flat',wall=args.wall)),
            wall_length_m=env.wall.length,wall_height_m=env.wall.height,
            origin=env.wall.origin.tolist(),u_direction=env.wall.u_dir.tolist(),normal=env.wall.normal.tolist())
        record['mask']=dict(file='masks.npz',shape=list(c.paintable.shape),pixels_per_m=c.ppm,
            axes='row=(height-v)*ppm; column=(length-u)*ppm',
            target_area_m2=env.target_area,valid_wall_area_m2=c.paintable_area,
            target_pixels=int(env.target.sum()),paintable_pixels=int(c.paintable.sum()))
        if args.wall_type!='flat':
            record['route']=dict(exclusion_for_roller_centres=list(env.rect),
                segments=[ [a.tolist(),b.tolist()] for a,b in env.segments])
        if args.smoke_test:
            record["scene_checks"]=verify_scene(env)
            for _ in range(min(5,args.max_actions)):
                obs,_,terminated,truncated,info=env.step(np.zeros(4,np.float32))
                if terminated or truncated: break
            record.update(status='smoke_passed',termination_reason=info['reason'])
        else:
            painter=SAC.load(str(args.paint_actor),device='cpu')
            dipper=None if args.painting_only else SAC.load(str(args.dip_actor),device='cpu')
            if painter.observation_space['state'].shape!=(56,) or painter.action_space.shape!=(4,):
                raise RuntimeError('Expected Stage3 V4 56D state and 4D actions')
            if dipper is not None and (dipper.observation_space.shape!=(36,) or dipper.action_space.shape!=(4,)):
                raise RuntimeError('Expected Stage4 V2 36D state and 4D actions')
            # Original Refill and its original distance accounting are reused.
            class TrackedDipper:
                def predict(self, *a, **kw):
                    record['neural_dipping_used']=True
                    return dipper.predict(*a, **kw)
            refill=Refill(env,TrackedDipper()) if dipper is not None else None
            previous=np.asarray(env.robot.get_roller_position(),float)
            record.update(status='completed',termination_reason='action_limit')
            for _ in range(args.max_actions):
                if refill is not None and level<=args.low_fraction:
                    if refill.refills>=args.max_refills:
                        record['termination_reason']='refill_limit';break
                    obs=refill.run()
                    level=1.;since_refill=0.
                    previous=np.asarray(env.robot.get_roller_position(),float)
                action,_=painter.predict(obs,deterministic=True)
                record['neural_painting_used']=True
                actions+=1
                obs,_,terminated,truncated,info=env.step(action)
                roller=np.asarray(env.robot.get_roller_position(),float)
                travel=float(np.linalg.norm(roller-previous))
                if env.wall.canvas._prev is not None:
                    since_refill+=travel;total_distance+=travel
                    if not args.painting_only:
                        level=max(0.,1.-since_refill/args.paint_distance_m)
                previous=roller
                if actions%args.log_every==0:
                    print(json.dumps(dict(actions=actions,target_coverage=info['target_coverage'],
                          whole_valid_wall_coverage=info['wall_coverage'],paint_level=None if args.painting_only else level,
                          refills=refill.refills if refill else 0,
                          phase=getattr(env,'phase','original_v8'),
                          elapsed_seconds=round(time.perf_counter()-started,2))),flush=True)
                if terminated or truncated:
                    record['termination_reason']=info['reason'];break
    except KeyboardInterrupt:
        record.update(status='interrupted',termination_reason='user_interrupt',failure_reason=None)
    except Exception as exc:
        reason=str(exc)
        record.update(status='blocked' if reason.startswith(('dependencies_unavailable','checkpoint_unavailable')) else 'failed',
                      termination_reason=reason.split(':')[0],failure_reason=reason,
                      traceback=traceback.format_exc())
        # Only classify a user-closed connection when Bullet confirms disconnect.
        if env is not None:
            try:
                import pybullet as p
                if not p.isConnected(env._client):
                    record.update(status='interrupted',termination_reason='simulation_disconnected')
            except Exception:
                pass
        if reason=='geometry_route_exhausted':
            record.update(status='completed',failure_reason=None)
    finally:
        record.update(painting_actions=actions,painting_contact_distance_m=total_distance,
                      paint_level=None if args.painting_only else level,
                      elapsed_seconds=round(time.perf_counter()-started,2))
        if refill is not None:
            record.update(verified_refills=sum(e['event']=='refill_verified' for e in refill.events),
                          completed_refill_returns=refill.refills,refill_events=refill.events)
        if env is not None:
            try:
                if env.wall is not None and getattr(env,'target',None) is not None:
                    import numpy as np
                    c=env.wall.canvas
                    record.update(target_coverage=float((c.painted&env.target).sum()/env.target.sum()),
                                  whole_valid_wall_coverage=c.painted_area/c.paintable_area,
                                  target_area_m2=float(env.target.sum()/c.ppm**2),valid_wall_area_m2=c.paintable_area)
                    np.savez_compressed(args.out/'masks.npz',paintable=c.paintable,target=env.target,painted=c.painted)
                events=getattr(env,'collision_events',[])
                record['collision_events']=events
                record['collision_count']=max(len(events),int(record['termination_reason']=='collision'))
            except Exception as exc:
                record['final_metrics_error']=str(exc)
            try: env.close()
            except Exception as exc: record['close_error']=str(exc)
        write(args.out/'result.json',record)
        print(json.dumps({k:v for k,v in record.items() if k not in ("route","traceback")},indent=2),flush=True)
    return 0 if record['status'] in ('completed','smoke_passed') and record['termination_reason']!='collision' else 1


if __name__=='__main__':
    raise SystemExit(main())
