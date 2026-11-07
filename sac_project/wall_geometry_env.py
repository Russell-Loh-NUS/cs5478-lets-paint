"""Opt-in geometry experiment. The flat mode never instantiates this subclass."""
import copy
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path
import sys
import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_td3_wall as core
from simulation.paint_wall import Wall, Opening
from simulation.robot import Robot
from train_stage3_goal_dagger_v4 import Stage3TrainEnv
from wall_geometry_paths import detour, sweep_segments


class CollisionRobot(Robot):
    """Add the missing arm collision solids from existing URDF visual primitives."""
    def _write_urdf(self):
        path = super()._write_urdf()
        tree = ET.parse(path)
        for link in tree.getroot().findall('link'):
            if link.get('name') not in ('shoulder_link','extension_link','wrist_roll_link','roller_handle_link'):
                continue
            if link.find('collision') is None:
                visual = link.find('visual')
                collision = ET.SubElement(link,'collision')
                for key in ('origin','geometry'):
                    if visual.find(key) is not None:
                        collision.append(copy.deepcopy(visual.find(key)))
        tree.write(path)
        return path


class GeometryScene(core.WallScene):
    def __init__(self, *args, wall_name, wall_type, standalone=False, fast_gui=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fast_gui = fast_gui
        old = self.walls_by_name[wall_name]
        if standalone:
            for other in self.walls:
                if other is not old:
                    p.removeBody(other.overlay_id)
                    p.removeBody(other.body_id)
            self.walls = [old]
            self.walls_by_name = {wall_name: old}
        self.extra_collision_bodies = []
        if wall_type == 'flat':
            return
        width = .8 if wall_type == 'hole' else .5
        height, depth = .6, .15
        u, v = old.length*.5-width*.5, min(.9,old.height-height-.3)
        self.geometry = dict(type=wall_type,wall=wall_name,u=u,v=v,width=width,height=height,
                             depth=depth if wall_type=='protrusion' else 0.)
        self.extra_collision_bodies=[]
        if wall_type == 'hole':
            new = Wall(old.name,old.origin,old.u_dir,old.normal,old.length,old.height,
                       old.thickness,(Opening(u,v,width,height),),old.ext)
            p.removeBody(old.overlay_id)
            p.removeBody(old.body_id)
            new.spawn(self._work.name,core.PPM,(235,232,222),(40,90,200),(.92,.91,.87,1))
            self.walls[next(i for i,w in enumerate(self.walls) if w is old)]=new
            self.walls_by_name[wall_name]=new
        else:
            half=[width/2,(depth+old.thickness)/2,height/2]
            col=p.createCollisionShape(p.GEOM_BOX,halfExtents=half)
            vis=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=[.92,.91,.87,1])
            pos=old.uv_to_world(u+width/2,v+height/2)+old.normal*(depth-old.thickness)/2
            body=p.createMultiBody(0,col,vis,basePosition=pos.tolist(),
                                  baseOrientation=p.getQuaternionFromEuler([0,0,old.yaw]))
            self.extra_collision_bodies.append(body)
            c=old.canvas
            rows,cols=np.indices(c.paintable.shape)
            us=old.length-(cols+.5)/c.ppm
            vs=old.height-(rows+.5)/c.ppm
            c.paintable &= ~((us>=u)&(us<u+width)&(vs>=v)&(vs<v+height))


    def step(self):
        if not self.fast_gui or not self.gui:
            return super().step()
        # Same timestep/substeps and collision checks; only display work changes.
        if not p.isConnected(self.client):
            raise RuntimeError('simulation_disconnected: GUI was closed or stopped')
        p.stepSimulation()
        self._steps += 1
        if self._steps % 240 == 0:
            self.flush_textures()
        if self._steps % 12 == 0:
            self._camera_keys()


class GeometryEnv(Stage3TrainEnv):
    def __init__(self, wall, wall_type, episode_steps, standalone=False, fast_gui=False):
        self.scene_factory=partial(GeometryScene,wall_name=wall,wall_type=wall_type,
                                   standalone=standalone,fast_gui=fast_gui)
        self.robot_factory=CollisionRobot
        self.route=None
        self.check_proposals=False
        self.collision_events=[]
        super().__init__([wall],episode_steps=episode_steps)

    def reset(self, *, seed=None, options=None):
        self.route=None
        self.check_proposals=False
        self.collision_events=[]
        obs,info=super().reset(seed=seed,options=options)
        original_step=self.sim.step
        def checked_step():
            original_step()
            # Includes refill motor travel and dipping; queries must fail closed.
            if self.planner.current_collision(self.wall.body_id,executing=True):
                event={'physics_scene_step':self.sim._steps,'painting_step':self.steps,
                       'reason':'collision','roller_world':list(self.robot.get_roller_position())}
                self.collision_events.append(event)
                self.wall.canvas.lift()
                raise RuntimeError('collision: '+str(event))
        self.sim.step=checked_step
        return obs,info

    def _ik(self, command, previous=None):
        q=super()._ik(command,previous)
        if q is not None and self.check_proposals:
            # Saved/restored FK collision probes only; execution always uses motors.
            start=np.asarray(self.robot.get_joint_states(),float)
            for fraction in (.25,.5,.75,1.):
                candidate=start+fraction*(np.asarray(q)-start)
                if self.planner._predicted_collision(candidate,self.wall.body_id):
                    return None
        return q

    def _drive(self, action):
        self.check_proposals=True
        try:
            return super()._drive(action)
        finally:
            self.check_proposals=False

    def _paint(self, draw):
        # Reject complete roller footprints crossing absent/obstructed material.
        _,_,a,b=self._pose()
        a,b=self.wall.world_to_uv(a),self.wall.world_to_uv(b)
        count=max(2,int(np.linalg.norm(np.subtract(a,b))*self.wall.canvas.ppm*2)+1)
        if not all(self._solid_at(q) for q in np.linspace(a,b,count)):
            self.wall.canvas.lift()
            return False
        return super()._paint(draw)

    def _prepare(self):
        # Keep Stage3's first 40 state values, maps and 16 goal-feature meanings.
        # Only geometry-aware target scheduling replaces the solid-wall V8 sweep.
        pose=self._pose()[0]
        if self.route is None:
            g=self.sim.geometry
            # Bound the entire roller at any orientation, plus tracking margin.
            margin=self.robot.ROLLER_LENGTH/2+self.robot.ROLLER_RADIUS+.03
            self.rect=(g['u']-margin,g['u']+g['width']+margin,
                       g['v']-margin,g['v']+g['height']+margin)
            self.bounds=core.center_bounds(self.wall.length,self.wall.height,0.)
            self.segments=sweep_segments(self.bounds,self.rect,
                lambda uv:self._ik(np.array([*uv,.0595,0.])) is not None)
            self.segment_index=0
            self.route=[]
            self.phase='choose'
            self.phase_started=self.steps
        if self.steps-self.phase_started>1000:
            raise RuntimeError('route_tracking_timeout: '+self.phase)
        if self.phase=='choose':
            if self.segment_index>=len(self.segments):
                raise RuntimeError('geometry_route_exhausted')
            start,end=self.segments[self.segment_index]
            self.route=detour(pose[:2],start,self.rect,self.bounds)
            self.stroke_end=end
            self.goal=pose[:2].copy()
            self.phase='lift'
            self.phase_started=self.steps
        normal=.105 if self.phase in ('lift','rotate','align') else .0595
        angle=abs((self.command[3]+np.pi)%(2*np.pi)-np.pi)
        reached=np.linalg.norm(pose[:2]-self.goal)<.012
        phase=self.phase
        if phase=='lift' and abs(pose[2]-.105)<.008:
            self.phase='rotate'
        elif phase=='rotate' and angle<.05:
            self.phase='align'; self.goal=self.route.pop(0)
        elif phase=='align' and reached:
            if self.route: self.goal=self.route.pop(0)
            else: self.phase='lower'
        elif phase=='lower' and abs(pose[2]-.0595)<.004:
            self.phase='stroke'; self.goal=self.stroke_end
        elif phase=='stroke' and reached:
            self.segment_index+=1; self.phase='choose'
        if self.phase!=phase:
            self.phase_started=self.steps
        normal=.105 if self.phase in ('lift','rotate','align') else .0595
        stroke=self.phase=='stroke'
        base8=np.concatenate(((self.goal-pose[:2])/[self.wall.length,self.wall.height],
                              [0.,0.,float(not stroke),float(stroke)],
                              [self.segment_index/max(1,len(self.segments)-1),
                               min(1.,(self.steps-self.last_progress_step)/100.)]))
        extra=[(normal-pose[2])/.06,0.,1.,float(self.phase=='lift'),
               float(self.phase=='rotate'),float(self.phase=='align'),
               float(self.phase=='lower'),float(stroke)]
        self._goal_features=np.asarray([*base8,*extra],np.float32)
        self._cached_action=np.zeros(4,np.float32) # Never used for execution.


def verify_scene(env):
    """Actual Bullet overlap probes, independent of the robot's current location."""
    g=getattr(env.sim,'geometry',None)
    wall=env.wall
    u=g['u']+g['width']/2 if g else wall.length/2
    v=g['v']+g['height']/2 if g else wall.height/2
    shape=p.createCollisionShape(p.GEOM_SPHERE,radius=.005)
    probe=p.createMultiBody(0,shape,basePosition=(wall.uv_to_world(u,v)-wall.normal*.05).tolist())
    try:
        hits=p.getClosestPoints(probe,wall.body_id,distance=0.)
        if hits is None: raise RuntimeError('scene_probe: collision query returned None')
        expected_wall=not (g and g['type']=='hole')
        if bool(hits)!=expected_wall:
            raise RuntimeError('scene_probe: wall material does not match requested geometry')
        results={'wall_material_probe_passed':True}
        if g and g['type']=='side_face':
            if not env._solid_at((u,v)):
                raise RuntimeError('scene_probe: side-face centre is not paintable')
            if len(env.sim.walls)!=1:
                raise RuntimeError('scene_probe: surrounding walls remain')
            expected=g['width']*wall.canvas.ppm*wall.canvas.height_px
            if abs(int(wall.canvas.paintable.sum())-expected)>wall.canvas.height_px:
                raise RuntimeError('scene_probe: paintable area exceeds side-face strip')
            results['side_face_mask_probe_passed']=True
            results['single_wall_probe_passed']=True
        elif g:
            if env._solid_at((u,v)):
                raise RuntimeError('scene_probe: exclusion centre remains paintable')
            results['excluded_mask_probe_passed']=True
        if g and g['type']=='protrusion':
            p.resetBasePositionAndOrientation(probe,
                (wall.uv_to_world(u,v)+wall.normal*g['depth']/2).tolist(),[0,0,0,1])
            hits=p.getClosestPoints(probe,env.sim.extra_collision_bodies[0],distance=0.)
            if hits is None or not hits:
                raise RuntimeError('scene_probe: missing physical protrusion')
            results['protrusion_collision_probe_passed']=True
        return results
    finally:
        p.removeBody(probe)
