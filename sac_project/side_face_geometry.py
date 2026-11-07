"""A thick standalone slab, painted only on its right-hand vertical end face.

The 4 m canvas is a coordinate workspace, not the physical paintable width.
Only the central side-width strip exists in the paintable map. This preserves
valid chassis joint limits for narrow faces without shrinking the robot.
"""
from functools import partial
import math
import numpy as np
import pybullet as p
import train_td3_wall as core
from simulation.paint_wall import Wall, Opening
from train_stage3_goal_dagger_v4 import Stage3TrainEnv
from wall_geometry_env import GeometryScene, GeometryEnv


def side_frame(origin, tangent, normal, slab_length, side_width, workspace_width=4.):
    if not .3 <= side_width <= 2.:
        raise ValueError('side_width must be between 0.3 and 2.0 metres')
    tangent,normal=np.asarray(tangent,float),np.asarray(normal,float)
    left=(workspace_width-side_width)/2
    u_direction=-normal
    side_normal=tangent
    edge=np.asarray(origin,float)+tangent*slab_length
    return edge-u_direction*left,u_direction,side_normal,left


class SideFaceScene(GeometryScene):
    def __init__(self,*args,wall_name,side_width=.6,fast_gui=False,**kwargs):
        super().__init__(*args,wall_name=wall_name,wall_type='flat',
                         standalone=True,fast_gui=fast_gui,**kwargs)
        old=self.walls_by_name[wall_name]
        H,slab_length=old.height,old.length
        workspace_width=4.
        origin,u,normal,left=side_frame(old.origin,old.u_dir,old.normal,slab_length,side_width)
        for body in (old.overlay_id,old.body_id,self.floor_id):
            p.removeBody(body)
        holes=(Opening(0.,0.,left,H),
               Opening(left+side_width,0.,workspace_width-left-side_width,H))
        wall=Wall(wall_name,origin,u,normal,workspace_width,H,.1,holes)
        wall.spawn(self._work.name,core.PPM,(235,232,222),(40,90,200),(.92,.91,.87,1))
        # Replace the thin strip's collision body with the entire original slab.
        # All faces collide, but only the side-face overlay receives paint.
        p.removeBody(wall.body_id)
        half=[side_width/2,slab_length/2,H/2]
        col=p.createCollisionShape(p.GEOM_BOX,halfExtents=half)
        vis=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=[.92,.91,.87,1])
        position=wall.uv_to_world(left+side_width/2,H/2)-normal*slab_length/2
        orn=p.getQuaternionFromEuler([0,0,wall.yaw])
        wall.body_id=p.createMultiBody(0,col,vis,basePosition=position.tolist(),baseOrientation=orn)
        self.walls=[wall]
        self.walls_by_name={wall_name:wall}
        self.width=self.depth=workspace_width
        # Floor supports both the slab and the robot's outward workspace.
        n0,n1=-slab_length-.5,workspace_width+.5
        floor_half=[workspace_width/2+.5,(n1-n0)/2,.05]
        col=p.createCollisionShape(p.GEOM_BOX,halfExtents=floor_half)
        vis=p.createVisualShape(p.GEOM_BOX,halfExtents=floor_half,rgbaColor=[.55,.42,.3,1])
        center=origin+u*workspace_width/2+normal*(n0+n1)/2+np.array([0,0,-.05])
        self.floor_id=p.createMultiBody(0,col,vis,basePosition=center.tolist(),baseOrientation=orn)
        self.geometry=dict(type='side_face',wall=wall_name,u=left,v=0.,width=side_width,
            height=H,slab_length_m=slab_length,slab_thickness_m=side_width,
            target_surface='right-hand narrow vertical end face',
            canvas_workspace_width_m=workspace_width,physical_paintable_width_m=side_width,
            large_faces_paintable=False)
        target=wall.uv_to_world(left+side_width/2,H/2)
        self.CAMERA_VIEWS={i:(max(2.2,H*1.1),math.degrees(wall.yaw)+180,-12,tuple(target))
                           for i in range(1,6)}
        if self.gui:
            self._cam_view=self.CAMERA_VIEWS[1]
            self._apply_cam()


class SideFaceEnv(GeometryEnv):
    def __init__(self,wall,episode_steps,side_width=.6,fast_gui=False):
        super().__init__(wall,'flat',episode_steps,standalone=True,fast_gui=fast_gui)
        self.side_width=side_width
        self.scene_factory=partial(SideFaceScene,wall_name=wall,side_width=side_width,fast_gui=fast_gui)

    def reset(self,*,seed=None,options=None):
        options=dict(options or {})
        # Fixed, reachable centre pose; no reset-pose rejection lottery on a thin strip.
        options.setdefault('command',[2.,.8,.0595,0.])
        return super().reset(seed=seed,options=options)

    def _ik(self,command,previous=None):
        # Permit transit in the wider workspace, but require the full roller
        # inside the real side face for near-wall commands.
        if getattr(self,'sim',None) is not None and command[2]<.09:
            g=self.sim.geometry
            theta=float(command[3])
            half=core.HALF_LENGTH*abs(math.cos(theta))+core.RADIUS*abs(math.sin(theta))+.008
            if not g['u']+half <= command[0] <= g['u']+g['width']-half:
                return None
        return super()._ik(command,previous)

    def _prepare(self):
        # Keep V8's phases and features; fit its initial lanes to actual roller
        # centres, not footprint-edge pixels (which can be unreachable goals).
        if getattr(self,'_side_expert',None) is not self.expert:
            self._side_expert=self.expert
            g=self.sim.geometry
            margin=core.HALF_LENGTH+.008
            lo,hi=g['u']+margin,g['u']+g['width']-margin
            count=max(2,math.ceil((hi-lo)/.075)+1)
            self.expert.lanes=np.linspace(lo,hi,count)
            self.expert.lane_failures=np.zeros(count,dtype=np.int32)
            _,_,v0,v1=core.center_bounds(self.wall.length,self.wall.height,0.)
            self.expert.v0=max(self.expert.v0,v0)
            self.expert.v1=min(self.expert.v1,v1)
        return Stage3TrainEnv._prepare(self)
