"""Read-only role labels, measured reactions and actual commands."""
import mujoco
import numpy as np


def draw_team(scene, positions, payload, commands, reactions, phase, mass, follower):
    def new(kind,position,color):
        if scene.ngeom>=scene.maxgeom: return None
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,kind,np.zeros(3),np.asarray(position,float),np.eye(3).ravel(),np.array(color,np.float32))
        scene.ngeom+=1
        return geom
    def label(text,pos,color):
        geom=new(mujoco.mjtGeom.mjGEOM_LABEL,pos,color)
        if geom is not None: geom.label=text
    label(f'{phase} | {mass:g}kg light load | {follower}',np.asarray(payload)+[0,0,-.5],[1,.9,.7,1])
    for i,(pos,cmd,reaction) in enumerate(zip(positions,commands,reactions)):
        color=[1,.65,.2,1] if i==0 else [.2,.8,1,1]
        label(f'{"LEADER" if i==0 else "F"+str(i)} v={cmd[0]:.2f}',np.asarray(pos)+[0,0,.8],color)
        force=np.r_[reaction[:2],0.]
        norm=np.linalg.norm(force)
        if norm>.01:
            start=np.asarray(pos)+[0,0,.5]
            geom=new(mujoco.mjtGeom.mjGEOM_ARROW,start,color)
            if geom is not None:
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.015,start,start+force/norm*min(.6,norm*.025))
