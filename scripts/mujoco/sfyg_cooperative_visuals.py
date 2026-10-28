"""Read-only arrows/labels. Debug geometry never applies simulation force."""
import mujoco
import numpy as np


def draw(scene,payload_position,applied,positions,commands,reactions,phase,controller_label='FT'):
    def geom(kind,pos,color):
        if scene.ngeom >= scene.maxgeom:
            return None
        result = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(result,kind,np.zeros(3),np.asarray(pos,float),np.eye(3).ravel(),np.array(color,np.float32))
        scene.ngeom += 1
        return result
    def label(text,pos,color):
        result = geom(mujoco.mjtGeom.mjGEOM_LABEL,pos,color)
        if result is not None:
            result.label = text
    def arrow(force,origin,color,scale):
        norm = np.linalg.norm(force)
        if norm < .01:
            return
        result = geom(mujoco.mjtGeom.mjGEOM_ARROW,origin,color)
        if result is not None:
            mujoco.mjv_connector(result,mujoco.mjtGeom.mjGEOM_ARROW,.018,origin,
                                 np.asarray(origin)+force/norm*min(.65,norm*scale))
    origin = np.asarray(payload_position)+[0,0,.15]
    arrow(np.asarray(applied)[:3],origin,[1,.2,.1,1],.15)
    label(f"{phase} F=({applied[0]:.1f},{applied[1]:.1f})N Tz={applied[5]:.2f}Nm",
          origin+[0,0,.25],[1,.8,.5,1])
    for i,(position,command,reaction) in enumerate(zip(positions,commands,reactions)):
        color = [.3,1,.5,1] if i == 0 else [.3,.7,1,1]
        label(f"R{i+1} {controller_label} v=({command[0]:.2f},{command[1]:.2f}) w={command[2]:.2f}",
              np.asarray(position)+[0,0,.75],color)
        arrow(np.r_[reaction[:2],0.],np.asarray(position)+[0,0,.5],color,.03)
