"""Tandem and lightweight sofa geometry; no world supports or pose drives."""
import xml.etree.ElementTree as ET
import numpy as np

from sfyg_bridge import numbers


def placements(layout, spacing):
    if layout == 'leader-follow':
        if spacing < 1.4:
            raise ValueError('Tandem base spacing must be >=1.4 m')
        return [[spacing,0.],[0.,0.]]
    if layout == 'sofa':
        if spacing < 1.:
            raise ValueError('Sofa follower spacing must be >=1 m')
        return [[1.6,0.],[-.9,-spacing/2],[-.9,spacing/2]]
    raise ValueError('Unknown team layout')


def add_payload(root, probe, state, count, mass, friction, grasp, layout, spacing):
    world, equality, contact = (root.find(t) for t in ('worldbody','equality','contact'))
    points = np.array([state.xpos[probe.body(f'r{i}/grasp_frame').id].copy() for i in range(count)])
    orientations = [state.xquat[probe.body(f'r{i}/grasp_frame').id].copy() for i in range(count)]
    center = points.mean(0)-[0,0,.08] if layout=='leader-follow' else np.array([.35,0.,points[:,2].min()-.32])
    payload = ET.SubElement(world,'body',name='payload',pos=numbers(center))
    ET.SubElement(payload,'freejoint',name='payload/free')
    pieces = []
    def box(name,pos,size,color):
        geom = ET.SubElement(payload,'geom',name='payload/'+name,type='box',pos=numbers(pos),
            size=numbers(size),rgba=color,friction=numbers([friction,0,0]),condim='3')
        pieces.append((geom,8*np.prod(size)))
    if layout=='leader-follow':
        # A free finite-mass beam with the COM below both ball grips.
        length = points[0,0]-points[1,0]
        if length < .3: raise ValueError('Insufficient clearance for tandem beam')
        box('beam',[0,0,0],[length/2,.045,.04],'.15 .55 .8 1')
    else:
        half_width = spacing/2+.30
        box('frame',[0,0,-.08],[.40,half_width,.09],'.16 .09 .05 1')
        box('back',[-.32,0,.18],[.08,half_width,.25],'.15 .28 .38 1')
        for i,y in enumerate(np.linspace(-half_width*.62,half_width*.62,3)):
            box(f'cushion{i}',[.045,y,.055],[.30,half_width/3-.015,.07],'.23 .48 .60 1')
        for sign in (-1,1):
            box(f'armrest{sign}',[0,sign*(half_width-.065),.10],[.40,.07,.16],'.18 .36 .46 1')
            for xsign in (-1,1):
                box(f'leg{xsign}_{sign}',[xsign*.30,sign*(half_width-.12),-.25],
                    [.035,.035,.08],'.12 .075 .045 1')
    # Physical geometry, COM and inertia agree; no hidden ballast/extra payload.
    total_volume = sum(volume for _,volume in pieces)
    for geom,volume in pieces: geom.set('mass',str(mass*volume/total_volume))
    for i,point in enumerate(points):
        local = point-center
        endpoint = local.copy()
        if layout=='leader-follow': endpoint[2]=0
        else: endpoint[0]=.4 if i==0 else -.4
        ET.SubElement(payload,'geom',name=f'payload/handle_bar{i}',type='capsule',size='.012',
            fromto=numbers(np.r_[endpoint,local]),mass='0',contype='0',conaffinity='0',rgba='.7 .7 .7 1')
        ET.SubElement(payload,'site',name=f'payload/handle_{i}',pos=numbers(local),
            quat=numbers(orientations[i]),size='.018',rgba='1 .6 .1 1' if i==0 else '.2 .8 1 1')
        # Leader always uses a rear ball hitch; follower weld is explicit opt-in.
        ball = i==0 or grasp=='ball'
        eq = ET.SubElement(equality,'connect' if ball else 'weld',name=f'grasp_{i}',
            site1=f'r{i}/grasp_site',site2=f'payload/handle_{i}',active='false',
            solref='.04 1',solimp='.99 .99 .001')
        if not ball: eq.set('torquescale','.2')
        # Only the local attachment pair is filtered; retain body/leg collisions.
        for name in (('grasp_frame',) if i==0 else ('gripper_base_Link','gripper1_Link','gripper2_Link')):
            ET.SubElement(contact,'exclude',body1=f'r{i}/{name}',body2='payload')
    for meter in range(-5,31):
        ET.SubElement(world,'geom',name=f'team_mark/{meter}',type='box',pos=f'{meter} 0 .001',
            size='.012 3 .0005',contype='0',conaffinity='0',rgba='.8 .65 .2 .5')
    for name,x,color in (('start',0,'.2 .7 .4 .6'),('goal',5,'.2 .5 .9 .6')):
        ET.SubElement(world,'geom',name='zone/'+name,type='box',pos=f'{x} 0 .002',
            size='.05 2 .001',contype='0',conaffinity='0',rgba=color)
