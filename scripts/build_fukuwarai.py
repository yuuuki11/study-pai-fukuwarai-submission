#!/usr/bin/env python3
"""Build the mouth-only scene; analyze PNG alpha to make meshes, never edit pixels.
Run inside the supplied ROS container (Pillow and NumPy are installed there).
"""
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_SCALE = 1.5
REFERENCE_BOARD_CENTER = (0.24, 0.0, 1.017)
REFERENCE_BOARD_LENGTH, REFERENCE_BOARD_WIDTH = 0.30, 0.20
BOARD_LENGTH = REFERENCE_BOARD_LENGTH * GEOMETRY_SCALE
BOARD_WIDTH = REFERENCE_BOARD_WIDTH * GEOMETRY_SCALE
SURFACE_Z, THICKNESS = 1.019, 0.008


def element(parent, tag, value=None, **attrs):
    child = ET.SubElement(parent, tag, attrs)
    if value is not None:
        child.text = str(value)
    return child


def write_xml(path, root):
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def shape(link, name, xyz, geometry, color=None, collision=False, friction=0.35):
    node = element(link, 'collision' if collision else 'visual', name=name)
    element(node, 'pose', f'{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0')
    kind, dims = geometry
    body = element(element(node, 'geometry'), kind)
    if kind == 'box':
        element(body, 'size', ' '.join(map(str, dims)))
    elif kind == 'mesh':
        element(body, 'uri', dims)
    else:
        element(body, 'radius', dims[0])
        element(body, 'length', dims[1])
    if color:
        material = element(node, 'material')
        element(material, 'ambient', color)
        element(material, 'diffuse', color)
    if collision:
        ode = element(element(element(node, 'surface'), 'friction'), 'ode')
        element(ode, 'mu', friction)
        element(ode, 'mu2', friction)
    return node


def texture(visual, uri):
    material = element(visual, 'material')
    element(material, 'ambient', '1 1 1 1')
    element(material, 'diffuse', '1 1 1 1')
    metal = element(element(material, 'pbr'), 'metal')
    element(metal, 'albedo_map', uri)
    element(metal, 'metalness', 0)
    element(metal, 'roughness', 1)


def model(name, static, pose):
    node = ET.Element('model', name=name)
    element(node, 'static', str(static).lower())
    element(node, 'pose', ' '.join(map(str, pose))+' 0 0 0')
    return node, element(node, 'link', name='body')


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def triangulate(points):
    ids, result = list(range(len(points))), []
    while len(ids) > 3:
        for k, i in enumerate(ids):
            a, b, c = ids[k-1], i, ids[(k+1) % len(ids)]
            if cross(points[a], points[b], points[c]) <= 1e-12:
                continue
            if any(all(v >= -1e-12 for v in (
                cross(points[a], points[b], points[j]),
                cross(points[b], points[c], points[j]),
                cross(points[c], points[a], points[j])))
                for j in ids if j not in (a,b,c)):
                continue
            result.append((a,b,c))
            del ids[k]
            break
        else:
            raise ValueError('Mouth outline cannot be triangulated')
    result.append(tuple(ids))
    return result


def hull(points):
    points = sorted(set(points))
    lower, upper = [], []
    for chain, ordered in ((lower,points),(upper,list(reversed(points)))):
        for p in ordered:
            while len(chain) > 1 and cross(chain[-2],chain[-1],p) <= 0:
                chain.pop()
            chain.append(p)
    return lower[:-1]+upper[:-1]


def collada(path, vertices, faces, uvs=None):
    ns = 'http://www.collada.org/2005/11/COLLADASchema'
    ET.register_namespace('', ns)
    root = ET.Element('COLLADA', xmlns=ns, version='1.4.1')
    asset = element(root,'asset')
    element(asset,'unit',name='meter',meter='1')
    element(asset,'up_axis','Z_UP')
    geometry = element(element(root,'library_geometries'),'geometry',id='mesh',name='mesh')
    mesh = element(geometry,'mesh')
    # DART requires one explicit normal per vertex for mesh collision shapes.
    normals = np.zeros((len(vertices),3))
    for a,b,c in faces:
        normal = np.cross(np.array(vertices[b])-vertices[a],np.array(vertices[c])-vertices[a])
        for i in (a,b,c):
            normals[i] += normal
    lengths = np.linalg.norm(normals,axis=1)
    if np.any(lengths < 1e-15):
        raise ValueError('Degenerate mesh normals')
    normals = (normals/lengths[:,None]).tolist()
    for name, data, labels in [('positions', vertices, ['X','Y','Z']),('normals',normals,['X','Y','Z'])]+(
            [('uv', uvs, ['S','T'])] if uvs else []):
        source = element(mesh,'source',id=name)
        element(source,'float_array',' '.join(str(v) for p in data for v in p),
                id=name+'-array',count=str(len(data)*len(labels)))
        accessor = element(element(source,'technique_common'),'accessor',
                           source='#'+name+'-array',count=str(len(data)),stride=str(len(labels)))
        for label in labels:
            element(accessor,'param',name=label,type='float')
    element(element(mesh,'vertices',id='verts'),'input',semantic='POSITION',source='#positions')
    tris = element(mesh,'triangles',count=str(len(faces)))
    element(tris,'input',semantic='VERTEX',source='#verts',offset='0')
    element(tris,'input',semantic='NORMAL',source='#normals',offset='1')
    if uvs:
        element(tris,'input',semantic='TEXCOORD',source='#uv',offset='2',set='0')
    element(tris,'p',' '.join(str(i) for face in faces for idx in face
                            for i in ((idx,idx,idx) if uvs else (idx,idx))))
    scene = element(element(root,'library_visual_scenes'),'visual_scene',id='Scene')
    element(element(scene,'node',id='node'),'instance_geometry',url='#mesh')
    element(element(root,'scene'),'instance_visual_scene',url='#Scene')
    write_xml(path,root)


def extruded_mesh(path, polygon, thickness):
    n = len(polygon)
    verts = [(x,y,-thickness/2) for x,y in polygon]+[(x,y,thickness/2) for x,y in polygon]
    faces = [(c,b,a) for a,b,c in triangulate(polygon)]
    faces += [(a+n,b+n,c+n) for a,b,c in triangulate(polygon)]
    for i in range(n):
        j=(i+1)%n
        faces += [(i,j,j+n),(i,j+n,i+n)]
    collada(path,verts,faces)


def analyze_assets():
    reference = np.array(Image.open(ROOT/'docs/design/approved-face-v2.png').convert('RGB'))
    h,w = reference.shape[:2]
    roi = reference[int(h*.57):int(h*.67),int(w*.34):int(w*.66)].astype(float)
    red = (roi[:,:,0] > roi[:,:,1]*1.55) & (roi[:,:,0] > roi[:,:,2]*1.45)
    yy,xx = np.where(red)
    if len(xx) < 100:
        raise ValueError('Cannot locate approved mouth in reference')
    xmin,xmax = xx.min()+int(w*.34), xx.max()+int(w*.34)
    ymin,ymax = yy.min()+int(h*.57), yy.max()+int(h*.57)
    # Anchor scaling at the existing mouth goal, preserving task coordinates.
    target = [REFERENCE_BOARD_CENTER[0]+(0.5-(ymin+ymax)/2/h)*REFERENCE_BOARD_LENGTH,
              REFERENCE_BOARD_CENTER[1]+(0.5-(xmin+xmax)/2/w)*REFERENCE_BOARD_WIDTH]
    width=(xmax-xmin+1)/w*BOARD_WIDTH
    depth=(ymax-ymin+1)/h*BOARD_LENGTH
    alpha=np.array(Image.open(ROOT/'models/fukuwarai_mouth/materials/textures/lips-v2.png').getchannel('A'))
    yy,xx=np.where(alpha>128)
    xa,xb,ya,yb=int(xx.min()),int(xx.max()),int(yy.min()),int(yy.max())
    cols=sorted(set(round(v) for v in np.linspace(xa,xb,65)))
    upper,lower=[],[]
    for x in cols:
        ys=np.where(alpha[:,x]>128)[0]
        if len(ys):
            upper.append((x,int(ys[0])))
            lower.append((x,int(ys[-1])))
    pixels=upper+list(reversed(lower))
    pixels=list(dict.fromkeys(pixels))
    points=[(-((y-(ya+yb)/2)/(yb-ya))*depth,
             -((x-(xa+xb)/2)/(xb-xa))*width) for x,y in pixels]
    if sum(points[i][0]*points[(i+1)%len(points)][1]-points[(i+1)%len(points)][0]*points[i][1]
           for i in range(len(points))) < 0:
        points.reverse()
        pixels.reverse()
    uv=[(x/(alpha.shape[1]-1),1-y/(alpha.shape[0]-1)) for x,y in pixels]
    return target,width,depth,points,uv


def choose_start(settings, seed=42):
    rng=random.Random(seed)
    tx,ty=settings['target_xy_m']
    for _ in range(1000):
        x=rng.uniform(*settings['random_start']['x_bounds_m'])
        y=rng.uniform(*settings['random_start']['y_bounds_m'])
        distance=math.hypot(x-tx,y-ty)
        if 0.03 <= distance <= 0.06:
            return [x,y,SURFACE_Z+THICKNESS/2+0.002]
    raise ValueError('No valid random start sampled')


def main():
    target,width,depth,polygon,uv=analyze_assets()
    board_center = [target[i] + GEOMETRY_SCALE * (REFERENCE_BOARD_CENTER[i] - target[i])
                    for i in range(2)] + [REFERENCE_BOARD_CENTER[2]]
    settings={'world':'default','task':'mouth_only','table_surface_z_m':1.015,
              'board_surface_z_m':SURFACE_Z,'board_center_m':board_center,
              'geometry_scale':GEOMETRY_SCALE,
              'board_size_m':[BOARD_LENGTH,BOARD_WIDTH,0.004],
              'target_xy_m':target,'position_tolerance_m':0.015,'hold_time_sim_s':1.0,
              'orientation_scored':False,'mouth_size_m':[depth,width,THICKNESS],
              'mouth_mass_kg':0.012,
              'random_start':{'x_bounds_m':[0.16,0.25],'y_bounds_m':[-0.055,0.055],
                              'distance_bounds_m':[0.03,0.06],'yaw_rad':0.0,
                              'default_seed':42,'reachability_verified':False}}
    settings['start_m']=choose_start(settings)
    mouth_dir=ROOT/'models/fukuwarai_mouth/meshes'
    mouth_dir.mkdir(parents=True,exist_ok=True)
    extruded_mesh(mouth_dir/'lips_body.dae',polygon,THICKNESS)
    extruded_mesh(mouth_dir/'lips_collision.dae',hull(polygon),THICKNESS)
    collada(mouth_dir/'lips_top.dae',[(x,y,THICKNESS/2+0.00003) for x,y in polygon],
            triangulate(polygon),uv)
    board_dir=ROOT/'models/fukuwarai_board/meshes'
    collada(board_dir/'face_top.dae',
            [(BOARD_LENGTH/2,BOARD_WIDTH/2,0.00203),(-BOARD_LENGTH/2,BOARD_WIDTH/2,0.00203),
             (-BOARD_LENGTH/2,-BOARD_WIDTH/2,0.00203),(BOARD_LENGTH/2,-BOARD_WIDTH/2,0.00203)],
            [(0,1,2),(0,2,3)],[(0,1),(0,0),(1,0),(1,1)])
    board,link=model('fukuwarai_board',True,board_center)
    shape(link,'collision',(0,0,0),('box',(BOARD_LENGTH,BOARD_WIDTH,0.004)),collision=True)
    shape(link,'board',(0,0,0),('box',(BOARD_LENGTH,BOARD_WIDTH,0.004)),'0.97 0.95 0.90 1')
    visual=shape(link,'face',(0,0,0),('mesh','model://fukuwarai_board/meshes/face_top.dae'))
    texture(visual,'model://fukuwarai_board/materials/textures/face-v2.png')
    mouth,link=model('fukuwarai_mouth',False,settings['start_m'])
    element(element(link,'inertial'),'mass',settings['mouth_mass_kg'])
    inertial=link.find('inertial')
    tensor=element(inertial,'inertia')
    mass=settings['mouth_mass_kg']
    for name,value in {'ixx':mass*(width**2+THICKNESS**2)/12,
                       'iyy':mass*(depth**2+THICKNESS**2)/12,
                       'izz':mass*(width**2+depth**2)/12,'ixy':0,'ixz':0,'iyz':0}.items():
        element(tensor,name,value)
    shape(link,'collision',(0,0,0),('mesh','model://fukuwarai_mouth/meshes/lips_collision.dae'),collision=True)
    shape(link,'lip_body',(0,0,0),('mesh','model://fukuwarai_mouth/meshes/lips_body.dae'))
    body_material=element(link.find("visual[@name='lip_body']"),'material')
    element(body_material,'ambient','0.95 0.32 0.31 1')
    element(body_material,'diffuse','0.95 0.32 0.31 1')
    visual=shape(link,'lip_print',(0,0,0),('mesh','model://fukuwarai_mouth/meshes/lips_top.dae'))
    texture(visual,'model://fukuwarai_mouth/materials/textures/lips-v2.png')
    models=[board,mouth]
    for m in models:
        sdf=ET.Element('sdf',version='1.7'); sdf.append(m)
        write_xml(ROOT/'models'/m.attrib['name']/'model.sdf',sdf)
        cfg=ET.Element('model')
        element(cfg,'name',m.attrib['name']);element(cfg,'version','2.0')
        element(cfg,'sdf','model.sdf',version='1.7')
        element(cfg,'description','Original mouth-only Fukuwarai asset.')
        write_xml(ROOT/'models'/m.attrib['name']/'model.config',cfg)
    sdf=ET.Element('sdf',version='1.7');world=element(sdf,'world',name='default')
    element(world,'gravity','0 0 -9.81')
    physics=element(world,'physics',name='default_physics',type='ignored')
    element(physics,'max_step_size',0.001);element(physics,'real_time_factor',1)
    for suffix,filename in [('Physics','physics'),('UserCommands','user-commands'),('SceneBroadcaster','scene-broadcaster')]:
        element(world,'plugin',filename=f'ignition-gazebo-{filename}-system',name=f'ignition::gazebo::systems::{suffix}')
    sensors=element(world,'plugin',filename='ignition-gazebo-sensors-system',
                    name='ignition::gazebo::systems::Sensors')
    element(sensors,'render_engine','ogre2')
    scene=element(world,'scene');element(scene,'ambient','0.8 0.8 0.8 1');element(scene,'background','0.90 0.93 0.96 1')
    sun=element(world,'light',name='sun',type='directional');element(sun,'pose','0 0 5 0 0 0')
    element(sun,'diffuse','0.7 0.7 0.7 1');element(sun,'direction','-0.3 0.2 -1')
    ground,link=model('ground',True,(0,0,-0.025))
    shape(link,'collision',(0,0,0),('box',(8,8,0.05)),collision=True)
    shape(link,'visual',(0,0,0),('box',(8,8,0.05)),'0.36 0.43 0.48 1');world.append(ground)
    table,link=model('fukuwarai_table',True,(0.29,0,0))
    shape(link,'collision',(0,0,1),('box',(0.8,1.5,0.03)),collision=True)
    shape(link,'top',(0,0,1),('box',(0.8,1.5,0.03)),'0.75 0.81 0.84 1')
    for i,(x,y) in enumerate([(-0.36,-0.70),(-0.36,0.70),(0.36,-0.70),(0.36,0.70)]):
        shape(link,f'leg_{i}',(x,y,0.5),('cylinder',(0.022,1)),'0.28 0.34 0.38 1')
    world.append(table)
    for m in models: world.append(m)
    # Fixed observation camera only: no collision or force acts on the task.
    camera_model,camera_link=model('fukuwarai_observation_camera',True,(0.24,0,1.75))
    sensor=element(camera_link,'sensor',name='overhead',type='camera')
    element(sensor,'pose','0 0 0 0 1.5707963267948966 0')
    element(sensor,'always_on','true'); element(sensor,'update_rate',10)
    element(sensor,'topic','/fukuwarai/overhead/image')
    camera=element(sensor,'camera')
    element(camera,'horizontal_fov',0.85)
    image=element(camera,'image'); element(image,'width',640); element(image,'height',480)
    element(image,'format','R8G8B8')
    clip=element(camera,'clip'); element(clip,'near',0.02); element(clip,'far',3)
    world.append(camera_model)
    settings['observation_camera']={'topic':'/fukuwarai/overhead/image','width':640,
                                    'height':480,'fps':10,'pose_m_rad':[0.24,0,1.75,0,math.pi/2,0],
                                    'horizontal_fov_rad':0.85}
    write_xml(ROOT/'worlds/fukuwarai.sdf',sdf)
    (ROOT/'config/fukuwarai.json').write_text(json.dumps(settings,indent=2)+'\n')
    print(json.dumps({'target_xy_m':target,'mouth_size_m':settings['mouth_size_m'],'start_m':settings['start_m']}))


if __name__=='__main__':
    main()
