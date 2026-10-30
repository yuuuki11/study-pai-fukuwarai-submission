"""URDF kinematics and collision-mesh bounds for the simulator expert."""
from pathlib import Path
import math
import struct
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ARM = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint4', 'crane_plus_joint3']
HAND = 'crane_plus_joint_hand'


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    t[:3, 3] = xyz
    return t


def points(t, vertices):
    return vertices @ t[:3, :3].T + t[:3, 3]


class CraneKinematics:
    def __init__(self, urdf, package_path, base_z=1.015, pitch=2.2):
        self.root = ET.fromstring(urdf)
        self.base_z, self.pitch = base_z, pitch
        self.joints = self.root.findall('joint')
        self.bounds = {}
        for joint in self.joints:
            limit = joint.find('limit')
            if limit is not None and 'lower' in limit.attrib:
                self.bounds[joint.attrib['name']] = (float(limit.get('lower')), float(limit.get('upper')))
        self.colliders = {}
        for link in self.root.findall('link'):
            mesh = link.find('collision/geometry/mesh')
            if mesh is None:
                continue
            name = mesh.get('filename').replace('package://crane_plus_description/', '')
            raw = (Path(package_path) / name).read_bytes()
            count = struct.unpack_from('<I', raw, 80)[0]
            if len(raw) != 84 + count * 50:
                raise ValueError('Expected binary STL: ' + name)
            dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
            vertices = np.unique(np.frombuffer(raw, dtype, offset=84)['vertices'].reshape(-1, 3), axis=0)
            scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            self.colliders[link.get('name')] = points(self.origin(link.find('collision/origin')), vertices * scale)
        # A point on the lowest closed-gripper surface at the planned pitch.
        wrist = self.colliders['crane_plus_link4']
        hand = points(transform((0, -.0145, .045)) @ transform(rpy=(-.1745, 0, 0)),
                      self.colliders['crane_plus_link_hand'])
        distal = np.concatenate((wrist, hand))
        rotated = points(transform(rpy=(0, pitch, 0)), distal)
        mask = rotated[:, 2] < rotated[:, 2].min() + .0003
        self.tcp = distal[mask].mean(axis=0)

    @staticmethod
    def origin(element):
        if element is None:
            return np.eye(4)
        return transform(np.fromstring(element.get('xyz', '0 0 0'), sep=' '),
                         np.fromstring(element.get('rpy', '0 0 0'), sep=' '))

    def frames(self, q, hand=.1745):
        values = dict(zip(ARM, q)) | {HAND: hand}
        frames = {'world': transform((0, 0, self.base_z))}
        for joint in self.joints:
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            axis = joint.find('axis')
            rotation = np.eye(4)
            if axis is not None:
                rotation[:3, :3] = Rotation.from_rotvec(np.fromstring(axis.get('xyz'), sep=' ') * values[joint.get('name')]).as_matrix()
            frames[child] = frames[parent] @ self.origin(joint.find('origin')) @ rotation
        return frames

    def tool(self, q):
        return points(self.frames(q)['crane_plus_link4'], self.tcp[None])[0]

    def solve(self, xyz, seed=None):
        lo, hi = np.array([self.bounds[n] for n in ARM]).T
        def residual(q):
            return np.r_[self.tool(q) - xyz, .05 * (q[1] - q[3] - q[2] - self.pitch)]
        seeds = [seed] if seed is not None else []
        yaw = math.atan2(xyz[1], xyz[0])
        seeds += [[yaw, 1.0, -.7, -0.5], [yaw, 1.5, -.5, -.2], [yaw, .4, -.8, -1.0]]
        best = None
        for s in seeds:
            result = least_squares(residual, np.clip(s, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), max_nfev=80)
            err = np.linalg.norm(residual(result.x))
            if best is None or err < best[0]:
                best = (err, result.x)
            if err < .00001:
                return result.x
        if best[0] > .0005:
            raise ValueError(f'Unreachable tool point {xyz}, residual {best[0]:.6f}')
        return best[1]

    def minimum_z(self, q):
        frames = self.frames(q)
        return {name: float(points(frames[name], vertices)[:, 2].min())
                for name, vertices in self.colliders.items() if name not in ('crane_plus_base', 'crane_plus_link1')}
