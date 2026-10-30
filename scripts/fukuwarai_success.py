"""Versioned scorer only: simulator poses must never be policy inputs."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/fukuwarai_success_v2.json'


def load_rules():
    rules = json.loads(CONFIG.read_text())
    mesh = ROOT / 'models/fukuwarai_mouth/meshes/lips_top.dae'
    rules['config_sha256'] = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    rules['contour_sha256'] = hashlib.sha256(mesh.read_bytes()).hexdigest()
    return rules


def load_contour():
    root = ET.parse(ROOT / 'models/fukuwarai_mouth/meshes/lips_top.dae').getroot()
    source = next(x for x in root.iter() if x.tag.endswith('source') and x.get('id') == 'positions')
    array = next(x for x in source if x.tag.endswith('float_array'))
    return np.fromstring(array.text, sep=' ').reshape(-1, 3)


def sample_edges(poly, spacing):
    return np.concatenate([a + np.arange(max(1, math.ceil(np.linalg.norm(b-a)/spacing)))[:, None]
                           / max(1, math.ceil(np.linalg.norm(b-a)/spacing)) * (b-a)
                           for a, b in zip(poly, np.roll(poly, -1, axis=0))])


def outside_distance(points, polygon):
    """Exact segment distance at each sample; inside points have zero distance."""
    a = polygon; b = np.roll(a, -1, axis=0); edge = b-a
    delta = points[:, None, :] - a[None, :, :]
    t = np.clip(np.sum(delta*edge, axis=2)/np.sum(edge*edge, axis=1), 0, 1)
    distance = np.linalg.norm(delta-t[:, :, None]*edge, axis=2).min(axis=1)
    x, y = points[:, 0, None], points[:, 1, None]
    crossing = ((a[:, 1] > y) != (b[:, 1] > y))
    denominator = b[:, 1]-a[:, 1]
    safe = np.where(abs(denominator) > 1e-15, denominator, 1.)
    inside = np.logical_xor.reduce(crossing & (x < (b[:, 0]-a[:, 0])*(y-a[:, 1])/safe+a[:, 0]), axis=1)
    return np.where(inside, 0., distance)


class MouthScorer:
    def __init__(self, scene, rules=None):
        self.scene = scene
        self.rules = rules or load_rules()
        self.vertices = load_contour()
        self.contour = self.vertices[:, :2]
        self.samples = sample_edges(self.contour, self.rules['contour_sample_spacing_m'])

    def metrics(self, xyz, quaternion_xyzw, tool_z):
        q = np.asarray(quaternion_xyzw, dtype=float)
        xyz = np.asarray(xyz, dtype=float)
        if q.shape != (4,) or not np.all(np.isfinite(np.r_[q, xyz, tool_z])) or np.linalg.norm(q) < 1e-8:
            raise ValueError('Missing or invalid scoring telemetry')
        x,y,z,w = q / np.linalg.norm(q)
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        yaw = abs(math.degrees(math.atan2(rotation[1,0],rotation[0,0])))
        tilt = math.degrees(math.acos(float(np.clip(rotation[2,2],-1,1))))
        offset = xyz[:2]-self.scene['target_xy_m']
        actual_poly = (self.vertices @ rotation.T)[:, :2] + offset
        top_offset = self.vertices[0,2] * rotation[:2,2]
        actual_samples = self.samples @ rotation[:2,:2].T + top_offset + offset
        # Bidirectional containment avoids accepting a collapsed/tilted projection.
        # Add half spacing as a conservative upper bound between sampled points.
        if np.linalg.norm(offset) < 1e-12 and np.max(abs(rotation-np.eye(3))) < 1e-12:
            outside = 0.
        else:
            outside = max(float(outside_distance(actual_samples,self.contour).max()),
                          float(outside_distance(self.samples,actual_poly).max())) + self.rules['contour_sample_spacing_m']/2
        height_error = abs(float(xyz[2])-(self.scene['board_surface_z_m']+self.scene['mouth_size_m'][2]/2))
        values = {'center_error_m':float(np.linalg.norm(offset)), 'yaw_error_deg':yaw,
                  'tilt_error_deg':tilt, 'outline_error_upper_bound_m':outside,
                  'height_error_m':height_error, 'tool_z_m':float(tool_z)}
        checks = {'center':values['center_error_m'] <= self.rules['center_tolerance_m'],
                  'yaw':yaw <= self.rules['yaw_tolerance_deg'],
                  'tilt':tilt <= self.rules['tilt_tolerance_deg'],
                  'outline':outside <= self.rules['outline_tolerance_m'],
                  'on_board':height_error <= self.rules['height_tolerance_m'],
                  'released':tool_z >= self.scene['board_surface_z_m']+self.rules['release_height_m']}
        return dict(values, checks=checks, placement_pass=all(v for k,v in checks.items() if k!='released'),
                    instantaneous_pass=all(checks.values()))


class HoldScorer:
    def __init__(self, scorer):
        self.scorer = scorer
        self.since = None
        self.previous = None

    def update(self, sim_s, xyz, quaternion_xyzw, tool_z):
        m = self.scorer.metrics(xyz, quaternion_xyzw, tool_z)
        if self.previous is not None and (sim_s < self.previous or sim_s-self.previous > self.scorer.rules['max_telemetry_gap_s']):
            self.since = None
        self.previous = sim_s
        if m['instantaneous_pass']:
            if self.since is None:
                self.since = sim_s
        else:
            self.since = None
        held = max(0., sim_s-self.since) if self.since is not None else 0.
        return dict(m, held_sim_s=held, success=held >= self.scorer.rules['hold_time_sim_s'])
