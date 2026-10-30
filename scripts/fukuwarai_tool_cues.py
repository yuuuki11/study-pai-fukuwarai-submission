"""Static robot FK and image-relative tool cues from observed joints and RGB.

No simulator telemetry, target actions, or future observations are accepted.
This module is opt-in; existing checkpoint preprocessing remains unchanged.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def static_geometry():
    robot = json.loads((ROOT / 'config/fukuwarai_tool_kinematics.json').read_text())
    scene = json.loads((ROOT / 'config/fukuwarai.json').read_text())
    camera = scene['observation_camera']
    if not np.allclose(camera['pose_m_rad'][3:], [0, np.pi / 2, 0], atol=1e-8):
        raise ValueError('Tool cues require the fixed overhead camera orientation')
    return robot, scene


def tool_xyz_course(state):
    """FK of the same closed-gripper TCP used by the existing robot helper."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (5,) or not np.isfinite(state).all():
        raise ValueError('Expected five finite observed joints in course order')
    robot, _ = static_geometry()
    raw = state[[0, 1, 3, 2]]
    values = dict(zip(robot['arm_raw_order'], raw))
    # Existing TCP is defined on link4 with the fixed closed-gripper geometry.
    values['crane_plus_joint_hand'] = float(state[4])
    base = np.eye(4)
    base[2, 3] = robot['base_z_m']
    frames = {'world': base}
    for joint in robot['joints']:
        rotation = np.eye(4)
        if joint['axis'] is not None:
            rotation[:3, :3] = Rotation.from_rotvec(
                np.asarray(joint['axis']) * values[joint['name']]).as_matrix()
        frames[joint['child']] = frames[joint['parent']] @ np.asarray(joint['origin']) @ rotation
    tcp = np.r_[robot['tcp_link4_m'], 1.]
    return (frames['crane_plus_link4'] @ tcp)[:3]


def tool_relative_cues(state, mouth_cues):
    """Return tool-minus-mouth image-plane xy and tool height above the board.

    Mouth center is the current RGB cue, not a simulator position. Both xy
    coordinates use the known mouth-top plane; tool height is a separate cue.
    Missing mouths have zero relative xy. RGB visibility remains in the token.
    """
    cue = np.asarray(mouth_cues)
    if cue.shape != (5,) or not np.isfinite(cue).all():
        raise ValueError('Expected five finite current RGB mouth cues')
    xyz = tool_xyz_course(state)
    _, scene = static_geometry()
    camera = scene['observation_camera']
    cx, cy, cz = camera['pose_m_rad'][:3]
    width, height = camera['width'], camera['height']
    plane = scene['board_surface_z_m'] + scene['mouth_size_m'][2]
    scale = width / (2 * np.tan(camera['horizontal_fov_rad'] / 2)) / (cz - plane)
    tool_pixel = np.array([(width - 1) / 2 - (xyz[1] - cy) * scale,
                           (height - 1) / 2 - (xyz[0] - cx) * scale])
    mouth_pixel = cue[:2] * 100 + [320., 270.]
    relative = (tool_pixel - mouth_pixel) / 100 if cue[4] > 0 else np.zeros(2)
    return np.r_[relative, (xyz[2] - scene['board_surface_z_m']) / .1].clip(-5, 5).astype(np.float32)
