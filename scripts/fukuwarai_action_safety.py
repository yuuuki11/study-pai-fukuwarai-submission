"""Optional board-clearance projection of learned actions, without object poses."""
import numpy as np


def path_clear(kine, current, command, board_z):
    return all(min(kine.minimum_z(current[:4] + alpha * (command[:4]-current[:4])).values())
               >= board_z + .0005 for alpha in np.linspace(0, 1, 5))


def project_floor(kine, current, command, board_z):
    """Keep the policy's requested TCP xy; raise unsafe targets by at most 4cm.

    The authored gripper pitch is recovered by IK. Joint corrections are
    limited to .25rad, then the original .1rad/tick speed limit is reapplied.
    The same intermediate-pose collision check remains mandatory.
    """
    if path_clear(kine, current, command, board_z):
        return command.copy(), None
    xyz = kine.tool(command[:4])
    minimum = min(kine.minimum_z(command[:4]).values())
    lift = max(.001, board_z + .003 - minimum)
    for extra in [0., .003, .010, .020]:
        if lift + extra > .040:
            continue
        target = xyz.copy()
        target[2] += lift + extra
        try:
            corrected = kine.solve(target, command[:4])
        except ValueError:
            continue
        if np.max(np.abs(corrected-command[:4])) > .25:
            continue
        candidate = command.copy()
        candidate[:4] = current[:4] + np.clip(corrected-current[:4], -.1, .1)
        if path_clear(kine, current, candidate, board_z):
            return candidate, {'requested_tool_xyz': xyz.tolist(),
                               'raised_tool_target_xyz': target.tolist(),
                               'requested_minimum_z': float(minimum),
                               'lift_m': float(lift + extra),
                               'executed_tool_xy_shift_m': float(np.linalg.norm(kine.tool(candidate[:4])[:2]-xyz[:2])),
                               'joint_correction_rad': (candidate[:4]-command[:4]).tolist()}
    raise RuntimeError('Safety stop: learned action could not be projected above board')
