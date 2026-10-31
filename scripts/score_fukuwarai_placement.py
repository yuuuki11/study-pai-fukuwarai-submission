"""Read-only supplementary placement scoring; never replaces strict-v2 results."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from fukuwarai_tool_cues import tool_xyz_course

ROOT = Path(__file__).resolve().parents[1]


def update_hold(t, passed, since, previous, gap):
    if previous is not None and (t < previous or t - previous > gap):
        since = None
    since = (t if since is None else since) if passed else None
    return since, max(0., t - since) if since is not None else 0.


def score_episode(directory, rules, scene):
    since = previous = first = final = None
    maximum = 0.
    count = 0
    observations = directory / 'observations.jsonl'
    for line in observations.open():
        row = json.loads(line)
        t = float(row['sim_s'])
        q = np.asarray(row['mouth_quaternion_xyzw'], dtype=float)
        xyz = np.asarray(row['mouth_xyz_m'], dtype=float)
        if not np.isfinite(np.r_[t, q, xyz]).all() or np.linalg.norm(q) < 1e-8:
            raise ValueError('Invalid pose telemetry')
        x,y,z,w = q / np.linalg.norm(q)
        yaw = abs(math.degrees(math.atan2(2*(x*y+z*w),1-2*(y*y+z*z))))
        tilt = math.degrees(math.acos(float(np.clip(1-2*(x*x+y*y),-1,1))))
        center = float(np.linalg.norm(xyz[:2] - scene['target_xy_m']))
        height = abs(float(xyz[2]) - scene['board_surface_z_m'] - scene['mouth_size_m'][2]/2)
        names = dict(zip(row['joint_names'], row['joint_positions']))
        state = [names[n] for n in ('crane_plus_joint1','crane_plus_joint2','crane_plus_joint4','crane_plus_joint3','crane_plus_joint_hand')]
        tool_z = float(tool_xyz_course(state)[2])
        checks = dict(center=center*1000<=rules['center_mm'], yaw=yaw<=rules['yaw_deg'],
                      tilt=tilt<=rules['tilt_deg'], on_board=height*1000<=rules['height_mm'],
                      released=tool_z>=scene['board_surface_z_m']+rules['release_mm']/1000)
        since, held = update_hold(t, all(checks.values()), since, previous, rules['max_gap_sim_s'])
        previous = t
        maximum = max(maximum, held)
        final = dict(sim_s=t, center_mm=center*1000, yaw_deg=yaw, checks=checks)
        if held >= rules['hold_sim_s'] and first is None:
            first = dict(**final, held_sim_s=held)
        count += 1
    result = json.loads((directory/'result.json').read_text())
    return dict(seed=result['seed'], samples=count, strict_v2_success=bool(result['success']),
                placement_success=first is not None, maximum_hold_sim_s=maximum,
                first_pass=first, final=final, reason=result.get('reason'),
                observations_sha256=hashlib.sha256(observations.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--rules',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    rules = json.loads(a.rules.read_text())
    if rules.get('outline_gate') is not False:
        raise ValueError('This supplemental scorer explicitly excludes outline')
    scene = json.loads((ROOT/'config/fukuwarai.json').read_text())
    episodes = [score_episode(f.parent,rules,scene) for f in sorted(a.input.glob('*/result.json'))]
    report = dict(rules=rules, rules_sha256=hashlib.sha256(a.rules.read_bytes()).hexdigest(),
                  completed=len(episodes), successes=sum(e['placement_success'] for e in episodes),
                  strict_v2_successes=sum(e['strict_v2_success'] for e in episodes), episodes=episodes,
                  note='Supplementary scoring at saved observation times. Raw strict-v2 results preserved. All outcomes retained.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(completed=report['completed'],successes=report['successes'])))


if __name__ == '__main__':
    main()
