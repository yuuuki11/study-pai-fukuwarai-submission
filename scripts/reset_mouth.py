#!/usr/bin/env python3
"""Re-create just the mouth at a reproducible random position; the arm is unchanged."""
import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def sample_start(config, seed):
    rng = random.Random(seed)
    region = config['random_start']
    for _ in range(1000):
        x = rng.uniform(*region['x_bounds_m'])
        y = rng.uniform(*region['y_bounds_m'])
        distance = math.dist((x,y),config['target_xy_m'])
        if region['distance_bounds_m'][0] <= distance <= region['distance_bounds_m'][1]:
            return [x,y,config['board_surface_z_m']+config['mouth_size_m'][2]/2+0.002]
    raise ValueError('No valid initial position')


def service(world, suffix, request_type, request):
    result = subprocess.run(['ign','service','-s',f'/world/{world}/{suffix}',
                             '--reqtype',request_type,'--reptype','ignition.msgs.Boolean',
                             # World entity recreation can briefly lag after a
                             # long rosbag write; allow the existing Gazebo
                             # service more than the former 3 s to respond.
                             '--timeout','8000','--req',request],capture_output=True,text=True)
    if result.returncode or 'data:' not in result.stdout:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'Gazebo service timed out')
    return 'data: true' in result.stdout


def current_mouth_position(timeout=1.5):
    """Read the existing model pose without using it to control the arm."""
    try:
        result = subprocess.run(
            ['ign', 'topic', '-e', '--json-output', '-t',
             '/world/default/dynamic_pose/info'],
            capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout or ''
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ''
    if isinstance(output, bytes):
        output = output.decode(errors='replace')
    latest = None
    for line in output.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        for pose in message.get('pose', []):
            if pose.get('name') == 'fukuwarai_mouth':
                position = pose.get('position', {})
                latest = [float(position.get(axis, 0.0)) for axis in ('x', 'y', 'z')]
    return latest


def at_position(actual, expected, tolerance=0.001):
    return actual is not None and math.dist(actual, expected) <= tolerance


def remove_existing_mouth(world):
    request = 'name: "fukuwarai_mouth" type: MODEL'
    for attempt in range(3):
        try:
            # A false response means the entity was already absent.
            service(world, 'remove', 'ignition.msgs.Entity', request)
            return
        except RuntimeError:
            # Gazebo can complete a removal but lose its service reply. Check
            # the read-only pose stream before deciding whether to retry.
            if current_mouth_position() is None:
                return
            if attempt == 2:
                raise
            time.sleep(0.25)


def create_mouth(world, sdf_xml, position):
    request = 'sdf: ' + json.dumps(sdf_xml)
    for attempt in range(3):
        try:
            if service(world, 'create', 'ignition.msgs.EntityFactory', request):
                return
        except RuntimeError:
            pass

        # The create request may have succeeded even when its response timed
        # out. Confirm by the requested seeded pose before retrying. If another
        # pose remains, remove that same-name entity before trying again.
        actual = current_mouth_position()
        if at_position(actual, position):
            return
        if actual is not None:
            remove_existing_mouth(world)
        if attempt < 2:
            time.sleep(0.25)
    raise RuntimeError('Failed to re-create mouth at the requested seeded position')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,help='Same seed reproduces the same position.')
    parser.add_argument('--at-goal',action='store_true',help='Visual alignment check only, not a scored trial.')
    args = parser.parse_args()
    config = json.loads((ROOT/'config/fukuwarai.json').read_text())
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    position = sample_start(config,seed)
    if args.at_goal:
        position[:2] = config['target_xy_m']
    sdf = ET.parse(ROOT/'models/fukuwarai_mouth/model.sdf')
    sdf.find('model/pose').text = ' '.join(map(str,position))+' 0 0 0'
    # Re-creation clears linear and angular velocity from the previous trial.
    remove_existing_mouth(config['world'])
    create_mouth(config['world'], ET.tostring(sdf.getroot(), encoding='unicode'), position)
    print(json.dumps({'seed':seed,'position_m':position,'yaw_rad':0,
                      'mode':'visual_goal_check' if args.at_goal else 'random_start'}))


if __name__=='__main__':
    main()
