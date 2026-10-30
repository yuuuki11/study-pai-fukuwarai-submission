#!/usr/bin/env python3
"""Identify teacher-only return-to-upright intervals without modifying raw data.
Run in the ROS container. Exact command timestamps, not image heuristics, define
loss masks; retained action chunks keep their original 10Hz temporal spacing.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
from rclpy.serialization import deserialize_message
from trajectory_msgs.msg import JointTrajectory

ROOT = Path(__file__).resolve().parents[1]
TOPIC = '/crane_plus_arm_controller/joint_trajectory'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, default=ROOT/'data/lerobot_v2.1/crane_plus_fukuwarai')
    ap.add_argument('--data-root',type=Path,default=None)
    ap.add_argument('--output', type=Path, default=ROOT/'data/training/return_home_mask.json')
    args = ap.parse_args()
    manifest = json.loads((args.dataset/'conversion_manifest.json').read_text())
    data_root = args.data_root or ROOT / manifest.get('source_data_root','data/expert_images')
    sources = json.loads((args.dataset/'episode_sources.json').read_text())
    report = {'method': 'mask all-zero arm command until the next nonzero arm command; never concatenate across gaps',
              'source_dataset': str(args.dataset), 'source_data_root': str(data_root.relative_to(ROOT)) if data_root.is_relative_to(ROOT) else str(data_root), 'raw_data_modified': False, 'episodes': [],
              'total_frames': 0, 'excluded_frames': 0, 'moving_return_frames': 0, 'upright_stationary_frames': 0}
    for ep, src in enumerate(sources):
        folder = ROOT/src['source_repo_relative'] if src.get('source_repo_relative') else data_root/src['source_episode']
        bag_folder = ROOT/src['derived_segment']['raw_source_relative_to_repo'] if src.get('derived_segment') else folder
        raw = [json.loads(l) for l in (folder/'frames.jsonl').read_text().splitlines()]
        commands = []
        for db in sorted((bag_folder/'rosbag').glob('*.db3')):
            conn = sqlite3.connect('file:'+str(db)+'?mode=ro', uri=True)
            rows = conn.execute('SELECT messages.timestamp,messages.data FROM messages JOIN topics ON messages.topic_id=topics.id WHERE topics.name=? ORDER BY messages.timestamp', (TOPIC,))
            for ns, data in rows:
                msg = deserialize_message(bytes(data), JointTrajectory)
                if not msg.points:
                    continue
                is_home = all(abs(x) < 1e-7 for point in msg.points for x in point.positions)
                commands.append((ns, is_home))
            conn.close()
        intervals = []
        for i, (ns, home) in enumerate(commands):
            if not home:
                continue
            end = next((t for t, flag in commands[i+1:] if not flag), int(raw[-1]['sim_time_ns'])+1)
            intervals.append([ns, end])
        excluded, moving, static = [], 0, 0
        for i, row in enumerate(raw):
            ns = int(row['sim_time_ns'])
            if any(a <= ns < b for a,b in intervals):
                excluded.append(i)
                q = np.array(row['observation.state'][:4])
                previous = np.array(raw[max(0,i-1)]['observation.state'][:4])
                if np.max(np.abs(q)) < .03 and np.max(np.abs(q-previous)) < .005:
                    static += 1
                else:
                    moving += 1
        report['episodes'].append({'episode_index': ep, 'source_episode': src['source_episode'], 'source_repo_relative':src.get('source_repo_relative'), 'derived_segment':src.get('derived_segment'),
             'frames': len(raw), 'excluded_frames': excluded, 'home_intervals_sim_ns': intervals,
             'excluded_count': len(excluded), 'moving_return_frames': moving, 'upright_stationary_frames': static})
        report['total_frames'] += len(raw)
        report['excluded_frames'] += len(excluded)
        report['moving_return_frames'] += moving
        report['upright_stationary_frames'] += static
    report['excluded_ratio'] = report['excluded_frames']/report['total_frames']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'}))


if __name__ == '__main__':
    main()
