#!/usr/bin/env python3
"""Also mask the initial all-zero command wait, without changing episode time."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-mask', type=Path, default=ROOT/'data/training/return_home_mask.json')
    ap.add_argument('--restore-moving-returns',action='store_true')
    ap.add_argument('--keep-verified-hold',action='store_true')
    ap.add_argument('--data-root',type=Path,default=None)
    ap.add_argument('--output', type=Path, default=ROOT/'data/training/return_and_initial_wait_mask.json')
    args = ap.parse_args()
    report = json.loads(args.base_mask.read_text())
    data_root = args.data_root or ROOT / report.get('source_data_root','data/expert_images')
    added = 0
    restored_moving = restored_hold = 0
    new_moving = new_static = 0
    for ep in report['episodes']:
        folder = ROOT/ep['source_repo_relative'] if ep.get('source_repo_relative') else data_root/ep['source_episode']
        rows = [json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
        assert len(rows) == ep['frames']
        first = next((i for i, row in enumerate(rows)
                      if max(abs(v) for v in row['action'][:4]) > .001), None)
        if first is None:
            raise RuntimeError('No nonzero arm command in ' + str(folder))
        previous = set(ep['excluded_frames'])
        excluded = previous | set(range(first))
        moving = {i for i in excluded if max(abs(v) for v in rows[i]['observation.state'][:4]) >= .03
                  or max(abs(a-b) for a,b in zip(rows[i]['observation.state'][:4],rows[max(0,i-1)]['observation.state'][:4])) >= .005}
        if args.restore_moving_returns:
            restored_moving += len(excluded & moving)
            excluded -= moving
        if args.keep_verified_hold:
            raw_folder = ROOT/ep['derived_segment']['raw_source_relative_to_repo'] if ep.get('derived_segment') else folder
            result=json.loads((raw_folder/'result.json').read_text())
            if result.get('success_rules',{}).get('version') != 'mouth_outline_v2' or not result.get('physical_success'):
                raise RuntimeError('Only verified v2 episodes can teach terminal hold')
            terminal={i for i,r in enumerate(rows) if r['phase']=='verify_released'}
            terminal.update([i for i,r in enumerate(rows) if r['phase']=='post_success_recording'][:10])
            restored_hold += len(excluded & terminal)
            excluded -= terminal
        ep['excluded_frames'] = sorted(excluded)
        ep['moving_return_frames'] = len(excluded & moving)
        ep['upright_stationary_frames'] = len(excluded-moving)
        new_moving += ep['moving_return_frames']
        new_static += ep['upright_stationary_frames']
        ep['initial_wait_prefix_frames'] = first
        ep['excluded_count'] = len(ep['excluded_frames'])
        added += len(set(range(first)) - previous)
    report['initial_wait_frames_added'] = added
    report['moving_return_frames'] = new_moving
    report['upright_stationary_frames'] = new_static
    report['restored_moving_return_frames'] = restored_moving
    report['restored_verified_hold_frames'] = restored_hold
    if args.restore_moving_returns:report['method'] += '; restore moving return trajectories for state continuity'
    if args.keep_verified_hold:report['method'] += '; retain verified v2 release hold and first 10 post-success frames'
    report['excluded_frames'] = sum(ep['excluded_count'] for ep in report['episodes'])
    report['excluded_ratio'] = report['excluded_frames'] / report['total_frames']
    report['method'] += '; additionally mask initial zero-target prefix until first |arm action| >0.001rad'
    report['time_compressed'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text()) != report:
            raise RuntimeError('Preserving a different existing mask')
    else:
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'}))


if __name__ == '__main__':
    main()
