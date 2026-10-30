"""Derive sampling weights from recorded joints/FK, without robot commands."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def contact_weight(tool_z, board_z):
    if not all(math.isfinite(x) for x in [tool_z, board_z]):
        raise ValueError('Nonfinite tool height')
    return 8. if tool_z <= board_z + .015 else 1.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from collect_expert_demo import kinematics
    config = json.loads((ROOT/'config/fukuwarai.json').read_text())
    manifest = json.loads((args.dataset/'conversion_manifest.json').read_text())
    sources = json.loads((args.dataset/'episode_sources.json').read_text())
    source_root = ROOT/manifest['source_data_root']
    kine = kinematics()
    rows = []
    for source in sources:
        folder = ROOT/source['source_repo_relative'] if source.get('source_repo_relative') else source_root/source['source_episode']
        frames = [json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()]
        weights = [contact_weight(float(kine.tool(frame['observation.state'][:4])[2]),
                                  config['board_surface_z_m']) for frame in frames]
        rows.append(dict(episode_index=source['episode_index'], source_episode=source['source_episode'],
                         frames=len(frames), near_board_frames=sum(w > 1 for w in weights), weights=weights))
    report = dict(method='8x sampling at tool height <= board+15mm; 1x elsewhere; recorded joint FK only',
                  raw_data_modified=False, threshold_above_board_m=.015,
                  scene_config_sha256=hashlib.sha256((ROOT/'config/fukuwarai.json').read_bytes()).hexdigest(),
                  conversion_manifest_sha256=hashlib.sha256((args.dataset/'conversion_manifest.json').read_bytes()).hexdigest(),
                  episodes=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(episodes=len(rows), frames=sum(r['frames'] for r in rows),
                         near_board_frames=sum(r['near_board_frames'] for r in rows))))


if __name__ == '__main__':
    main()
