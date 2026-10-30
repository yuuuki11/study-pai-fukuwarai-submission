"""Materialize audited contiguous recovery intervals; never relabel raw episodes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize(source, output, first, last):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists; preserve prior artifacts')
    result = json.loads((source/'result.json').read_text())
    if result.get('source') != 'camera_visual_recovery' or not result.get('physical_success'):
        raise ValueError('Requires physically successful camera recovery source')
    if result.get('oracle_coordinates_in_control') is not False:
        raise ValueError('Oracle-free provenance required')
    frames = [json.loads(l) for l in (source/'frames.jsonl').read_text().splitlines()]
    if not 0 <= first <= last < len(frames) or last-first+1 < 100:
        raise ValueError('Invalid segment boundary or fewer than100 frames')
    rows = frames[first:last+1]
    times = [r['sim_time_ns'] for r in rows]
    gaps = [(b-a)/1e9 for a,b in zip(times,times[1:])]
    if any(not .09 <= gap <= .11 for gap in gaps):
        raise ValueError('Segment crosses a cadence gap')
    for row in rows:
        if not 0 <= row['nearest_state_offset_ms'] <= 30:
            raise ValueError('Joint offset exceeds30ms')
        if any(len(row[k]) != 5 or not all(math.isfinite(v) for v in row[k]) for k in ('observation.state','action')):
            raise ValueError('Invalid joint/action features')
        image=(source/row['image']).resolve()
        if not image.is_relative_to(source) or not image.is_file():
            raise ValueError('Image missing or escapes source')
    output.mkdir(parents=True)
    (output/'images').mkdir()
    hashes={name:digest(source/name) for name in ['result.json','frames.jsonl','observations.jsonl','scene_config.json']}
    with (output/'frames.jsonl').open('w') as handle:
        for i,row in enumerate(rows):
            row=dict(row);image=source/row['image'];target=output/'images'/f'{i:06d}.jpg';target.symlink_to(os.path.relpath(image,target.parent))
            row.update(frame_index=i,timestamp=(row['sim_time_ns']-times[0])/1e9,image=f'images/{i:06d}.jpg')
            handle.write(json.dumps(row)+'\n')
    manifest={'schema':'contiguous_recovery_segment_v1','raw_source':str(source),'raw_source_relative_to_repo':str(source.relative_to(Path(__file__).resolve().parents[1])) if source.is_relative_to(Path(__file__).resolve().parents[1]) else None,'raw_hashes':hashes,
              'raw_accepted_for_training':result.get('accepted_for_training'),
              'original_frame_range_inclusive':[first,last],'frames':len(rows),
              'maximum_interval_s':max(gaps),'seed_group':result['seed'],
              'policy_prefix_excluded':True,'cross_gap_history_allowed':False,
              'cross_gap_action_targets_allowed':False,'model_success':False,
              'segment_success_requires_independent_rescore':True,
              'training_eligible':False}
    (output/'segment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--first',required=True,type=int);p.add_argument('--last',required=True,type=int)
    a=p.parse_args();print(json.dumps(materialize(a.source,a.output,a.first,a.last)))
