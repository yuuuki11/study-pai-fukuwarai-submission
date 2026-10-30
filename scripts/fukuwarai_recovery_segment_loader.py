"""Strict opt-in loader for separately audited recovery intervals."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from fukuwarai_success import MouthScorer,HoldScorer,load_rules
from fukuwarai_tool_cues import tool_xyz_course


def load_segment(folder, repo_root, config, expected_config):
    folder,repo_root=Path(folder),Path(repo_root)
    manifest=json.loads((folder/'segment_manifest.json').read_text())
    if manifest['schema']!='contiguous_recovery_segment_v1':raise ValueError('Unknown segment schema')
    raw=(repo_root/manifest['raw_source_relative_to_repo']).resolve()
    if not raw.is_relative_to(repo_root.resolve()):raise ValueError('Source escapes repository')
    for name,sha in manifest['raw_hashes'].items():
        if name not in ('result.json','frames.jsonl','observations.jsonl','scene_config.json') or hashlib.sha256((raw/name).read_bytes()).hexdigest()!=sha:raise ValueError('Raw source hash mismatch')
    if set(manifest['raw_hashes'])!={'result.json','frames.jsonl','observations.jsonl','scene_config.json'}:raise ValueError('Incomplete provenance')
    result=json.loads((raw/'result.json').read_text())
    if result.get('source')!='camera_visual_recovery' or not result.get('physical_success') or result.get('oracle_coordinates_in_control') is not False:raise ValueError('Invalid recovery provenance')
    if result['scene_config_sha256']!=expected_config or json.loads((raw/'scene_config.json').read_text())!=config or result['success_rules']!=load_rules():raise ValueError('Changed physical scene or success rules')
    original=[json.loads(l) for l in (raw/'frames.jsonl').read_text().splitlines()];first,last=manifest['original_frame_range_inclusive'];rows=[json.loads(l) for l in (folder/'frames.jsonl').read_text().splitlines()]
    if len(rows)!=last-first+1 or len(rows)<100:raise ValueError('Segment length mismatch')
    for i,row in enumerate(rows):
        prior=original[first+i]
        if row['frame_index']!=i or any(row[k]!=prior[k] for k in ('sim_time_ns','action','observation.state','nearest_state_offset_ms')):raise ValueError('Segment values changed')
        if row['timestamp']!=(row['sim_time_ns']-rows[0]['sim_time_ns'])/1e9:raise ValueError('Invalid segment clock')
        if row['nearest_state_offset_ms']>30 or not np.isfinite(row['action']+row['observation.state']).all():raise ValueError('Invalid state/action')
        if (folder/row['image']).resolve()!=(raw/prior['image']).resolve():raise ValueError('Image provenance mismatch')
        with Image.open(folder/row['image']) as im:
            if im.size!=(640,480):raise ValueError('Image dimensions changed')
            im.verify()
    gaps=np.diff([r['sim_time_ns'] for r in rows])/1e9
    if np.any((gaps<.09)|(gaps>.11)):raise ValueError('Segment crosses cadence gap')
    h=HoldScorer(MouthScorer(config));best=0;lo,hi=rows[0]['sim_time_ns']/1e9,rows[-1]['sim_time_ns']/1e9
    for line in (raw/'observations.jsonl').read_text().splitlines():
        r=json.loads(line)
        if not lo<=r['sim_s']<=hi:continue
        q=np.asarray(r['joint_positions']);metrics=h.update(r['sim_s'],r['mouth_xyz_m'],r['mouth_quaternion_xyzw'],float(tool_xyz_course(q[[0,1,3,2,4]])[2]));best=max(best,metrics['held_sim_s'])
    if best<1:raise ValueError('Segment strict v2 hold missing')
    # This is conversion metadata, not a rewritten raw acceptance decision.
    result=dict(result,source='camera_visual_recovery_segment',segment_provenance=manifest,segment_independent_hold_s=best)
    return folder,result,rows
