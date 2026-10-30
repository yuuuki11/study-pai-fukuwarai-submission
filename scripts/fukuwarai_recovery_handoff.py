"""Validate a future model-to-camera-teacher handoff without object-pose input.

Validation alone does not authorize moving the arm or bypassing the collector's
normal reset. Callers must retain the shared lock across prefix and recovery.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

FORBIDDEN = set(range(2026102510, 2026102515)) | set(range(2026102200, 2026102205)) | set(range(2026102300, 2026102305))

def validate_handoff(path, *, seed, scene_sha256, executor_sha256,
                     current_sim_s, current_joints):
    path = Path(path)
    data = json.loads(path.read_text())
    required = {'version', 'seed', 'scene_sha256', 'executor_sha256',
                'handoff_sim_s', 'handoff_joints', 'prefix_result',
                'prefix_result_sha256', 'model_success', 'reset_count'}
    if set(data) != required or data['version'] != 1:
        raise ValueError('Unsupported handoff schema')
    if seed in FORBIDDEN or data['seed'] != seed:
        raise ValueError('Forbidden or mismatched seed')
    if data['scene_sha256'] != scene_sha256 or data['executor_sha256'] != executor_sha256:
        raise ValueError('Changed scene or executor')
    if data['model_success'] is not False or data['reset_count'] != 1:
        raise ValueError('Recovery requires one original reset and failed prefix')
    age = float(current_sim_s) - float(data['handoff_sim_s'])
    if not np.isfinite(age) or not 0 <= age <= 2:
        raise ValueError(f'Stale handoff: age_sim_s={age:.6f}, current={current_sim_s}, saved={data["handoff_sim_s"]}')
    q, saved = np.asarray(current_joints, float), np.asarray(data['handoff_joints'], float)
    if q.shape != (5,) or saved.shape != (5,) or not np.isfinite([q, saved]).all() or np.max(np.abs(q-saved)) > .01:
        raise ValueError('Arm moved since handoff')
    result_path = Path(data['prefix_result'])
    if not result_path.is_absolute():
        result_path = path.parent / result_path
    raw = result_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != data['prefix_result_sha256']:
        raise ValueError('Prefix record changed')
    result = json.loads(raw)
    if result.get('success') is not False or result.get('seed') != seed:
        raise ValueError('Prefix must be a recorded failure for this seed')
    return data
