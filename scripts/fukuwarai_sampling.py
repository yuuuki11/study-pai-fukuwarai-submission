"""Validate frame-aligned reports before selecting training-only weights."""
import numpy as np


def selected_weights(report, sources, bounds, indices, scene_hash, manifest_hash):
    if (report.get('scene_config_sha256') != scene_hash or
            report.get('conversion_manifest_sha256') != manifest_hash):
        raise RuntimeError('Sampling report source dataset or scene mismatch')
    weights = np.full(int(bounds['to'][-1]), np.nan)
    seen = set()
    for record in report['episodes']:
        ep = record['episode_index']
        if ep in seen or not 0 <= ep < len(sources):
            raise RuntimeError('Duplicate or invalid sampling episode')
        seen.add(ep)
        start, end = int(bounds['from'][ep]), int(bounds['to'][ep])
        values = np.asarray(record['weights'], dtype=float)
        if (record['source_episode'] != sources[ep]['source_episode'] or record['frames'] != end-start
                or values.shape != (end-start,) or not np.isfinite(values).all()
                or np.any(values <= 0) or np.any(values > 100)):
            raise RuntimeError('Invalid sampling report frame alignment or weights')
        weights[start:end] = values
    if not np.isfinite(weights).all():
        raise RuntimeError('Sampling report omits episodes')
    # Caller supplies retained training indices; validation and masked frames
    # remain outside the sampler, with original episode/timestamp boundaries.
    return weights[indices]
