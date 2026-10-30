#!/usr/bin/env python3
"""Offline motion/hold diagnostics on validation episodes, without robot control."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from train_fukuwarai_act import ROOT, source_setup, prepare, checkpoint_crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--dataset', type=Path, default=ROOT/'data/lerobot_v2.1/crane_plus_fukuwarai')
    ap.add_argument('--loss-mask', type=Path, default=None)
    args = ap.parse_args()
    source_setup(Path('/tmp/fukuwarai-pai-training-source/lerobot'))
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    args.output.mkdir(parents=True, exist_ok=True)
    # Snapshot the selected checkpoint so a concurrent best-model save cannot
    # change the model partway through this diagnostic.
    copy = args.output / 'checkpoint_snapshot'
    if copy.exists():
        raise RuntimeError('Preserving existing diagnostic snapshot; choose another output')
    shutil.copytree(args.checkpoint, copy)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(4)
    policy = ACTPolicy.from_pretrained(copy).to(device).eval()
    crop = checkpoint_crop(copy)
    root = args.dataset
    data = LeRobotDataset('crane_plus_fukuwarai', root=root, video_backend='pyav')
    val_ids = json.loads((root/'splits.json').read_text())['validation_episode_indices']
    indices = np.array([i for ep in val_ids for i in range(int(data.episode_data_index['from'][ep]),
                                                         int(data.episode_data_index['to'][ep]))])
    excluded_count = 0
    if args.loss_mask:
        sources = json.loads((root/'episode_sources.json').read_text())
        mask = json.loads(args.loss_mask.read_text())
        excluded = set()
        for ep in mask['episodes']:
            index = ep['episode_index']
            assert ep['source_episode'] == sources[index]['source_episode']
            start = int(data.episode_data_index['from'][index])
            assert ep['frames'] == int(data.episode_data_index['to'][index])-start
            excluded.update(start+i for i in ep['excluded_frames'])
        retained = np.array([i for i in indices if i not in excluded], dtype=int)
        excluded_count = len(indices)-len(retained)
        indices = retained
    states = np.array(data.hf_dataset.select(indices.tolist())['observation.state'])
    actions = np.array(data.hf_dataset.select(indices.tolist())['action'])
    moving = np.max(np.abs(actions[:,:4]-states[:,:4]), axis=1) > .005
    random = np.random.default_rng(20261022)
    groups = {'moving': indices[moving], 'stationary': indices[~moving]}
    # Always inspect the first retained task observation of every validation
    # episode; random frame samples can miss a failure to initiate movement.
    starts = []
    available = set(indices.tolist())
    for ep in val_ids:
        first = next((i for i in range(int(data.episode_data_index['from'][ep]),
                                      int(data.episode_data_index['to'][ep])) if i in available), None)
        if first is not None:
            starts.append(first)
    groups['task_start'] = np.array(starts, dtype=int)
    report = {'checkpoint': str(args.checkpoint), 'validation_only': True,
              'purpose': 'validation motion/hold/start diagnostics; not robot success',
              'dataset': str(root), 'image_crop_xywh': crop, 'loss_mask': str(args.loss_mask) if args.loss_mask else None,
              'excluded_validation_frames': excluded_count, 'groups': {}}
    for name, candidates in groups.items():
        if not len(candidates):
            report['groups'][name] = {'available_frames': 0, 'samples': 0,
                                     'mean_mae_rad': None, 'rows': []}
            continue
        sample_indices = random.choice(candidates, min(40, len(candidates)), replace=False)
        rows = []
        with torch.inference_mode():
            for i in sample_indices:
                item = data[int(i)]
                batch = prepare(torch.utils.data.default_collate([item]), device, crop)
                policy.reset()
                predicted = policy.select_action({k: batch[k] for k in policy.config.input_features}).cpu().numpy()[0]
                actual = item['action'].numpy()
                current = item['observation.state'].numpy()
                desired_delta, predicted_delta = actual[:4]-current[:4], predicted[:4]-current[:4]
                denom = np.linalg.norm(desired_delta)*np.linalg.norm(predicted_delta)
                rows.append({'global_frame': int(i), 'episode': int(item['episode_index']),
                             'mae_rad': float(np.mean(np.abs(predicted[:4]-actual[:4]))),
                             'hold_baseline_mae_rad': float(np.mean(np.abs(desired_delta))),
                             'desired_delta_norm_rad': float(np.linalg.norm(desired_delta)),
                             'predicted_delta_norm_rad': float(np.linalg.norm(predicted_delta)),
                             'direction_cosine': float(np.dot(desired_delta,predicted_delta)/denom) if denom > 1e-10 else None})
        values = [r['direction_cosine'] for r in rows if r['direction_cosine'] is not None]
        report['groups'][name] = {'available_frames': len(candidates), 'samples': len(rows),
                                 'mean_mae_rad': float(np.mean([r['mae_rad'] for r in rows])),
                                 'hold_baseline_mae_rad': float(np.mean([r['hold_baseline_mae_rad'] for r in rows])),
                                 'mean_direction_cosine': float(np.mean(values)) if values else None,
                                 'rows': rows}
    (args.output/'diagnostic.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({name:{k:v for k,v in result.items() if k!='rows'}
                      for name,result in report['groups'].items()}), flush=True)


if __name__ == '__main__':
    main()
