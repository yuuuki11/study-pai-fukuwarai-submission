#!/usr/bin/env python3
"""Local ACT baseline using the lecture's pinned LeRobot v2.1 implementation."""
import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CAMERA = 'observation.images.overhead'


def source_setup(path):
    sys.path.insert(0, str(path))
    code = path / 'lerobot/common/datasets/lerobot_dataset.py'
    expected = '5d5e8db8a9c563a4ab39dafa7323fdacc70eb6d44bdcab3fcc5903a0c86c97c1'
    if hashlib.sha256(code.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Expected the pinned lecture LeRobot v2.1 source')


def preprocessing_spec(crop=None, camera_cues=False, joint_history=False, tool_cues=False, silhouette_cues=False):
    if crop is not None:
        x, y, width, height = crop
        if min(x, y) < 0 or min(width, height) < 1 or x + width > 640 or y + height > 480:
            raise ValueError('Image crop must be inside the 640x480 camera image')
    spec = {'schema_version': 1, 'input_size_hw': [480, 640], 'output_size_hw': [240, 320],
            'crop_xywh': list(crop) if crop is not None else None,
            'resize': 'torch bilinear align_corners=False'}
    if camera_cues:
        spec.update(schema_version=2, environment_state='current_rgb_mouth_center_axis_visibility',
                    camera_features_sha256=hashlib.sha256((ROOT/'scripts/fukuwarai_camera_features.py').read_bytes()).hexdigest())
    if joint_history:
        if not camera_cues:
            raise ValueError('Joint history requires the RGB cue token')
        spec.update(schema_version=3, environment_state='current_rgb_mouth_cues_and_previous_joint_velocity',
                    joint_history_sha256=hashlib.sha256((ROOT/'scripts/fukuwarai_joint_history.py').read_bytes()).hexdigest(),
                    joint_history='previous raw observation within episode; delta*10/4 clipped to [-1,1]')
    if tool_cues:
        if not camera_cues:
            raise ValueError('Tool cues require camera cues')
        spec.update(schema_version=4, environment_state='current_rgb_mouth_cues_and_observed_tool_relative_xy_height',
                    tool_cues_sha256=hashlib.sha256((ROOT/'scripts/fukuwarai_tool_cues.py').read_bytes()).hexdigest(),
                    tool_geometry_sha256=hashlib.sha256((ROOT/'config/fukuwarai_tool_kinematics.json').read_bytes()).hexdigest(),
                    scene_geometry_sha256=hashlib.sha256((ROOT/'config/fukuwarai.json').read_bytes()).hexdigest())
    if tool_cues and joint_history:
        spec.update(schema_version=5, environment_state='current_rgb_mouth_cues_tool_relative_and_previous_joint_velocity')
    if silhouette_cues:
        if not (camera_cues and joint_history and tool_cues):
            raise ValueError('Silhouette cues require camera/tool/history inputs')
        spec.update(schema_version=6, environment_state='current_rgb_silhouette_center_axis_visibility_tool_and_joint_history',
                    silhouette_features_sha256=hashlib.sha256((ROOT/'scripts/fukuwarai_silhouette_features.py').read_bytes()).hexdigest(),
                    silhouette_pose_sha256=hashlib.sha256((ROOT/'scripts/fukuwarai_image_pose.py').read_bytes()).hexdigest())
    return spec


def checkpoint_crop(checkpoint):
    path = Path(checkpoint) / 'preprocessing.json'
    if not path.exists():
        return None  # Existing models were trained on the full camera frame.
    spec = json.loads(path.read_text())
    if spec != preprocessing_spec(spec.get('crop_xywh'), spec.get('schema_version') in (2,3,4,5,6), spec.get('schema_version') in (3,5,6), spec.get('schema_version') in (4,5,6), spec.get('schema_version') == 6):
        raise ValueError('Unsupported checkpoint image preprocessing')
    return spec['crop_xywh']


def checkpoint_camera_cues(checkpoint):
    checkpoint_crop(checkpoint)  # Validate schema and feature source hash first.
    path = Path(checkpoint) / 'preprocessing.json'
    return path.exists() and json.loads(path.read_text()).get('schema_version') in (2,3,4,5,6)


def checkpoint_joint_history(checkpoint):
    checkpoint_crop(checkpoint)
    path = Path(checkpoint) / 'preprocessing.json'
    return path.exists() and json.loads(path.read_text()).get('schema_version') in (3,5,6)


def checkpoint_tool_cues(checkpoint):
    checkpoint_crop(checkpoint)
    path = Path(checkpoint) / 'preprocessing.json'
    return path.exists() and json.loads(path.read_text()).get('schema_version') in (4,5,6)


def checkpoint_silhouette_cues(checkpoint):
    checkpoint_crop(checkpoint)
    path = Path(checkpoint) / 'preprocessing.json'
    return path.exists() and json.loads(path.read_text()).get('schema_version') == 6


def prepare(batch, device, crop=None, camera_cues=False, joint_history=False, tool_cues=False, silhouette_cues=False):
    if tool_cues and not camera_cues:
        raise ValueError('Tool cues require camera cues')
    if camera_cues:
        from fukuwarai_camera_features import camera_features
        if silhouette_cues:
            from fukuwarai_silhouette_features import cached_silhouette_features as camera_features
        rgb = batch[CAMERA].detach().cpu()
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, 480, 640):
            raise ValueError('Camera cues require original RGB batch')
        images = (rgb.permute(0, 2, 3, 1).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        batch = dict(batch)
        batch['observation.environment_state'] = torch.from_numpy(np.stack([camera_features(im) for im in images]))
        if tool_cues:
            from fukuwarai_tool_cues import tool_relative_cues
            states = batch['observation.state'].detach().cpu().numpy()
            mouths = batch['observation.environment_state'].numpy()
            extra = torch.from_numpy(np.stack([tool_relative_cues(q, m) for q, m in zip(states, mouths)]))
            batch['observation.environment_state'] = torch.cat([batch['observation.environment_state'], extra], dim=-1)
        if joint_history:
            from fukuwarai_joint_history import velocity_cue
            cue = velocity_cue(batch['observation.state'].cpu(), batch['observation.previous_state'].cpu())
            batch['observation.environment_state'] = torch.cat([batch['observation.environment_state'], cue], dim=-1)
    elif joint_history:
        raise ValueError('Joint history requires camera cues')
    batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    if crop is not None:
        preprocessing_spec(crop)
        x, y, width, height = crop
        if tuple(batch[CAMERA].shape[-2:]) != (480, 640):
            raise ValueError('Cropping expects the original 640x480 camera image')
        batch[CAMERA] = batch[CAMERA][..., y:y+height, x:x+width]
    batch[CAMERA] = F.interpolate(batch[CAMERA], size=(240, 320), mode='bilinear', align_corners=False)
    return batch


def validation_indices(candidates, episode_count):
    count=min(len(candidates),max(3,60//max(1,episode_count)))
    if count==3:
        return [candidates[0],candidates[len(candidates)//2],candidates[-1]]
    return [candidates[int(i)] for i in np.linspace(0,len(candidates)-1,count)]


def evaluate(policy, dataset, device, episode_ids, crop=None, retained_frames=None, camera_cues=False, joint_history=False, tool_cues=False, silhouette_cues=False):
    errors, stationary, per_episode = [], [], []
    policy.eval()
    with torch.inference_mode():
        for ep in episode_ids:
            start = int(dataset.episode_data_index['from'][ep])
            end = int(dataset.episode_data_index['to'][ep])
            ep_errors = []
            candidates = list(range(start, end)) if retained_frames is None else [i for i in range(start, end) if retained_frames[i]]
            if not candidates:
                raise ValueError('Validation episode has no retained task frames')
            for i in validation_indices(candidates,len(episode_ids)):
                item = dataset[i]
                batch = torch.utils.data.default_collate([item])
                batch = prepare(batch, device, crop, camera_cues, joint_history, tool_cues, silhouette_cues)
                policy.reset()
                inputs = {k: batch[k] for k in policy.config.input_features}
                assert 'action' not in inputs and 'action_is_pad' not in inputs
                pred = policy.select_action(inputs)
                target = batch['action'][:, 0]
                err = (pred - target).abs().cpu().numpy()[0]
                baseline = (batch['observation.state'] - target).abs().cpu().numpy()[0]
                errors.append(err)
                stationary.append(baseline)
                ep_errors.append(float(err.mean()))
            per_episode.append({'episode': ep, 'mean_absolute_joint_error_rad': float(np.mean(ep_errors))})
    policy.train()
    return {'samples': len(errors), 'ground_truth_actions_passed_to_policy': False,
            'sampling': 'retained task uniform episode-balanced' if retained_frames is not None else 'raw uniform episode-balanced',
            'inference_inputs': list(policy.config.input_features), 'mae_rad': float(np.mean(errors)),
            'per_joint_mae_rad': np.mean(errors, axis=0).tolist(),
            'hold_current_joint_baseline_mae_rad': float(np.mean(stationary)),
            'per_episode': per_episode,
            'note': 'Offline one-step prediction error is not task success.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path('/tmp/fukuwarai-pai-training-source/lerobot'))
    ap.add_argument('--dataset', type=Path, default=ROOT / 'data/lerobot_v2.1/crane_plus_fukuwarai_outline_v2')
    ap.add_argument('--output', type=Path, default=ROOT / 'data/training/act_baseline_20261022')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--learning-rate', type=float, default=0.0001)
    ap.add_argument('--backbone-lr', type=float, default=None)
    ap.add_argument('--kl-weight', type=float, default=10.)
    ap.add_argument('--motion-sampling', action='store_true')
    ap.add_argument('--sampling-weights', type=Path, default=None, help='Read-only joint-FK contact sampling report')
    ap.add_argument('--cosine-lr', action='store_true')
    ap.add_argument('--loss-mask', type=Path, default=None)
    ap.add_argument('--image-crop', type=int, nargs=4, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'), default=None)
    ap.add_argument('--bounded-action-targets', action='store_true', help='Bound teacher absolute targets to recorded state +/-0.1rad, matching rollout limiter')
    ap.add_argument('--tool-cues', action='store_true', help='Add observed-joint FK tool-to-RGB-mouth xy and board height cues')
    ap.add_argument('--joint-history', action='store_true', help='Add previous observed joint velocity to RGB cue token')
    ap.add_argument('--camera-cues', action='store_true', help='Supply current RGB mouth pixel/axis/visibility cues to ACT environment token')
    ap.add_argument('--tcp-loss-weight',type=float,default=0.,help='Optional teacher-action TCP loss weight per metre;0 preserves pinned ACT loss')
    ap.add_argument('--silhouette-cues', action='store_true')
    ap.add_argument('--allow-recovery-segments', action='store_true', help='Opt in to independently verified contiguous recovery segment sources')
    ap.add_argument('--task-validation', action='store_true', help='Select checkpoints on retained task frames; requires --loss-mask')
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--audit-only', action='store_true')
    ap.add_argument('--initial-checkpoint',type=Path,default=None)
    ap.add_argument('--allow-legacy-data',action='store_true')
    ap.add_argument('--evaluate-checkpoint', type=Path, default=None)
    args = ap.parse_args()
    preprocessing_spec(args.image_crop, args.camera_cues, args.joint_history, args.tool_cues, args.silhouette_cues)
    if args.sampling_weights and args.motion_sampling:
        ap.error('--sampling-weights and --motion-sampling are mutually exclusive')
    if not 0 <= args.tcp_loss_weight <= 10:
        ap.error('tcp-loss-weight must be within[0,10]')
    if args.task_validation and not args.loss_mask:
        ap.error('--task-validation requires --loss-mask')
    source_setup(args.source)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.compute_stats import aggregate_stats
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.policies.act.configuration_act import ACTConfig
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import FeatureType
    from convert_fukuwarai_to_lerobot import check_episode, read_json

    if not args.audit_only and not args.evaluate_checkpoint and (args.output / 'checkpoint').exists():
        raise RuntimeError('Preserving existing checkpoint; choose a new --output')
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError('steps and batch size must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    split = read_json(args.dataset / 'splits.json')
    train_ids, val_ids = split['train_episode_indices'], split['validation_episode_indices']
    assert not set(train_ids) & set(val_ids)
    config = read_json(ROOT / 'config/fukuwarai.json')
    scene_hash = hashlib.sha256((ROOT / 'config/fukuwarai.json').read_bytes()).hexdigest()
    sources = read_json(args.dataset / 'episode_sources.json')
    if (len(sources) < 2 or len({s['seed'] for s in sources}) != len(sources)
            or len({s['source_episode'] for s in sources}) != len(sources)
            or [s['episode_index'] for s in sources] != list(range(len(sources)))
            or set(train_ids) | set(val_ids) != set(range(len(sources)))
            or not train_ids or not val_ids
            or any(s['seed'] in set(range(2026102200,2026102205)) | set(range(2026102300,2026102305)) | set(range(2026102510,2026102515)) for s in sources)):
        raise RuntimeError('Invalid source mapping, split partition, or reserved evaluation seed in training data')
    import convert_fukuwarai_to_lerobot as converter
    from fukuwarai_success import load_rules
    manifest = read_json(args.dataset/'conversion_manifest.json')
    selected_rules = manifest.get('success_rules')
    if selected_rules != load_rules() and not args.allow_legacy_data:
        raise RuntimeError('This improvement run requires v2-qualified data; legacy needs explicit --allow-legacy-data')
    converter.SUCCESS_RULES = selected_rules
    source_root = ROOT / manifest.get('source_data_root','data/expert_images')
    failures = []
    for s in sources:
        if s.get('derived_segment'):
            if not args.allow_recovery_segments or s['episode_index'] not in train_ids:
                raise RuntimeError('Recovery segments require explicit opt-in and train-only split')
            from fukuwarai_recovery_segment_loader import load_segment
            load_segment(ROOT/s['source_repo_relative'], ROOT, config, scene_hash)
            reason = None
        else:
            item, reason = check_episode(source_root / s['source_episode'], scene_hash, config)
        if reason:
            failures.append({'source': s['source_episode'], 'reason': reason})
    if failures:
        raise RuntimeError(str(failures))
    # Chunk index zero is the recorded next-tick target. Do not shift it by -0.1s.
    delta = {'action': [i / 10 for i in range(16)]}
    # Pinned v2.1 indexes delta timestamps by original episode id. Loading a
    # noncontiguous episode subset breaks that code. Keep the full read-only
    # loader and subset global frame indices at the PyTorch sampler boundary.
    full = LeRobotDataset('crane_plus_fukuwarai', root=args.dataset,
                         delta_timestamps=delta, video_backend='pyav')
    target_spec = {'version': 'recorded_absolute_next_tick', 'bounded': False}
    if args.evaluate_checkpoint:
        target_path = args.evaluate_checkpoint / 'training_targets.json'
        if target_path.exists():
            target_spec = json.loads(target_path.read_text())
            if target_spec.get('bounded') and (target_spec.get('version') != 'recorded_state_bounded_absolute_v1'
                    or target_spec.get('limit_rad') != .1
                    or target_spec.get('helper_sha256') != hashlib.sha256((ROOT / 'scripts/fukuwarai_action_targets.py').read_bytes()).hexdigest()):
                raise RuntimeError('Unsupported checkpoint target limit')
        if args.bounded_action_targets and not target_spec.get('bounded'):
            raise RuntimeError('Evaluation target semantics must match checkpoint')
    if args.bounded_action_targets or target_spec.get('bounded'):
        from fukuwarai_action_targets import BoundedActionTargets
        full = BoundedActionTargets(full)
        target_spec = {'version': 'recorded_state_bounded_absolute_v1', 'bounded': True, 'limit_rad': .1,
                       'future_labels': 'each slot uses its own recorded state; episode end repeats last label',
                       'policy_inputs_unchanged': True,
                       'helper_sha256': hashlib.sha256((ROOT / 'scripts/fukuwarai_action_targets.py').read_bytes()).hexdigest()}
    if args.joint_history or (args.evaluate_checkpoint and checkpoint_joint_history(args.evaluate_checkpoint)):
        from fukuwarai_joint_history import CausalJointHistory
        full = CausalJointHistory(full)
    if full.meta.total_episodes != len(sources):
        raise RuntimeError('Dataset metadata and source episode mapping disagree')
    def frame_indices(ids):
        return [j for ep in ids for j in range(int(full.episode_data_index['from'][ep]),
                                              int(full.episode_data_index['to'][ep]))]
    train = torch.utils.data.Subset(full, frame_indices(train_ids))
    val = torch.utils.data.Subset(full, frame_indices(val_ids))
    retained_validation = None
    if args.loss_mask:
        mask_report = json.loads(args.loss_mask.read_text())
        excluded = np.zeros(len(full), dtype=bool)
        for record in mask_report['episodes']:
            ep = record['episode_index']
            assert record['source_episode'] == sources[ep]['source_episode']
            start = int(full.episode_data_index['from'][ep])
            end = int(full.episode_data_index['to'][ep])
            assert record['frames'] == end - start
            excluded[start + np.array(record['excluded_frames'], dtype=int)] = True
        allowed_train = [i for i in train.indices if not excluded[i]]
        class LossMaskedFrames(torch.utils.data.Dataset):
            def __init__(self, indices):
                self.indices = indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, i):
                global_index = self.indices[i]
                item = full[global_index]
                ep = int(item['episode_index'])
                end = int(full.episode_data_index['to'][ep])
                future = np.minimum(np.arange(global_index, global_index + 16), end - 1)
                item['action_is_pad'] = item['action_is_pad'] | torch.tensor(excluded[future])
                return item
        train = LossMaskedFrames(allowed_train)
        if args.task_validation:
            retained_validation = ~excluded
    audit = {'episodes': len(sources), 'frames': len(full),
             'train_episodes': len(train_ids), 'validation_episodes': len(val_ids),
             'train_frames': len(train), 'validation_frames': len(val),
             'invalid_episodes': failures, 'scene_config_sha256': scene_hash,
             'source_joint_order': ['joint1', 'joint2', 'joint3', 'joint4', 'gripper'],
             'normalization': 'joint/action: current train episodes only; image: initial train statistics if warm-started',
             'raw_data_modified': False, 'training_targets': target_spec, 'loss_mask': str(args.loss_mask) if args.loss_mask else None,
             'raw_train_frames': len(frame_indices(train_ids)), 'sampled_train_frames': len(train),
             'conversion_manifest_sha256': hashlib.sha256((args.dataset / 'conversion_manifest.json').read_bytes()).hexdigest()}
    (args.output / 'data_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)
    if args.audit_only:
        return
    if args.evaluate_checkpoint:
        policy = ACTPolicy.from_pretrained(args.evaluate_checkpoint).to(args.device).eval()
        result = evaluate(policy, full, args.device, val_ids, checkpoint_crop(args.evaluate_checkpoint), retained_validation, checkpoint_camera_cues(args.evaluate_checkpoint), checkpoint_joint_history(args.evaluate_checkpoint), checkpoint_tool_cues(args.evaluate_checkpoint), checkpoint_silhouette_cues(args.evaluate_checkpoint))
        result['checkpoint_sha256'] = hashlib.sha256((args.evaluate_checkpoint / 'model.safetensors').read_bytes()).hexdigest()
        (args.output / 'inference_validation.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    spec = importlib.util.spec_from_file_location('features', ROOT / 'config/pai_training_visual_features.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    features = dataset_to_policy_features(mod.features)
    features[CAMERA].shape = (3, 240, 320)
    if args.tool_cues:
        features['observation.environment_state'].shape = (8,)
    if args.joint_history:
        features['observation.environment_state'].shape = (13 if args.tool_cues else 10,)
    cfg = ACTConfig(input_features={k: v for k, v in features.items() if v.type != FeatureType.ACTION},
                    output_features={k: v for k, v in features.items() if v.type == FeatureType.ACTION},
                    chunk_size=16, n_action_steps=1, device=args.device, kl_weight=args.kl_weight)
    stats = aggregate_stats([full.meta.episodes_stats[i] for i in train_ids])
    if args.bounded_action_targets:
        # Before any joint-history wrapper: train-only target statistics.
        target_dataset = full.base if args.joint_history else full
        stats['action'] = target_dataset.train_stats(train_ids)
    policy = ACTPolicy(cfg, dataset_stats=stats).to(args.device)
    initial_sha = None
    if args.initial_checkpoint:
        if checkpoint_camera_cues(args.initial_checkpoint) != args.camera_cues:
            raise RuntimeError('Initial checkpoint environment token semantics differ; train a fresh camera-cue ACT first')
        if checkpoint_crop(args.initial_checkpoint) != args.image_crop:
            raise RuntimeError('Initial checkpoint crop differs from this training run')
        if checkpoint_tool_cues(args.initial_checkpoint) and not args.tool_cues:
            raise RuntimeError('Cannot remove observed tool cues on warm start')
        if args.tool_cues and checkpoint_joint_history(args.initial_checkpoint) and not checkpoint_tool_cues(args.initial_checkpoint):
            raise RuntimeError('Cannot replace joint history semantics with tool cues on warm start')
        extending_tool = args.tool_cues and not checkpoint_tool_cues(args.initial_checkpoint)
        extending_history = args.joint_history and not checkpoint_joint_history(args.initial_checkpoint)
        if checkpoint_joint_history(args.initial_checkpoint) and not args.joint_history:
            raise RuntimeError('Cannot remove observed joint history on warm start')
        from safetensors.torch import load_file
        initial = load_file(str(args.initial_checkpoint/'model.safetensors'))
        from fukuwarai_warmstart import rebase_state
        current, rebase_report = rebase_state(initial,policy.state_dict(),extend_environment_history=extending_history, extend_environment_tool=extending_tool)
        policy.load_state_dict(current,strict=True)
        (args.output/'warmstart_rebase.json').write_text(json.dumps(rebase_report,indent=2))
        initial_sha = hashlib.sha256((args.initial_checkpoint/'model.safetensors').read_bytes()).hexdigest()
    settings = {'training_targets': target_spec, 'initial_checkpoint_sha256': initial_sha, 'steps': args.steps, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate, 'backbone_lr': args.backbone_lr,
                'kl_weight': args.kl_weight, 'cosine_lr': args.cosine_lr, 'motion_sampling': args.motion_sampling, 'loss_mask': str(args.loss_mask) if args.loss_mask else None,
                'seed': 2026, 'chunk_size': 16, 'n_action_steps': 1, 'device': args.device,
                'image_resize': [240, 320], 'image_resize_mode': 'torch bilinear align_corners=False',
                'tcp_loss_weight':args.tcp_loss_weight,
                'tcp_loss_helper_sha256':hashlib.sha256((ROOT/'scripts/fukuwarai_tcp_loss.py').read_bytes()).hexdigest() if args.tcp_loss_weight else None,
                'image_preprocessing': preprocessing_spec(args.image_crop, args.camera_cues, args.joint_history, args.tool_cues, args.silhouette_cues),
                'checkpoint_selection': 'retained task uniform episode-balanced validation MAE' if args.task_validation else 'raw uniform episode-balanced validation MAE',
                'environment_token': 'current RGB mouth cues plus observed FK tool-relative xy/height and previous joint velocity' if args.tool_cues and args.joint_history else ('current RGB mouth cues plus observed FK tool-relative xy and height' if args.tool_cues else ('current RGB mouth cues plus previous observed joint velocity' if args.joint_history else ('current RGB-only mouth cues' if args.camera_cues else 'duplicate joint state'))),
                'lecture_source_commit': 'b4d6b5459066a80433475a83567504988875248a',
                'lerobot_commit': '768e36660d1408c71118a2760f831c037fbfa17d',
                'train_episode_indices': train_ids, 'validation_episode_indices': val_ids,
                'sampling_weights': str(args.sampling_weights) if args.sampling_weights else None,
                'sampling_weights_sha256': hashlib.sha256(args.sampling_weights.read_bytes()).hexdigest() if args.sampling_weights else None,
                'model_inputs': list(cfg.input_features), 'action_times_s': delta['action'],
                'torch': torch.__version__, 'dataset_scene_hash': scene_hash,
                'trainer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'loss_mask_sha256': hashlib.sha256(args.loss_mask.read_bytes()).hexdigest() if args.loss_mask else None,
                'purpose': 'Local lecture ACT experiment; see data audit, preprocessing, and checkpoint selection settings'}
    (args.output / 'run_config.json').write_text(json.dumps(settings, indent=2))
    sampler = None
    if args.motion_sampling:
        # Sample movement frames more often, retaining stationary frames for
        # learning completion. This never changes the raw data or validation.
        states = np.array(full.hf_dataset.select(train.indices)['observation.state'])
        targets = np.array(full.hf_dataset.select(train.indices)['action'])
        moving = np.max(np.abs(targets - states), axis=1) > .005
        weights = torch.tensor(np.where(moving, 4., 1.), dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(train), replacement=True,
                    generator=torch.Generator().manual_seed(2026))
    if args.sampling_weights:
        from fukuwarai_sampling import selected_weights
        selected = torch.tensor(selected_weights(read_json(args.sampling_weights), sources,
            full.episode_data_index, train.indices, scene_hash, audit['conversion_manifest_sha256']), dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(selected, len(train), replacement=True,
                    generator=torch.Generator().manual_seed(2026))
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=sampler is None,
                                         sampler=sampler, num_workers=0,
                                         generator=torch.Generator().manual_seed(2026), drop_last=True)
    if args.backbone_lr is None:
        params = policy.parameters()
    else:
        params = [{'params': [p for n, p in policy.named_parameters() if p.requires_grad and not n.startswith('model.backbone')]},
                  {'params': [p for n, p in policy.named_parameters() if p.requires_grad and n.startswith('model.backbone')],
                   'lr': args.backbone_lr}]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    scheduler = (torch.optim.lr_scheduler.LambdaLR(optimizer,
                  lambda step: .1 + .9 * .5 * (1 + math.cos(math.pi * min(step,args.steps)/args.steps)))
                 if args.cosine_lr else None)
    policy.train()
    started = time.monotonic()
    step = 0
    best_mae = float("inf")
    with (args.output / 'training.jsonl').open('w') as log:
        while step < args.steps:
            for raw in loader:
                batch = prepare(raw, args.device, args.image_crop, args.camera_cues, args.joint_history, args.tool_cues, args.silhouette_cues)
                from fukuwarai_tcp_loss import forward_with_tcp_loss
                loss, details = forward_with_tcp_loss(policy,batch,args.tcp_loss_weight,config['board_surface_z_m'])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                optimizer.zero_grad()
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                step += 1
                row = {'step': step, 'loss': loss.item(), **details, 'gradient_norm': float(grad),
                       'elapsed_wall_s': time.monotonic() - started, 'learning_rates': [g['lr'] for g in optimizer.param_groups]}
                log.write(json.dumps(row) + '\n')
                log.flush()
                if step == 1 or step % 10 == 0:
                    print(json.dumps(row), flush=True)
                if step % 250 == 0 or step == args.steps:
                    # Checkpoint selection uses validation only; test seeds
                    # are reserved for the final measured robot rollouts.
                    validation = evaluate(policy, full, args.device, val_ids, args.image_crop, camera_cues=args.camera_cues, joint_history=args.joint_history, tool_cues=args.tool_cues, silhouette_cues=args.silhouette_cues)
                    validation['step'] = step
                    (args.output / f'validation_{step:06d}.json').write_text(json.dumps(validation, indent=2))
                    selected_validation = validation
                    if retained_validation is not None:
                        selected_validation = evaluate(policy, full, args.device, val_ids, args.image_crop, retained_validation, args.camera_cues, args.joint_history, args.tool_cues, args.silhouette_cues)
                        selected_validation['step'] = step
                        (args.output / f'validation_task_{step:06d}.json').write_text(json.dumps(selected_validation, indent=2))
                    policy.save_pretrained(args.output / 'checkpoint_last')
                    (args.output / 'checkpoint_last/training_targets.json').write_text(json.dumps(target_spec, indent=2))
                    (args.output / 'checkpoint_last/preprocessing.json').write_text(json.dumps(preprocessing_spec(args.image_crop, args.camera_cues, args.joint_history, args.tool_cues, args.silhouette_cues), indent=2))
                    if selected_validation['mae_rad'] < best_mae:
                        best_mae = selected_validation['mae_rad']
                        policy.save_pretrained(args.output / 'checkpoint')
                        (args.output / 'checkpoint/training_targets.json').write_text(json.dumps(target_spec, indent=2))
                        (args.output / 'checkpoint/preprocessing.json').write_text(json.dumps(preprocessing_spec(args.image_crop, args.camera_cues, args.joint_history, args.tool_cues, args.silhouette_cues), indent=2))
                        (args.output / 'best_checkpoint.json').write_text(json.dumps({'step': step, 'mae_rad': best_mae,
                            'sampling': selected_validation['sampling'], 'raw_mae_rad': validation['mae_rad']}))
                    # Save optimizer + CPU/CUDA/MPS-independent model state for
                    # an interrupted long run to be recoverable.
                    torch.save({'step': step, 'optimizer': optimizer.state_dict()}, args.output / 'optimizer.pt')
                if step >= args.steps:
                    break
    result = evaluate(policy, full, args.device, val_ids, args.image_crop, camera_cues=args.camera_cues, joint_history=args.joint_history, tool_cues=args.tool_cues, silhouette_cues=args.silhouette_cues)
    result['training_wall_s'] = time.monotonic() - started
    (args.output / 'offline_validation.json').write_text(json.dumps(result, indent=2))
    if retained_validation is not None:
        task_result = evaluate(policy, full, args.device, val_ids, args.image_crop, retained_validation, args.camera_cues, args.joint_history, args.tool_cues, args.silhouette_cues)
        (args.output / 'offline_task_validation.json').write_text(json.dumps(task_result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
