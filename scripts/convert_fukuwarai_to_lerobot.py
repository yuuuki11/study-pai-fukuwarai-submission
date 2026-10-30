#!/usr/bin/env python3
"""Convert accepted camera-directed episodes to the lecture's LeRobot v2.1 format.

Run inside pai-advanced1 so this uses the pinned course LeRobot source and its
installed Python dependencies. No Hub or other remote upload is performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image
from fukuwarai_success import load_rules

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "expert_images_v2"
SUCCESS_RULES = load_rules()
CONFIG_PATH = ROOT / "config" / "fukuwarai.json"
FPS = 10
TARGET_ACCEPTED = 100
LE_ROBOT_COMMIT = "768e36660d1408c71118a2760f831c037fbfa17d"
PAI_TRAINING_COMMIT = "b4d6b5459066a80433475a83567504988875248a"
LE_ROBOT_DATASET_SOURCE_SHA256 = "5d5e8db8a9c563a4ab39dafa7323fdacc70eb6d44bdcab3fcc5903a0c86c97c1"
RAW_STATE_NAMES = [
    "crane_plus_joint1", "crane_plus_joint2", "crane_plus_joint4",
    "crane_plus_joint3", "crane_plus_joint_hand",
]
COURSE_STATE_NAMES = ["joint1", "joint2", "joint3", "joint4", "gripper"]
COURSE_INDEX_FROM_RAW = [0, 1, 3, 2, 4]
TASK = "Push the mouth piece to the face target and release it for a one second hold."


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_sources(source_root):
    dataset_file = source_root / "lerobot" / "common" / "datasets" / "lerobot_dataset.py"
    if not dataset_file.is_file():
        raise RuntimeError("The lecture LeRobot source is missing: " + str(dataset_file))
    source_hash = hashlib.sha256(dataset_file.read_bytes()).hexdigest()
    if source_hash != LE_ROBOT_DATASET_SOURCE_SHA256:
        raise RuntimeError(
            "LeRobot source does not match the pinned lecture checkout (expected commit %s, source hash %s)."
            % (LE_ROBOT_COMMIT, source_hash)
        )
    sys.path.insert(0, str(source_root))
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.common.datasets import lerobot_dataset as dataset_module
    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError("Expected lecture LeRobot format v2.1, found " + str(CODEBASE_VERSION))

    # The pinned course source defaults to libsvtav1, but this container's
    # ffmpeg build does not include that encoder. Keep the course's v2.1 writer
    # and video layout, using its same encoder function with widely supported
    # H.264 at CRF 18 for a high-quality local dataset that PyAV can decode.
    course_encode_video_frames = dataset_module.encode_video_frames

    def encode_h264(imgs_dir, video_path, fps, **kwargs):
        return course_encode_video_frames(
            imgs_dir, video_path, fps, vcodec="libx264", crf=18, **kwargs
        )

    dataset_module.encode_video_frames = encode_h264
    return LeRobotDataset


def scene_config_hash():
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def check_episode(episode_dir, expected_config, config):
    result = read_json(episode_dir / "result.json")
    if not result or result.get("accepted_for_training") is not True:
        return None, "accepted_for_training is not true"
    if result.get("scene_config_sha256") != expected_config:
        return None, "scene_config_sha256 mismatch"
    if result.get("geometry_scale") != 1.5:
        return None, "geometry_scale is not 1.5"
    scene = read_json(episode_dir / "scene_config.json")
    if scene != config:
        return None, "scene_config.json differs from current config"
    if (result.get("source") != "camera_visual_servo"
            or result.get("camera_image_used_for_control") is not True
            or result.get("oracle_coordinates_in_control") is not False):
        return None, "episode is not camera-directed control without oracle coordinates"
    if result.get("screen_observed") is not False:
        return None, "camera-control provenance metadata is inconsistent"
    if result.get("control_observation") != ["observation.images.overhead", "observation.state"]:
        return None, "controller did not use only camera images and joint state"
    if result.get("recording_camera") is not True or result.get("mouth_teleported_during_episode") is not False:
        return None, "camera/provenance metadata does not match the task"
    tolerance = SUCCESS_RULES['center_tolerance_m'] if SUCCESS_RULES else .015
    if SUCCESS_RULES and (result.get('success_rules') != SUCCESS_RULES or
                          not result.get('final_placement_metrics',{}).get('success',False)):
        return None, 'outline v2 success rules or released placement audit did not pass'
    if result.get('tolerance_m') != tolerance or result.get('hold_time_sim_s') != 1.:
        return None, 'success settings do not match selected criteria'
    if not result.get('physical_success') or result.get('final_error_m',math.inf) > tolerance:
        return None, 'physical placement did not pass'
    if result.get('held_after_withdrawal_sim_s',0.) < 1.:
        return None, 'released hold is shorter than one simulation second'
    sync = result.get("synchronization") or {}
    if sync.get("quality_pass") is not True or sync.get("oracle_coordinates_in_model_input") is not False:
        return None, "synchronization/provenance audit did not pass"
    if sync.get("input_features") != ["observation.images.overhead", "observation.state"]:
        return None, "recorded model inputs include unexpected features"
    if sync.get("state_joint_names") != RAW_STATE_NAMES or sync.get("action_joint_names") != RAW_STATE_NAMES:
        return None, "state/action joint ordering differs from the recorded controller order"
    if sync.get("fps_requested") != FPS:
        return None, "recording cadence is not 10 fps"
    if sync.get("frames_written", 0) < 100:
        return None, "fewer than 100 synchronized frames"
    if sync.get("frames_received") != sync.get("frames_written") or sync.get("frames_rejected") != 0:
        return None, "camera recorder received or rejected incomplete frames"
    if sync.get("maximum_frame_interval_sim_s") is None or sync["maximum_frame_interval_sim_s"] > 0.11:
        return None, "camera frame gap exceeds 0.11 simulation seconds"
    if sync.get("maximum_nearest_state_offset_ms") is None or sync["maximum_nearest_state_offset_ms"] > 30:
        return None, "camera/state offset exceeds 30 ms"
    if config.get("target_xy_m") != result.get("goal_xy_m"):
        return None, "target differs from the scene configuration"

    frame_path = episode_dir / "frames.jsonl"
    try:
        rows = [json.loads(line) for line in frame_path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None, "frames.jsonl is missing or unreadable"
    if len(rows) != sync["frames_written"]:
        return None, "frame count differs from synchronization audit"
    sim_times = []
    state_offsets = []
    seen = set()
    for index, row in enumerate(rows):
        relative_image = row.get("image")
        if not relative_image or relative_image in seen:
            return None, "missing or duplicate image path"
        seen.add(relative_image)
        if row.get("frame_index") != index:
            return None, "frame indices are not contiguous"
        if len(row.get("observation.state", [])) != 5 or len(row.get("action", [])) != 5:
            return None, "state/action feature does not contain five joints"
        if not all(math.isfinite(float(v)) for v in row["observation.state"] + row["action"]):
            return None, "state/action contains a non-finite number"
        try:
            sim_times.append(int(row["sim_time_ns"]) / 1e9)
            state_offsets.append(abs(float(row["nearest_state_offset_ms"])))
        except (KeyError, TypeError, ValueError):
            return None, "frame is missing camera/state synchronization timestamps"
        path = episode_dir / relative_image
        if not path.is_file() or path.stat().st_size == 0:
            return None, "image is missing or empty: " + relative_image
        try:
            with Image.open(path) as image:
                if image.size != (640, 480):
                    return None, "image dimensions are not 640x480: " + relative_image
                image.verify()
        except Exception as exc:
            return None, "image decode failed for %s: %s" % (relative_image, exc)
    intervals = np.diff(sim_times)
    if len(intervals) != len(rows) - 1 or np.any(intervals <= 0) or float(np.max(intervals)) > 0.11:
        return None, "per-frame timestamps are missing, non-monotonic, or over 0.11 seconds apart"
    if max(state_offsets, default=math.inf) > 30:
        return None, "a recorded state is more than 30 ms from its camera frame"
    return {"result": result, "frames": rows}, None


def collect_episodes(expected_config, config, limit, deduplicate_starts=False):
    accepted = []
    excluded = []
    for episode_dir in sorted(DATA_ROOT.iterdir()):
        if not episode_dir.is_dir() or episode_dir.name == "attempt_logs":
            continue
        result = read_json(episode_dir / "result.json")
        if not result or result.get("accepted_for_training") is not True:
            continue
        item, reason = check_episode(episode_dir, expected_config, config)
        if item:
            if deduplicate_starts:
                start = item['result']['initial_mouth_xyz_m'][:2]
                duplicate = next((previous for previous in accepted
                                  if math.dist(start, previous[1]['initial_mouth_xyz_m'][:2]) < .001), None)
                if duplicate is not None:
                    excluded.append({'episode':episode_dir.name,
                                     'reason':'start within1mm of '+duplicate[0].name,
                                     'raw_acceptance_preserved':True})
                    continue
            accepted.append((episode_dir, item["result"], item["frames"]))
        else:
            excluded.append({"episode": episode_dir.name, "reason": reason})
    if len(accepted) < limit:
        raise RuntimeError("Need %d eligible episodes; found %d. Exclusions: %s" %
                           (limit, len(accepted), json.dumps(excluded, ensure_ascii=False)))
    accepted = accepted[:limit]
    seeds = [item[1].get("seed") for item in accepted]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Accepted episode set contains duplicate seeds")
    if any(seed in set(range(2026102200,2026102205)) | set(range(2026102300,2026102305)) for seed in seeds):
        raise RuntimeError("Reserved evaluation seed is present in demonstration sources")
    starts = [item[1]["initial_mouth_xyz_m"][:2] for item in accepted]
    for index, start in enumerate(starts):
        if any(math.dist(start, previous) < 0.001 for previous in starts[:index]):
            raise RuntimeError("Accepted episodes have duplicate starts within 1 mm")
    return accepted, excluded


def canonical(values):
    if len(values) != len(RAW_STATE_NAMES):
        raise ValueError("Expected the recorded five-joint vector")
    return np.asarray([values[i] for i in COURSE_INDEX_FROM_RAW], dtype=np.float32)


def make_features():
    joint_feature = {
        "dtype": "float32",
        "shape": (5,),
        "names": COURSE_STATE_NAMES,
    }
    return {
        "observation.images.overhead": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        },
        "observation.state": dict(joint_feature),
        # The lecture ACT configuration expects this field. It is a duplicate of
        # the observed joint state, not simulator coordinates or object pose.
        "observation.environment_state": dict(joint_feature),
        "action": dict(joint_feature),
    }


def episode_source_record(episode_index, episode_dir, result, rows):
    return {
        "episode_index": episode_index,
        "source_episode": episode_dir.name,
        **({"source_repo_relative":str(episode_dir.resolve().relative_to(ROOT)),
            "derived_segment":result["segment_provenance"]} if result.get("segment_provenance") else {}),
        "seed": result["seed"],
        "start_xy_m": result["initial_mouth_xyz_m"][:2],
        "goal_xy_m": result["goal_xy_m"],
        "final_error_m": result["final_error_m"],
        "released_hold_sim_s": result["held_after_withdrawal_sim_s"],
        "scene_config_sha256": result["scene_config_sha256"],
        "frame_count": len(rows),
        "state_action_joint_names": COURSE_STATE_NAMES,
        "source": result["source"],
        "screen_observed": result["screen_observed"],
    }


def make_split(episode_count, validation_ratio, split_seed, sources=None, preserve_from=None):
    if preserve_from:
        baseline = Path(preserve_from)
        previous = read_json(baseline / "splits.json")
        old_sources = read_json(baseline / "episode_sources.json")
        if not previous or not old_sources or not sources:
            raise RuntimeError("Preserving a split requires both source mappings and the previous split")
        old_names = [s['source_episode'] for s in old_sources]
        names = [s['source_episode'] for s in sources]
        old_train, old_val = previous['train_episode_indices'], previous['validation_episode_indices']
        if (len(set(old_names)) != len(old_names) or len(set(names)) != len(names)
                or len(names) != episode_count
                or len(old_train) + len(old_val) != len(old_names)
                or set(old_train) & set(old_val)
                or set(old_train) | set(old_val) != set(range(len(old_names)))):
            raise RuntimeError("Previous split or source mapping is invalid")
        if not set(old_names) <= set(names):
            raise RuntimeError("Expanded dataset is missing previous source episodes")
        index = {name: i for i, name in enumerate(names)}
        train = [index[old_names[i]] for i in old_train]
        validation = [index[old_names[i]] for i in old_val]
        # Keep the original validation set fixed. All new demonstrations are
        # training data; source names preserve membership even if indices move.
        train.extend(i for i, name in enumerate(names) if name not in set(old_names))
        return {
            'method': 'preserve previous source membership; append new episodes to training',
            'preserved_from': str(baseline),
            'preserved_split_sha256': hashlib.sha256((baseline / 'splits.json').read_bytes()).hexdigest(),
            'preserved_validation_source_episodes': [old_names[i] for i in old_val],
            'seed': split_seed, 'validation_ratio': len(validation) / episode_count,
            'train_episode_indices': sorted(train), 'validation_episode_indices': sorted(validation),
        }
    indexes = list(range(episode_count))
    random.Random(split_seed).shuffle(indexes)
    validation_count = max(1, round(episode_count * validation_ratio))
    validation = sorted(indexes[:validation_count])
    train = sorted(indexes[validation_count:])
    if set(train) & set(validation) or set(train) | set(validation) != set(range(episode_count)):
        raise AssertionError("Episode split is not a complete disjoint partition")
    payload = {
        "method": "deterministic random split by episode_index; frames never cross splits",
        "seed": split_seed,
        "validation_ratio": validation_ratio,
        "train_episode_indices": train,
        "validation_episode_indices": validation,
    }
    return payload


def verify_loader(LeRobotDataset, root, repo_id, split, expected_episode_count):
    all_data = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    if all_data.meta._version.base_version != "2.1":
        raise RuntimeError("Loaded dataset metadata is not LeRobot v2.1")
    if all_data.meta.total_episodes != expected_episode_count:
        raise RuntimeError("LeRobot metadata episode count mismatch")
    if all_data.meta.total_frames != len(all_data):
        raise RuntimeError("LeRobot metadata frame count mismatch")
    expected_all = list(range(expected_episode_count))
    if list(all_data.meta.episodes.keys()) != expected_all:
        raise RuntimeError("LeRobot episode metadata does not contain indices 0..N-1")
    source_records = read_json(root / "episode_sources.json")
    if not isinstance(source_records, list) or len(source_records) != expected_episode_count:
        raise RuntimeError("Source episode manifest does not match LeRobot episode count")
    if len(all_data.episode_data_index["from"]) != expected_episode_count:
        raise RuntimeError("LeRobot per-episode boundary index has an unexpected length")
    cursor = 0
    for episode_index, source in enumerate(source_records):
        start = int(all_data.episode_data_index["from"][episode_index])
        end = int(all_data.episode_data_index["to"][episode_index])
        if start != cursor or end <= start:
            raise RuntimeError("Episode %d has a gap, overlap, or empty frame range" % episode_index)
        if source.get("episode_index") != episode_index or source.get("frame_count") != end - start:
            raise RuntimeError("Episode %d boundary length differs from its source manifest" % episode_index)
        cursor = end
    if cursor != len(all_data):
        raise RuntimeError("Episode boundary ranges do not cover the complete LeRobot dataset")
    if all_data.meta.video_keys != ["observation.images.overhead"]:
        raise RuntimeError("Unexpected LeRobot camera keys: " + repr(all_data.meta.video_keys))
    if all_data.meta.image_keys:
        raise RuntimeError("Expected camera frames to use the lecture's v2.1 video representation")

    train_ids = split["train_episode_indices"]
    val_ids = split["validation_episode_indices"]
    if set(train_ids) & set(val_ids) or sorted(train_ids + val_ids) != expected_all:
        raise RuntimeError("Train/validation split overlaps or omits episodes")
    train_data = LeRobotDataset(repo_id, root=root, episodes=train_ids, video_backend="pyav")
    val_data = LeRobotDataset(repo_id, root=root, episodes=val_ids, video_backend="pyav")
    if train_data.num_episodes != len(train_ids) or val_data.num_episodes != len(val_ids):
        raise RuntimeError("LeRobot episode-subset loader returned an unexpected episode count")

    samples = []
    # Decode the official loader's video at each episode's beginning, middle,
    # and end. Checking every episode catches a broken or shifted video file
    # that aggregate metadata and a few representative episodes could miss.
    for episode_index in range(expected_episode_count):
        start = int(all_data.episode_data_index["from"][episode_index])
        end = int(all_data.episode_data_index["to"][episode_index])
        if not start < end <= len(all_data):
            raise RuntimeError("Invalid episode data boundaries for episode %d" % episode_index)
        for label, frame_index in (("first", start), ("middle", start + (end - start) // 2), ("last", end - 1)):
            item = all_data[frame_index]
            if int(item["episode_index"]) != episode_index:
                raise RuntimeError("LeRobot sample crossed an episode boundary")
            if int(item["frame_index"]) != frame_index - start:
                raise RuntimeError("LeRobot frame index does not restart at episode boundary")
            image = item["observation.images.overhead"]
            if tuple(image.shape) != (3, 480, 640):
                raise RuntimeError("LeRobot image shape is unexpected: " + repr(tuple(image.shape)))
            for key in ("observation.state", "observation.environment_state", "action"):
                if tuple(item[key].shape) != (5,) or not torch_is_finite(item[key]):
                    raise RuntimeError("LeRobot %s sample is invalid" % key)
            if not np.array_equal(item["observation.state"].numpy(), item["observation.environment_state"].numpy()):
                raise RuntimeError("Course environment_state must be an exact duplicate of joint state")
            samples.append({
                "episode_index": episode_index,
                "position": label,
                "global_frame_index": frame_index,
                "frame_index": int(item["frame_index"]),
                "image_shape": list(image.shape),
                "state_shape": list(item["observation.state"].shape),
                "action_shape": list(item["action"].shape),
                "task": item["task"],
            })

    # Load one sample from each split with the official dataset API.
    for name, subset, allowed in (("train", train_data, set(train_ids)), ("validation", val_data, set(val_ids))):
        if len(subset) == 0:
            raise RuntimeError("Empty %s split" % name)
        item = subset[0]
        if int(item["episode_index"]) not in allowed:
            raise RuntimeError("The %s loader returned an episode from the other split" % name)
        if tuple(item["observation.images.overhead"].shape) != (3, 480, 640):
            raise RuntimeError("The %s split did not decode its camera frame" % name)

    return {
        "codebase_version": "v2.1",
        "total_episodes": all_data.meta.total_episodes,
        "total_frames": all_data.meta.total_frames,
        "train_episode_count": len(train_ids),
        "validation_episode_count": len(val_ids),
        "train_validation_overlap": sorted(set(train_ids) & set(val_ids)),
        "feature_keys": sorted(all_data.features.keys()),
        "camera_keys": all_data.meta.camera_keys,
        "video_keys": all_data.meta.video_keys,
        "episode_boundary_samples": samples,
        "episode_boundaries_verified": expected_episode_count,
        "episode_video_samples_verified": len(samples),
        "split_loader_sample_checks": ["train", "validation"],
        "decoder": "pyav",
    }


def torch_is_finite(tensor):
    import torch
    return bool(torch.isfinite(tensor).all())


def convert(args):
    global DATA_ROOT, SUCCESS_RULES
    DATA_ROOT = Path(args.data_root).resolve()
    SUCCESS_RULES = None if args.legacy_position_criteria else load_rules()
    source_root = Path(args.lerobot_source)
    LeRobotDataset = load_sources(source_root)
    config = json.loads(CONFIG_PATH.read_text())
    config_hash = scene_config_hash()
    episodes, excluded = collect_episodes(config_hash, config, args.target_accepted, getattr(args, 'deduplicate_starts', False))
    if getattr(args, 'recovery_segments_root', None):
        if args.legacy_position_criteria or not args.preserve_split_from:
            raise ValueError('Recovery segments require strict v2 and preserved source split')
        from fukuwarai_recovery_segment_loader import load_segment
        for folder in sorted(Path(args.recovery_segments_root).iterdir()):
            if folder.is_dir():
                episodes.append(load_segment(folder, ROOT, config, config_hash))
    source_records = [episode_source_record(i, *episode) for i, episode in enumerate(episodes)]
    split = make_split(len(episodes), args.validation_ratio, args.split_seed,
                       source_records, args.preserve_split_from)
    output = Path(args.output_root)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        manifest = read_json(output / "conversion_manifest.json")
        if not manifest or manifest.get("scene_config_sha256") != config_hash:
            raise RuntimeError("Output exists without a matching conversion manifest; preserving it: " + str(output))
        if manifest.get('success_rules') != SUCCESS_RULES:
            raise RuntimeError('Preserving dataset from different success criteria')
        if manifest.get("source_episode_ids") != [item[0].name for item in episodes]:
            raise RuntimeError("Existing dataset episode list differs; preserving it: " + str(output))
        print("Existing conversion matches this source set; validating through LeRobotDataset.")
        existing_split = read_json(output / "splits.json")
        if existing_split is None:
            raise RuntimeError("Existing dataset is missing splits.json")
        if existing_split != split:
            raise RuntimeError("Requested split differs from existing dataset; preserving it")
        validation = verify_loader(LeRobotDataset, output, args.repo_id, split, len(episodes))
        (output / "loader_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        print(json.dumps(validation, indent=2))
        return

    staging = parent / (output.name + ".staging")
    if staging.exists():
        raise RuntimeError("Preserving existing partial staging output; choose a new --output-root: " + str(staging))
    reuse = None
    converted_count = 0
    if args.reuse_converted_from:
        baseline = Path(args.reuse_converted_from).resolve()
        baseline_manifest = read_json(baseline / "conversion_manifest.json")
        baseline_sources = read_json(baseline / "episode_sources.json")
        if (not baseline_manifest or not baseline_sources
                or baseline_manifest.get('success_rules') != SUCCESS_RULES
                or baseline_manifest.get('scene_config_sha256') != config_hash
                or baseline_manifest.get('course_source', {}).get('lerobot_commit') != LE_ROBOT_COMMIT
                or source_records[:len(baseline_sources)] != baseline_sources):
            raise RuntimeError('Reusable conversion must match the exact source episode prefix and pinned format')
        # Copy, never hardlink: appending metadata cannot mutate the baseline.
        staging.mkdir()
        for directory in ('data', 'meta', 'videos'):
            shutil.copytree(baseline / directory, staging / directory)
        hashes = {}
        for directory in ('data', 'videos'):
            for original in sorted((baseline / directory).rglob('*')):
                if original.is_file():
                    relative = original.relative_to(baseline)
                    digest = hashlib.sha256(original.read_bytes()).hexdigest()
                    if hashlib.sha256((staging / relative).read_bytes()).hexdigest() != digest:
                        raise RuntimeError('Reusable dataset copy differs: ' + str(relative))
                    hashes[str(relative)] = digest
        dataset = LeRobotDataset(args.repo_id, root=staging, video_backend='pyav')
        converted_count = len(baseline_sources)
        if dataset.meta.total_episodes != converted_count or dataset.fps != FPS:
            raise RuntimeError('Reusable conversion metadata disagrees with source mapping')
        reuse = {'source': str(baseline), 'episodes': converted_count,
                 'copied_file_sha256': hashes, 'baseline_modified': False}
        print('Reused %d existing episodes byte-for-byte; converting additions only.' % converted_count, flush=True)
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id, fps=FPS, root=staging,
            robot_type="crane_plus", features=make_features(), use_videos=True,
        )
    for episode_index in range(converted_count, len(episodes)):
        episode_dir, result, rows = episodes[episode_index]
        print("Converting episode %d/%d: %s (%d frames)" %
              (episode_index + 1, len(episodes), episode_dir.name, len(rows)), flush=True)
        for frame_index, row in enumerate(rows):
            image_path = episode_dir / row["image"]
            with Image.open(image_path) as loaded:
                image = loaded.convert("RGB")
                image.load()
            state = canonical(row["observation.state"])
            action = canonical(row["action"])
            dataset.add_frame({
                "observation.images.overhead": image,
                "observation.state": state,
                "observation.environment_state": state.copy(),
                "action": action,
                "task": TASK,
                # Let the pinned v2.1 writer derive its scalar timestamp from
                # the episode-local frame index and FPS. Passing an explicit
                # array validates per frame but stacks into (N, 1), which its
                # episode sync check does not accept.
            })
        dataset.save_episode()

    if dataset.meta.total_episodes != len(episodes):
        raise RuntimeError("LeRobot writer did not finalize all accepted episodes")
    if reuse:
        for relative, digest in reuse['copied_file_sha256'].items():
            if (hashlib.sha256((Path(reuse['source']) / relative).read_bytes()).hexdigest() != digest
                    or hashlib.sha256((staging / relative).read_bytes()).hexdigest() != digest):
                raise RuntimeError('Appending changed a protected baseline episode: ' + relative)
    (staging / "splits.json").write_text(json.dumps(split, indent=2) + "\n")
    (staging / "episode_sources.json").write_text(json.dumps(source_records, indent=2) + "\n")
    conversion = {
        "dataset": "local LeRobot v2.1",
        "repo_id": args.repo_id,
        "scene_config_sha256": config_hash,
        "geometry_scale": 1.5,
        "success_tolerance_m": SUCCESS_RULES["center_tolerance_m"] if SUCCESS_RULES else .015,
        "success_rules": SUCCESS_RULES,
        "source_data_root": str(DATA_ROOT.relative_to(ROOT)) if DATA_ROOT.is_relative_to(ROOT) else str(DATA_ROOT),
        "released_hold_sim_s": 1.0,
        "fps": FPS,
        "source": "camera_visual_servo; camera_image_used_for_control=true; oracle_coordinates_in_control=false",
        "input_features": ["observation.images.overhead", "observation.state", "observation.environment_state"],
        "output_features": ["action"],
        "video_encoding": {
            "container": "mp4",
            "codec": "libx264",
            "crf": 18,
            "reason": "The container ffmpeg build lacks the pinned v2.1 source default libsvtav1 encoder.",
        },
        "oracle_coordinates_in_model_input": False,
        "source_episode_ids": [item[0].name for item in episodes],
        "reused_conversion": reuse,
        "excluded_accepted_flagged_episodes": excluded,
        "course_source": {
            "pai_training_commit": PAI_TRAINING_COMMIT,
            "lerobot_commit": LE_ROBOT_COMMIT,
            "lerobot_dataset_source_sha256": LE_ROBOT_DATASET_SOURCE_SHA256,
        },
        "converted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (staging / "conversion_manifest.json").write_text(json.dumps(conversion, indent=2) + "\n")
    os.replace(str(staging), str(output))

    validation = verify_loader(LeRobotDataset, output, args.repo_id, split, len(episodes))
    (output / "loader_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({"output": str(output), "conversion": conversion, "loader_validation": validation}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=DATA_ROOT)
    parser.add_argument('--recovery-segments-root',type=Path,default=None)
    parser.add_argument('--legacy-position-criteria',action='store_true',help='Explicit historical conversion only; not v2 demonstrations')
    parser.add_argument('--deduplicate-starts', action='store_true', help='Select earliest eligible source per1mm neighborhood; retain raw labels and record exclusions')
    parser.add_argument("--target-accepted", type=int, default=TARGET_ACCEPTED)
    parser.add_argument("--repo-id", default="crane_plus_fukuwarai")
    parser.add_argument("--output-root", default="data/lerobot_v2.1/crane_plus_fukuwarai_outline_v2")
    parser.add_argument("--lerobot-source", default="/tmp/fukuwarai-course-lerobot")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--preserve-split-from", type=Path,
                        help="Keep previous source episode split membership; append new episodes to training")
    parser.add_argument("--reuse-converted-from", type=Path,
                        help="Copy an exact converted source prefix; encode only additional episodes")
    args = parser.parse_args()
    if args.target_accepted < 1:
        parser.error("target-accepted must be positive")
    if not 0.0 < args.validation_ratio < 1.0:
        parser.error("validation-ratio must be between 0 and 1")
    convert(args)


if __name__ == "__main__":
    main()
