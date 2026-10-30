#!/usr/bin/env python3
"""Resume camera-directed collection until a target number of accepted episodes.

Run on the Mac host from the repository. Each trial runs serially in the existing
pai-advanced1 container; this script never starts or stops Gazebo.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "expert_images_v2"
CONFIG_PATH = ROOT / "config" / "fukuwarai.json"
sys.path.insert(0, str(ROOT / "scripts"))
from reset_mouth import sample_start  # noqa: E402
from fukuwarai_success import load_rules

CAMPAIGN = "fukuwarai-mouth-scale-1.5-camera-visual-outline-v2"
COURSE_CONTAINER_ROOT = "/workspace"
MANIFEST_PATH = DATA_ROOT / "visual_collection_manifest.json"
LOCK_PATH = DATA_ROOT / ".visual-collection.lock"


def load_config():
    raw = CONFIG_PATH.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def image_audit(episode_dir, result):
    sync = result.get("synchronization") or {}
    frames_path = episode_dir / "frames.jsonl"
    if sync.get("quality_pass") is not True:
        return False, "synchronization quality_pass is not true"
    if sync.get("fps_requested") != 10:
        return False, "recording cadence is not 10 fps"
    if sync.get("frames_written", 0) < 100:
        return False, "fewer than 100 synchronized frames"
    if sync.get("frames_received") != sync.get("frames_written") or sync.get("frames_rejected") != 0:
        return False, "camera frames were missing, rejected, or not synchronized"
    if sync.get("maximum_frame_interval_sim_s") is None or sync["maximum_frame_interval_sim_s"] > 0.11:
        return False, "camera frame gap exceeds 0.11 sim seconds"
    if sync.get("maximum_nearest_state_offset_ms") is None or sync["maximum_nearest_state_offset_ms"] > 30:
        return False, "camera/state offset exceeds 30 ms"
    if not frames_path.is_file():
        return False, "frames.jsonl is missing"
    try:
        rows = [json.loads(line) for line in frames_path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False, "frames.jsonl is unreadable"
    expected = sync.get("frames_written")
    if len(rows) != expected:
        return False, "frame manifest count does not match sync audit"
    if sync.get("input_features") != ["observation.images.overhead", "observation.state"]:
        return False, "recorded model inputs include unexpected features"
    times = []
    state_offsets = []
    seen = set()
    for index, row in enumerate(rows):
        image = row.get("image")
        if not image or image in seen:
            return False, "missing or duplicate image path in frame manifest"
        seen.add(image)
        image_path = episode_dir / image
        if not image_path.is_file() or image_path.stat().st_size == 0:
            return False, "missing or empty image: " + str(image)
        if row.get("frame_index") != index:
            return False, "non-contiguous frame indices"
        if len(row.get("observation.state", [])) != 5 or len(row.get("action", [])) != 5:
            return False, "state/action vector does not have 5 joint values"
        for key in ("observation.state", "action"):
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in row[key]):
                return False, key + " contains a non-finite value"
        try:
            times.append(int(row["sim_time_ns"]) / 1e9)
            state_offsets.append(abs(float(row["nearest_state_offset_ms"])))
        except (KeyError, TypeError, ValueError):
            return False, "frame is missing camera/state synchronization timestamps"
    intervals = [right - left for left, right in zip(times, times[1:])]
    if not intervals or min(intervals) <= 0 or max(intervals) > 0.11:
        return False, "per-frame timestamps are missing, non-monotonic, or over 0.11 seconds apart"
    if max(state_offsets, default=math.inf) > 30:
        return False, "a recorded state is more than 30 ms from its camera frame"
    return True, None


def inspect_episode(episode_dir, config_hash):
    result = read_json(episode_dir / "result.json")
    if result is None:
        return {"eligible": False, "reason": "result.json missing or unreadable", "result": None}
    if result.get("accepted_for_training") is not True:
        reason = result.get("error")
        if not reason and not result.get("physical_success", result.get("success", False)):
            reason = "physical success condition not met"
        if not reason and not (result.get("synchronization") or {}).get("quality_pass", False):
            reason = "recording quality failed"
        return {"eligible": False, "reason": reason or "accepted_for_training is not true", "result": result}
    if result.get("scene_config_sha256") != config_hash:
        return {"eligible": False, "reason": "scene_config_sha256 mismatch", "result": result}
    if result.get("geometry_scale") != 1.5:
        return {"eligible": False, "reason": "geometry_scale is not 1.5", "result": result}
    if (result.get("source") != "camera_visual_servo"
            or result.get("camera_image_used_for_control") is not True
            or result.get("oracle_coordinates_in_control") is not False):
        return {"eligible": False, "reason": "source provenance does not match camera visual control", "result": result}
    if (result.get("screen_observed") is not False
            or result.get("control_observation") != ["observation.images.overhead", "observation.state"]):
        return {"eligible": False, "reason": "control inputs are not camera image plus joint state", "result": result}
    if result.get("recording_camera") is not True:
        return {"eligible": False, "reason": "camera recording flag is not true", "result": result}
    rules = load_rules()
    if result.get('success_rules') != rules:
        return {'eligible':False,'reason':'outline v2 success rules missing or changed','result':result}
    if result.get("tolerance_m") != rules['center_tolerance_m'] or result.get("hold_time_sim_s") != rules['hold_time_sim_s']:
        return {"eligible": False, "reason": "success settings do not match this collection", "result": result}
    if (result.get("physical_success") is not True
            or result.get("final_error_m", math.inf) > rules["center_tolerance_m"]
            or result.get("held_after_withdrawal_sim_s", 0.0) < rules['hold_time_sim_s']
            or not result.get('final_placement_metrics',{}).get('success',False)):
        return {"eligible": False, "reason": "released position or one-second hold failed", "result": result}
    scene = read_json(episode_dir / "scene_config.json")
    scene_hash = hashlib.sha256((json.dumps(scene, indent=2) + "\n").encode()).hexdigest() if scene is not None else None
    if scene is None or scene_hash != config_hash:
        return {"eligible": False, "reason": "scene_config.json missing or unreadable", "result": result}
    visual_log = episode_dir / "visual_control.jsonl"
    try:
        visual_rows = [json.loads(line) for line in visual_log.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return {"eligible": False, "reason": "visual camera-control trace is missing or unreadable", "result": result}
    if not visual_rows or any("mouth_pixel_xy" not in row or "visual_error_estimate_m" not in row
                              for row in visual_rows):
        return {"eligible": False, "reason": "visual camera-control trace is incomplete", "result": result}
    ok, reason = image_audit(episode_dir, result)
    if not ok:
        return {"eligible": False, "reason": reason, "result": result}
    return {"eligible": True, "reason": None, "result": result}


def scan_episodes(config_hash):
    episodes = {}
    used_seeds = set()
    positions = []
    for episode_dir in sorted(DATA_ROOT.iterdir()):
        if not episode_dir.is_dir():
            continue
        result = read_json(episode_dir / "result.json")
        if result is None:
            continue
        seed = result.get("seed")
        if isinstance(seed, int):
            used_seeds.add(seed)
        position = result.get("initial_mouth_xyz_m")
        if isinstance(position, list) and len(position) >= 2:
            positions.append([float(position[0]), float(position[1])])
        episodes[episode_dir.name] = inspect_episode(episode_dir, config_hash)
    return episodes, used_seeds, positions


def new_manifest(config_hash, baseline):
    visual_attempts = []
    baseline_ids = []
    for episode_id, inspected in sorted(baseline.items()):
        result = inspected.get("result") or {}
        same_campaign = (
            result.get("source") == "camera_visual_servo"
            and result.get("scene_config_sha256") == config_hash
            and result.get("geometry_scale") == 1.5
            and result.get("success_rules") == load_rules()
            and result.get("tolerance_m") == load_rules()["center_tolerance_m"]
            and result.get("hold_time_sim_s") == 1.0
        )
        if not same_campaign:
            baseline_ids.append(episode_id)
            continue
        visual_attempts.append({
            "attempt_utc": datetime.now(timezone.utc).isoformat(),
            "seed": result.get("seed"),
            "result_directory": episode_id,
            "exit_code": None,
            "accepted_for_training": bool(inspected.get("eligible")),
            "physical_success": result.get("physical_success", result.get("success")),
            "recovered_from_visual_pilot": True,
            "reason": inspected.get("reason"),
        })
    return {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "scene_config_sha256": config_hash,
        "geometry_scale": 1.5,
        "target_xy_m": None,
        "position_tolerance_m": load_rules()["center_tolerance_m"],
        "success_rules": load_rules(),
        "hold_time_sim_s": 1.0,
        "baseline_episode_ids": baseline_ids,
        "attempts": visual_attempts,
        "next_seed_candidate": 10000,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_manifest(manifest):
    temp = MANIFEST_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temp, MANIFEST_PATH)


def reconcile_manifest(manifest, episodes):
    known = {item.get("result_directory") for item in manifest["attempts"] if item.get("result_directory")}
    baseline = set(manifest.get("baseline_episode_ids", []))
    for episode_id, inspected in episodes.items():
        if episode_id in baseline or episode_id in known:
            continue
        result = inspected.get("result") or {}
        manifest["attempts"].append({
            "attempt_utc": datetime.now(timezone.utc).isoformat(),
            "seed": result.get("seed"),
            "result_directory": episode_id,
            "exit_code": None,
            "accepted_for_training": bool(inspected.get("eligible")),
            "recovered_after_interruption": True,
            "reason": inspected.get("reason"),
        })


def accepted_ids(episodes):
    return [episode_id for episode_id, value in episodes.items() if value.get("eligible")]


def choose_seed(config, used_seeds, positions, start_seed):
    seed = max(0, start_seed)
    while True:
        candidate = sample_start(config, seed)
        xy = candidate[:2]
        too_close = any(math.dist(xy, p) < 0.001 for p in positions)
        if seed not in used_seeds and not too_close:
            return seed, candidate
        seed += 1


def container_is_ready(container):
    state = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", container],
        text=True, capture_output=True,
    )
    if state.returncode or state.stdout.strip() != "true":
        raise RuntimeError("Docker container is not running: " + container)
    check = subprocess.run(
        ["docker", "exec", "-u", "ubuntu", container, "bash", "-lc",
         "export ROS_LOG_DIR=$(mktemp -d /tmp/fukuwarai-batch-check.XXXXXX) && "
         "source /opt/ros/humble/setup.bash && ros2 topic list"],
        text=True, capture_output=True, timeout=20,
    )
    if check.returncode:
        raise RuntimeError("Cannot read ROS topics in the existing container: " + check.stderr.strip())
    topics = set(check.stdout.splitlines())
    required = {"/clock", "/joint_states", "/fukuwarai/overhead/image/compressed",
                "/crane_plus_arm_controller/joint_trajectory",
                "/crane_plus_gripper_controller/joint_trajectory"}
    missing = sorted(required - topics)
    if missing:
        raise RuntimeError("Existing simulator is missing expected topics: " + ", ".join(missing))


def assert_no_arm_publishers(container):
    """Refuse to collect while a teleop or other node can command either arm."""
    topics = ("/crane_plus_arm_controller/joint_trajectory",
              "/crane_plus_gripper_controller/joint_trajectory")
    for topic in topics:
        check = subprocess.run(
            ["docker", "exec", "-u", "ubuntu", container, "bash", "-lc",
             "export ROS_LOG_DIR=$(mktemp -d /tmp/fukuwarai-publisher-check.XXXXXX) && "
             "source /opt/ros/humble/setup.bash && ros2 topic info --verbose " + topic],
            text=True, capture_output=True, timeout=20,
        )
        if check.returncode:
            raise RuntimeError("Cannot inspect arm command publishers on %s: %s" %
                               (topic, check.stderr.strip()))
        count_lines = [line for line in check.stdout.splitlines()
                       if line.strip().startswith("Publisher count:")]
        if len(count_lines) != 1:
            raise RuntimeError("Could not determine publisher count for " + topic)
        count = int(count_lines[0].split(":", 1)[1].strip())
        if count:
            raise RuntimeError("Refusing to collect while %d node(s) publish to %s" % (count, topic))


def run_one(container, seed, predicted_position, attempt_number):
    assert_no_arm_publishers(container)
    log_dir = DATA_ROOT / "attempt_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ("attempt_%04d_seed_%d.log" % (attempt_number, seed))
    before = {path.name for path in DATA_ROOT.iterdir() if path.is_dir() and path.name.startswith("202")}
    command = ["docker", "exec", "-u", "ubuntu", container, "bash",
               COURSE_CONTAINER_ROOT + "/scripts/collect_visual_demo.sh", "--precise", "--seed", str(seed),
               "--output-root", COURSE_CONTAINER_ROOT + "/" + str(DATA_ROOT.relative_to(ROOT))]
    print("\n=== attempt %d, seed %d, predicted start %s ===" %
          (attempt_number, seed, json.dumps(predicted_position[:2])))
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        exit_code = process.wait()
    after = {path.name for path in DATA_ROOT.iterdir() if path.is_dir() and path.name.startswith("202")}
    created = sorted(after - before)
    episode_id = created[-1] if created else None
    return exit_code, episode_id, str(log_path)


def main():
    global DATA_ROOT, MANIFEST_PATH, LOCK_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="pai-advanced1")
    parser.add_argument('--output-root',type=Path,default=DATA_ROOT)
    parser.add_argument("--target-accepted", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="Stop after this many new trials in this invocation; useful for the first 10-trial review.")
    parser.add_argument("--start-seed", type=int, default=None,
                        help="Override the next seed candidate; otherwise resume the manifest's seed sequence.")
    args = parser.parse_args()
    if args.target_accepted < 1 or (args.max_attempts is not None and args.max_attempts < 1):
        parser.error("target and max-attempts must be positive")
    DATA_ROOT = args.output_root.resolve()
    if not DATA_ROOT.is_relative_to(ROOT):
        parser.error('output-root must be inside this repository')
    MANIFEST_PATH = DATA_ROOT/'visual_collection_manifest.json'
    LOCK_PATH = DATA_ROOT/'.visual-collection.lock'
    DATA_ROOT.mkdir(parents=True,exist_ok=True)
    prior = read_json(MANIFEST_PATH)
    if prior is not None and prior.get('campaign') != CAMPAIGN:
        parser.error('Preserve the legacy collection; choose a fresh v2 output-root')

    with LOCK_PATH.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another batch collector owns " + str(LOCK_PATH))
        config, config_hash = load_config()
        expected_hash = config_hash
        manifest = read_json(MANIFEST_PATH)
        episodes, used_seeds, positions = scan_episodes(config_hash)
        if manifest is None:
            manifest = new_manifest(config_hash, episodes)
        if (manifest.get("campaign") != CAMPAIGN or manifest.get("scene_config_sha256") != expected_hash
                or manifest.get("success_rules") != load_rules()):
            raise SystemExit("visual_collection_manifest.json belongs to another configuration; preserve it and use a fresh output root")
        manifest["target_xy_m"] = config["target_xy_m"]
        manifest["target_accepted_for_training"] = args.target_accepted
        reconcile_manifest(manifest, episodes)
        save_manifest(manifest)
        container_is_ready(args.container)

        batch_attempts = 0
        while len(accepted_ids(episodes)) < args.target_accepted:
            if args.max_attempts is not None and batch_attempts >= args.max_attempts:
                break
            start_seed = args.start_seed if args.start_seed is not None else int(manifest.get("next_seed_candidate", 10000))
            seed, predicted = choose_seed(config, used_seeds, positions, start_seed)
            attempt_number = len(manifest["attempts"]) + 1
            attempt_utc = datetime.now(timezone.utc).isoformat()
            exit_code, episode_id, log_path = run_one(args.container, seed, predicted, attempt_number)
            used_seeds.add(seed)
            positions.append(predicted[:2])
            batch_attempts += 1
            inspected = episodes.get(episode_id) if episode_id else None
            if episode_id:
                inspected = inspect_episode(DATA_ROOT / episode_id, expected_hash)
                episodes[episode_id] = inspected
            result = (inspected or {}).get("result") or {}
            actual = result.get("initial_mouth_xyz_m")
            record = {
                "attempt_utc": attempt_utc,
                "seed": seed,
                "predicted_initial_mouth_xyz_m": predicted,
                "actual_initial_mouth_xyz_m": actual,
                "result_directory": episode_id,
                "exit_code": exit_code,
                "accepted_for_training": bool((inspected or {}).get("eligible")),
                "physical_success": result.get("physical_success", result.get("success")),
                "reason": (inspected or {}).get("reason") if episode_id else "collector did not create an episode result",
                "log": str(Path(log_path).relative_to(ROOT)),
            }
            manifest["attempts"].append(record)
            manifest["next_seed_candidate"] = seed + 1
            save_manifest(manifest)
            if actual and math.dist(actual[:2], predicted[:2]) > 0.0005:
                print("WARNING: recorded start differs from planned seeded position by more than 0.5 mm")
            if record["accepted_for_training"]:
                print("ACCEPTED %s (%d/%d)" %
                      (episode_id, len(accepted_ids(episodes)), args.target_accepted))
            else:
                print("NOT ACCEPTED %s: %s" % (episode_id or "no episode", record["reason"]))
            if exit_code not in (0, 1):
                print("Collector exited with status %d; the outcome is recorded and collection will continue." % exit_code)

        reasons = Counter(item.get("reason") or "accepted" for item in manifest["attempts"])
        print(json.dumps({
            "campaign": CAMPAIGN,
            "scene_config_sha256": expected_hash,
            "target_accepted_for_training": args.target_accepted,
            "accepted_for_training": len(accepted_ids(episodes)),
            "eligible_episode_ids": accepted_ids(episodes),
            "campaign_attempts": len(manifest["attempts"]),
            "attempts_this_invocation": batch_attempts,
            "remaining": max(0, args.target_accepted - len(accepted_ids(episodes))),
            "attempt_reason_counts": dict(reasons),
            "manifest": str(MANIFEST_PATH),
        }, indent=2))
    return 0 if len(accepted_ids(episodes)) >= args.target_accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
