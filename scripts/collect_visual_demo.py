#!/usr/bin/env python3
"""Collect one camera-observed, image-based visual-servo episode.

The mouth's simulator pose is read only by the final evaluator. All movement
decisions are derived from the overhead camera image and current arm joints.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import rclpy
from rclpy.serialization import serialize_message
from builtin_interfaces.msg import Duration
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from collect_expert_demo import ARM, ARM_TOPIC, HAND, HAND_TOPIC, Expert, PoseStream, kinematics

ROOT = Path(__file__).resolve().parents[1]
# Center of the dashed mouth guide, measured in the fixed 640x480 camera image.
# This is an image-space target; task object poses are not used by the controller.
TARGET_PIXEL = np.array([320.0, 270.0], dtype=float)
POST_SUCCESS_RECORDING_S = 3.0


class CameraVisualExpert(Expert):
    def __init__(self, stream, config):
        super().__init__(stream, config)
        self.image_sequence = 0
        self.visual_rows = None
        self.last_detection = None
        self.precise = False

    def on_image(self, msg):
        self.image_sequence += 1
        super().on_image(msg)

    def wait_for_visible_mouth(self, minimum_image_sim_s, timeout=30.0):
        """Wait for rendering after entity recreation, before recording starts."""
        deadline = time.monotonic() + timeout
        checked_sequence = -1
        reason = "No fresh overhead image arrived after the mouth reset"
        while time.monotonic() < deadline:
            self.spin()
            if self.latest_image is None or self.image_sequence == checked_sequence:
                continue
            checked_sequence = self.image_sequence
            stamp = self.latest_image.header.stamp
            if stamp.sec + stamp.nanosec / 1e9 < minimum_image_sim_s:
                continue
            try:
                return self.compute(self.detect_mouth)
            except RuntimeError as exc:
                # A fresh simulation timestamp can precede the rendering of
                # the recreated visual. Retry only new camera observations.
                reason = str(exc)
        raise RuntimeError("Mouth camera preflight timed out: " + reason)

    def stage(self, name):
        self.phase = name
        self.recorder.phase = name
        msg = String(data=json.dumps({"phase": name, "controller": "camera_visual_servo"}))
        self.phase_pub.publish(msg)
        self.record("/expert/task_state", msg)
        print(f"{self.sim:.3f}: {name}", flush=True)

    def detect_mouth(self):
        """Return the solid mouth piece's pixel center from the newest JPEG."""
        msg = self.latest_image
        if msg is None or not msg.data:
            raise RuntimeError("No overhead camera image is available")
        encoded = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (480, 640):
            raise RuntimeError("Overhead camera image is missing or not 640x480")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        # A dark red eye can touch the mouth silhouette at low-value pixels;
        # filter those pixels while retaining the bright red mouth material.
        mask = (((hue <= 17) | (hue >= 168)) & (saturation >= 90) & (value >= 70)).astype(np.uint8) * 255
        # The sampled board region contains the mouth; this excludes the eyes and cheeks.
        mask[:200, :] = 0
        mask[350:, :] = 0
        mask[:, :230] = 0
        mask[:, 410:] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for label in range(1, count):
            x, y, width, height, area = map(int, stats[label])
            # The part can rotate during contact, so keep the area and ROI
            # checks while accepting tall and near-square silhouettes too.
            aspect = width / max(1, height)
            if 450 <= area <= 4500 and width >= 16 and height >= 16 and 0.25 <= aspect <= 4.0:
                candidates.append((area, label, x, y, width, height, centroids[label]))
        if not candidates:
            raise RuntimeError("Camera vision could not identify the mouth piece")
        area, label, x, y, width, height, center = max(candidates, key=lambda item: item[0])
        centroid = np.asarray(center, dtype=float)
        bbox_center = np.array([x + width / 2.0, y + height / 2.0], dtype=float)
        aspect = width / max(1, height)
        center_offset_px = float(np.linalg.norm(bbox_center - centroid))
        ys, xs = np.nonzero(labels == label)
        xy = np.column_stack((xs, ys)).astype(float)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(xy.T))
        principal = eigenvectors[:, int(np.argmax(eigenvalues))]
        principal_axis_angle = math.atan2(float(principal[1]), float(principal[0]))
        axis_confidence = float(max(eigenvalues) / max(float(min(eigenvalues)), 1e-9))
        # The box center is stable for the horizontal authored silhouette.
        # Use the mask centroid for rotations and asymmetric masks whose
        # bounding box is pulled off-center by a few stray red pixels.
        center_estimator = "mask_centroid" if aspect < 1.85 or center_offset_px > 2.0 else "bbox_center"
        point = centroid if center_estimator == "mask_centroid" else bbox_center
        if np.linalg.norm(point - TARGET_PIXEL) > 100:
            raise RuntimeError("Detected red component is outside the expected mouth region")
        self.last_detection = {
            "pixel": point.tolist(), "center_estimator": center_estimator,
            "bbox_center_px": bbox_center.tolist(), "mask_centroid_px": centroid.tolist(), "area_px": area,
            "bbox_xywh": [x, y, width, height],
            "principal_axis_image_angle_rad": principal_axis_angle,
            "principal_axis_eigenvalue_ratio": axis_confidence,
            "camera_sim_time_ns": int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec),
        }
        if getattr(self, 'precise', False):
            from fukuwarai_image_pose import MouthImagePose
            if not hasattr(self,'pose_estimator'):
                self.pose_estimator = MouthImagePose(self.config)
            pose = self.pose_estimator.estimate(image,(labels==label).astype(np.uint8))
            if getattr(self, 'subpixel_pose', False):
                from fukuwarai_subpixel_pose import refine_pose
                refined = refine_pose((labels==label).astype(np.uint8), self.pose_estimator.xy,
                                      self.pose_estimator.scale, pose)
                self.last_detection['coarse_pose'] = pose.copy()
                self.last_detection['subpixel_silhouette_iou'] = refined['silhouette_iou']
                # A local binary-mask refinement can be worse on shaded or
                # partly hidden lips. Retain the already validated coarse
                # detector in that case; never use a low-overlap refinement.
                use_refined = refined['silhouette_iou'] >= .95
                self.last_detection['subpixel_refinement_applied'] = use_refined
                if use_refined:
                    pose.update(pixel=refined['pixel'], yaw_rad=refined['yaw_rad'])
            point = np.array(pose['pixel'])
            self.last_detection.update(pixel=point.tolist(),center_estimator='authored_silhouette_fit',
                                       fitted_yaw_rad=pose['yaw_rad'],silhouette_confidence=pose['confidence'])
        return point

    def cartesian_xy_path(self, kine, start_xy, end_xy, height, seconds):
        """Execute one continuous, camera-planned horizontal push trajectory."""
        start_xy = np.asarray(start_xy, dtype=float)
        end_xy = np.asarray(end_xy, dtype=float)
        distance = float(np.linalg.norm(end_xy - start_xy))
        steps = max(2, int(math.ceil(distance / 0.003)))
        previous = [self.joints[n] for n in ARM]
        trajectory = JointTrajectory(joint_names=ARM)
        waypoints = []
        for index in range(1, steps + 1):
            fraction = index / steps
            xy = start_xy + (end_xy - start_xy) * fraction
            q = self.compute(kine.solve, np.r_[xy, height], previous)
            if min(kine.minimum_z(q).values()) < self.config["board_surface_z_m"] + 0.001:
                raise RuntimeError("Continuous camera-planned push would intersect the board")
            previous = q
            t = seconds * fraction
            trajectory.points.append(JointTrajectoryPoint(
                positions=list(map(float, q)),
                time_from_start=Duration(sec=int(t), nanosec=int((t % 1) * 1e9)),
            ))
            waypoints.append((t, list(map(float, q))))
        if getattr(self, 'shared_joint_path', False):
            seconds = self.issue_shared_path(ARM, waypoints, kine)
        else:
            self.recorder.plan(int(self.sim * 1e9), ARM, [self.joints[n] for n in ARM], waypoints)
            self.wait_for_controller(self.arm)
            self.arm.publish(trajectory)
            self.record(ARM_TOPIC, trajectory)
        self.wait(seconds + 0.1)
        settle_deadline = self.sim + 1.0
        while max(abs(self.joints[n] - p) for n, p in zip(ARM, previous)) > 0.04 and self.sim < settle_deadline:
            self.wait(0.05)
        if max(abs(self.joints[n] - p) for n, p in zip(ARM, previous)) > 0.09:
            raise RuntimeError("Continuous push trajectory joint tracking error")


class CameraCalibration:
    """Pinhole projection from image pixels to the known board plane."""
    def __init__(self, config):
        camera = config["observation_camera"]
        pose = camera["pose_m_rad"]
        self.camera_x, self.camera_y, self.camera_z = map(float, pose[:3])
        self.width, self.height = int(camera["width"]), int(camera["height"])
        self.cx, self.cy = self.width / 2.0, self.height / 2.0
        self.focal_px = self.width / (2.0 * math.tan(float(camera["horizontal_fov_rad"]) / 2.0))
        self.depth = self.camera_z - float(config["board_surface_z_m"])
        self.pixels_per_meter = self.focal_px / self.depth
        if self.depth <= 0:
            raise ValueError("Camera calibration is not above the board plane")

    def pixel_to_world(self, pixel):
        u, v = map(float, pixel)
        # For the fixed downward-facing camera, image-up is world +x and
        # image-left is world +y. Only detected image pixels enter this map.
        return np.array([
            self.camera_x + (self.cy - v) / self.pixels_per_meter,
            self.camera_y - (u - self.cx) / self.pixels_per_meter,
        ])

    def pixel_error_to_world(self, error_pixel):
        du, dv = map(float, error_pixel)
        return np.array([-dv / self.pixels_per_meter, -du / self.pixels_per_meter])


def select_contact(node, kine, mouth_xy, direction, config, footprint):
    """Find a near-radial, reachable camera-derived contact approach."""
    # Keep a positive 2 mm clearance while allowing starts near the edge of
    # the arm's reachable workspace; the earlier 4 mm margin rejected valid
    # camera-observed starts before any movement.
    height = float(config["board_surface_z_m"]) + 0.003
    # Try the direct image-to-goal direction first, then bounded deviations
    # only when the arm cannot safely reach that side of the part.
    angles = (0, -15, 15, -30, 30, -45, 45, -60, 60)
    for angle_deg in angles:
        angle = math.radians(angle_deg)
        candidate_direction = np.array([
            math.cos(angle) * direction[0] - math.sin(angle) * direction[1],
            math.sin(angle) * direction[0] + math.cos(angle) * direction[1],
        ])
        alignment = float(np.dot(candidate_direction, direction))
        # Cap fallback deviation at 45 degrees and never push against either
        # component of the camera-derived goal direction. This excludes the
        # 60-degree near-tangential paths that increased error in pilot trials.
        if alignment < 0.707 or np.any(candidate_direction * direction < -1e-6):
            continue
        support = collision_support(footprint, candidate_direction)
        minimum_offset = support + 0.002
        offsets = list(np.arange(minimum_offset, 0.045, 0.002))
        if not offsets or offsets[-1] < 0.045 - 1e-9:
            offsets.append(0.045)
        for offset in offsets:
            tool_xy = mouth_xy - candidate_direction * offset
            try:
                q_above = node.compute(kine.solve, np.r_[tool_xy, height + 0.06])
                q_low = node.compute(kine.solve, np.r_[tool_xy, height], q_above)
            except Exception:
                continue
            if min(kine.minimum_z(q_above).values()) < config["board_surface_z_m"] + 0.001:
                continue
            if min(kine.minimum_z(q_low).values()) < config["board_surface_z_m"] + 0.001:
                continue
            # Retraction to the high home pose keeps the camera's view clear
            # and checks the return path above the board before commanding it.
            home = np.zeros(4)
            return_clearance = min(
                min(kine.minimum_z(q_above + (home - q_above) * alpha).values())
                for alpha in np.linspace(0.0, 1.0, 11)
            )
            if return_clearance < config["board_surface_z_m"] + 0.001:
                continue
            return tool_xy, q_above, q_low, candidate_direction, offset, support
    raise RuntimeError("No reachable safe approach pose for the camera-detected mouth")


def load_collision_footprint():
    """Read the authored lip silhouette used by Gazebo's collision mesh."""
    root = ET.parse(ROOT / "models/fukuwarai_mouth/meshes/lips_collision.dae").getroot()
    source = next(element for element in root.iter()
                  if element.tag.endswith("source") and element.get("id") == "positions")
    array = next(element for element in source if element.tag.endswith("float_array"))
    vertices = np.fromstring(array.text, sep=" ").reshape(-1, 3)[:, :2]
    return np.unique(vertices, axis=0)


def collision_support(footprint, direction):
    return float(np.max(-footprint @ direction))


def support_radius(footprint, direction):
    # The first visible motion occurred with the TCP about 2 mm inside the
    # mesh support boundary; use this camera-calibrated contact estimate for
    # step sizing, while approach poses retain a positive mesh clearance.
    return max(0.0, collision_support(footprint, direction) - 0.002)


def camera_axis_delta_to_world_yaw(current_image_angle, initial_image_angle):
    """Estimate object yaw change from camera-mask axis rotation.

    The overhead view swaps and reflects the world XY axes, so image-axis
    rotation has the opposite sign to world yaw. The half-angle wrap accounts
    for PCA's 180-degree axis ambiguity.
    """
    delta = float(current_image_angle) - float(initial_image_angle)
    delta = 0.5 * math.atan2(math.sin(2.0 * delta), math.cos(2.0 * delta))
    return -delta


def rotate_footprint(footprint, yaw):
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return footprint @ np.array([[cosine, sine], [-sine, cosine]])


def run(args):
    config = json.loads((ROOT / "config/fukuwarai.json").read_text())
    from fukuwarai_success import MouthScorer, HoldScorer
    scorer = MouthScorer(config)
    hold_scorer = HoldScorer(scorer)
    footprint = load_collision_footprint()
    out = Path(args.output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True)
    rclpy.init()
    stream = PoseStream()
    node = CameraVisualExpert(stream, config)
    node.shared_joint_path = getattr(args, 'shared_joint_path', False)
    node.shared_path_projection = getattr(args, 'shared_path_projection', False)
    if node.shared_path_projection and not node.shared_joint_path:
        raise ValueError('Path projection requires shared joint path')
    node.precise = args.precise
    target_pixel = TARGET_PIXEL.copy()
    node.subpixel_pose = getattr(args, 'subpixel_pose', False)
    if node.subpixel_pose and not args.precise:
        raise ValueError('--subpixel-pose requires --precise')
    if getattr(args, 'stable_face_registration', False) and not getattr(args, 'registered_image_goal', False):
        raise ValueError('--stable-face-registration requires --registered-image-goal')
    if getattr(args, 'registered_image_goal', False) and not args.precise:
        raise ValueError('--registered-image-goal requires --precise')
    calibration = CameraCalibration(config)
    result = {
        "source": "camera_visual_servo", "screen_observed": False,
        "camera_image_used_for_control": True, "control_observation": ["observation.images.overhead", "observation.state"],
        "image_compute_mode": "worker_thread_with_ros_callbacks_active",
        "oracle_coordinates_in_control": False, "seed": args.seed,
        # Legacy camera-servo stop threshold is a control heuristic only.
        # Passing it does not establish v2 success: the independent scorer
        # checks contour, orientation, board contact and released hold.
        "visual_stop_threshold_m": 0.012,
        "control_method": "camera_directed_continuous_push_with_image_orientation_and_reachability_fallback",
        "success": False, "recording_camera": True, "evaluation": scorer.rules["version"], "success_rules": scorer.rules,
        "geometry_scale": config.get("geometry_scale", 1),
        "scene_config_sha256": hashlib.sha256((ROOT / "config/fukuwarai.json").read_bytes()).hexdigest(),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recording_helper_sha256": {
            name: hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
            for name in ['synced_recorder.py', 'collect_expert_demo.py']},
        "goal_xy_m": config["target_xy_m"], "target_pixel_xy": TARGET_PIXEL.tolist(),
        "tolerance_m": scorer.rules["center_tolerance_m"], "hold_time_sim_s": config["hold_time_sim_s"],
        "mouth_teleported_during_episode": False,
    }
    if node.shared_joint_path:
        result['joint_path_executor'] = {'version':'endpoint_preserving_rate1_v1',
            'helper_sha256':hashlib.sha256((ROOT/'scripts/fukuwarai_joint_path.py').read_bytes()).hexdigest(),
            'max_planned_speed_rad_s':1., 'board_clearance_m':.0005,
            'continuous_waypoints':True, 'model_success':False}
    if node.shared_path_projection:
        result['joint_path_executor'].update(version='endpoint_preserving_rate1_floor_v1',
            path_projection_sha256=hashlib.sha256((ROOT/'scripts/fukuwarai_path_projection.py').read_bytes()).hexdigest())
    if args.precise:
        result['visual_stop_threshold_m'] = .0008
        result['visual_yaw_stop_threshold_deg'] = 1.
        result['visual_outline_stop_bound_m'] = .0012
        result['teacher_helper_sha256'] = {
            name: hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
            for name in ['fukuwarai_visual_goal.py', 'fukuwarai_contact_calibration.py']}
        result['control_method'] = 'image_silhouette_short_push_with_yaw_correction_v2'
        if node.subpixel_pose:
            result['subpixel_pose'] = True
            for name in ['fukuwarai_image_pose.py', 'fukuwarai_subpixel_pose.py']:
                result['teacher_helper_sha256'][name] = hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
    try:
        # The mouth pose is evaluation-only. Do not require it before reset:
        # a previous timed-out entity-removal service can leave the model
        # temporarily absent, and this trial's reset below recreates it.
        deadline = time.monotonic() + 20
        next_recovery_bootstrap_step = 0.
        while not (all(n in node.joints for n in ARM + [HAND])
                   and node.sim > 0 and node.latest_image is not None):
            if getattr(args, 'recovery_handoff', None):
                # Parent paused physics before teardown. Bootstrap ROS telemetry
                # one physics millisecond at a time without an unbounded resume.
                from evaluate_fukuwarai_act import control
                if time.monotonic() >= next_recovery_bootstrap_step:
                    control('paused_step 1')
                    next_recovery_bootstrap_step = time.monotonic() + .1
            node.spin()
            if time.monotonic() > deadline:
                missing = []
                if not all(n in node.joints for n in ARM + [HAND]):
                    missing.append("joint_states")
                if node.sim <= 0:
                    missing.append("clock")
                if node.latest_image is None:
                    missing.append("overhead_camera")
                raise RuntimeError("No initial telemetry: " + ", ".join(missing))
        kine = kinematics()
        if node.count_publishers(ARM_TOPIC) > 1 or node.count_publishers(HAND_TOPIC) > 1:
            raise RuntimeError("Another teleoperation/controller publisher is active; stop it first")

        recovery_path = getattr(args, 'recovery_handoff', None)
        if recovery_path:
            if not node.shared_joint_path:
                raise ValueError('Recovery handoff requires shared joint path')
            from fukuwarai_recovery_handoff import validate_handoff
            handoff = validate_handoff(
                recovery_path, seed=args.seed,
                scene_sha256=result['scene_config_sha256'],
                executor_sha256=hashlib.sha256((ROOT/'scripts/fukuwarai_joint_path.py').read_bytes()).hexdigest(),
                current_sim_s=node.sim,
                current_joints=[node.joints[n] for n in ARM + [HAND]])
            result.update(source='camera_visual_recovery', recovery_handoff=handoff,
                          original_reset_count=1, recovery_reset_count=0,
                          model_success=False)
            node.stage('validated_recovery_takeover_no_reset')
            from evaluate_fukuwarai_act import physics
            physics(False)
            result['recovery_startup_physics_paused'] = True
            # No home, gripper command or mouth reset before recording recovery.
            reset_pose_sim_s = node.sim
        else:
            node.stage("home_before_recording")
            q = np.array([node.joints[n] for n in ARM])
            node.command(np.zeros(4), max(1.5, float(np.max(np.abs(q))) / 0.8))
            node.command([0.1745], 0.5, hand=True)
            previous_id = node.mouth_id
            import subprocess
            subprocess.run(["python3", str(ROOT / "scripts/reset_mouth.py"), "--seed", str(args.seed)], check=True)
            deadline = time.monotonic() + 25
            while node.mouth_id == previous_id:
                node.spin()
                if time.monotonic() > deadline:
                    raise RuntimeError("New mouth entity not observed after reset")
            reset_pose_sim_s = node.pose_sim
        node.wait(1.0)
        preflight_started = time.monotonic()
        try:
            mouth_pixel = node.wait_for_visible_mouth(reset_pose_sim_s + 0.5)
        finally:
            result["preflight_wait_wall_s"] = time.monotonic() - preflight_started
            # Keep the latest image even when the visual never appears, so
            # a rejected trial remains diagnosable without accepting it.
            if node.latest_image is not None and node.latest_image.data:
                (out / "preflight.jpg").write_bytes(bytes(node.latest_image.data))
                result["preflight_image_path"] = "preflight.jpg"
                result["preflight_camera_sim_time_ns"] = (
                    int(node.latest_image.header.stamp.sec) * 1_000_000_000
                    + int(node.latest_image.header.stamp.nanosec)
                )
        if getattr(args, 'registered_image_goal', False):
            from fukuwarai_image_goal import register_goal
            registration = node.compute(register_goal,
                cv2.imread(str(out/'preflight.jpg')),
                cv2.imread(str(ROOT/'models/fukuwarai_board/materials/textures/face-v2.png')),
                cv2.imread(str(ROOT/'docs/design/approved-face-v2.png')), config,
                getattr(args, 'stable_face_registration', False))
            target_pixel = np.array(registration['pixel'])
            result['image_goal_registration'] = registration
            if getattr(args, 'stable_face_registration', False):
                result['teacher_helper_sha256']['fukuwarai_registration_masks.py'] = hashlib.sha256((ROOT/'scripts/fukuwarai_registration_masks.py').read_bytes()).hexdigest()
            result['target_pixel_xy'] = target_pixel.tolist()
            paths = ['scripts/fukuwarai_image_goal.py',
                     'models/fukuwarai_board/materials/textures/face-v2.png',
                     'docs/design/approved-face-v2.png']
            result['image_goal_source_sha256'] = {
                path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths}
        # Ground-truth pose is retained solely as evaluation metadata.
        initial_eval_pose = node.mouth.copy()
        result["initial_mouth_xyz_m"] = initial_eval_pose.tolist()
        error_pixel = target_pixel - mouth_pixel
        error_world = calibration.pixel_error_to_world(error_pixel)
        result["visual_start_pixel_xy"] = mouth_pixel.tolist()
        result["visual_start_pixel_error"] = node.last_detection
        initial_axis_angle = node.last_detection["principal_axis_image_angle_rad"]
        result["visual_start_axis_image_angle_rad"] = initial_axis_angle
        result["visual_start_axis_eigenvalue_ratio"] = node.last_detection["principal_axis_eigenvalue_ratio"]
        result["visual_start_error_estimate_m"] = float(np.linalg.norm(error_world))
        if recovery_path:
            result["start_distribution_source"] = "original learned prefix reset; this is an intermediate recovery state"
        if not recovery_path and np.linalg.norm(error_world) < 0.010:
            raise RuntimeError("Seeded start appears too close to the visual goal for a scored episode")
        image_xy_audit = calibration.pixel_to_world(mouth_pixel)
        image_pose_audit_error = float(np.linalg.norm(image_xy_audit - initial_eval_pose[:2]))
        result["visual_detection_vs_evaluation_position_error_m"] = image_pose_audit_error
        if image_pose_audit_error > 0.004:
            raise RuntimeError("Fresh camera detection does not match the evaluation-only start audit")

        if recovery_path:
            # Freeze only bag startup; do not relax recorder cadence checks.
            from evaluate_fukuwarai_act import physics
            physics(True)
        try:
            node.start_bag(out)
            if recovery_path:
                # Declare capture start prospectively, before admitting images.
                # Startup samples remain outside the capture window; no recorded
                # rows are removed or relabelled after a quality failure.
                node.recorder.begin = int((node.sim + .5) * 1e9)
                result['recovery_capture_warmup_sim_s'] = .5
        finally:
            if recovery_path:
                physics(False)
        if recovery_path:
            node.wait(.8)  # no commanded motion during pre-capture warmup
        node.visual_rows = (out / "visual_control.jsonl").open("w")
        node.stage("visual_servo")
        height = float(config["board_surface_z_m"]) + 0.003
        initial_eval_push = node.mouth.copy()
        previous_error = None
        previous_push = None
        contact_bias_by_mode = dict(angular=0., translation=0.)
        no_progress_passes = 0
        max_push_passes = 60 if args.precise else 6
        node.stage("continuous_visual_push")
        for pass_index in range(max_push_passes + 1):
            # Observe with the arm clear. One image-derived trajectory then
            # keeps contact for the entire pass; retract only after that pass.
            mouth_pixel = node.compute(node.detect_mouth)
            error_pixel = target_pixel - mouth_pixel
            error_world = calibration.pixel_error_to_world(error_pixel)
            error_m = float(np.linalg.norm(error_world))
            orientation_delta_yaw = camera_axis_delta_to_world_yaw(
                node.last_detection["principal_axis_image_angle_rad"], initial_axis_angle)
            if args.precise:
                orientation_delta_yaw = node.last_detection['fitted_yaw_rad']
            mouth_xy_observed = calibration.pixel_to_world(mouth_pixel)
            observed_contact_progress = None
            if args.precise and previous_push is not None:
                from fukuwarai_contact_calibration import update_mode_bias
                yaw_change = (previous_push['yaw'] - orientation_delta_yaw + math.pi) % (2*math.pi) - math.pi
                if previous_push['angular'] and abs(yaw_change) <= math.radians(20):
                    observed_contact_progress = .018 * yaw_change * math.copysign(1., previous_push['yaw'])
                elif (not previous_push['angular'] and previous_push['error'] < .012
                      and abs(yaw_change) <= math.radians(5)):
                    observed_contact_progress = float(np.dot(mouth_xy_observed-previous_push['xy'], previous_push['direction']))
                if observed_contact_progress is not None:
                    contact_bias_by_mode = update_mode_bias(contact_bias_by_mode, previous_push['angular'], previous_push['progress'], observed_contact_progress)
            angular_only = args.precise and error_m < .006 and abs(orientation_delta_yaw) > math.radians(1.)
            contact_bias_m = contact_bias_by_mode['angular' if angular_only else 'translation']
            row = {
                "pass": pass_index, "sim_time_s": float(node.sim),
                "camera_sim_time_ns": node.last_detection["camera_sim_time_ns"],
                "mouth_pixel_xy": mouth_pixel.tolist(), "target_pixel_xy": target_pixel.tolist(),
                "pixel_error_xy": error_pixel.tolist(), "visual_error_estimate_m": error_m,
                "detector_area_px": node.last_detection["area_px"],
                "detector_center_estimator": node.last_detection["center_estimator"],
                "detector_bbox_xywh": node.last_detection["bbox_xywh"],
                "detector_bbox_center_px": node.last_detection["bbox_center_px"],
                "detector_mask_centroid_px": node.last_detection["mask_centroid_px"],
                "mouth_orientation_delta_yaw_camera_rad": orientation_delta_yaw,
                "orientation_estimator": "authored_silhouette_fit" if args.precise else "camera_red_mask_principal_axis",
                "orientation_axis_eigenvalue_ratio": node.last_detection["principal_axis_eigenvalue_ratio"],
                "image_contact_bias_m": contact_bias_m,
                "image_contact_bias_by_mode_m": dict(contact_bias_by_mode),
                "image_observed_contact_progress_m": observed_contact_progress,
            }
            if node.subpixel_pose:
                row['subpixel_refinement_applied'] = node.last_detection['subpixel_refinement_applied']
                row['subpixel_silhouette_iou'] = node.last_detection['subpixel_silhouette_iou']
                row['coarse_pose'] = node.last_detection['coarse_pose']
            if args.precise:
                from fukuwarai_visual_goal import visual_goal_pass, visual_outline_bound
                shape_radius = float(np.linalg.norm(footprint, axis=1).max())
                row['image_outline_displacement_bound_m'] = visual_outline_bound(error_m, orientation_delta_yaw, shape_radius)
                visual_ready = visual_goal_pass(error_m, orientation_delta_yaw, shape_radius)
            else:
                visual_ready = error_m <= result["visual_stop_threshold_m"]
            if visual_ready:
                result["success"] = True
                result["visual_final_pixel_xy"] = mouth_pixel.tolist()
                result["visual_final_error_estimate_m"] = error_m
                if node.visual_rows:
                    node.visual_rows.write(json.dumps(row) + "\n")
                    node.visual_rows.flush()
                break
            if previous_error is not None:
                if not args.precise and error_m > previous_error + 0.004:
                    raise RuntimeError("Visual error increased after a continuous push; stopping safely")
                if error_m >= previous_error - 0.0015:
                    no_progress_passes += 1
                else:
                    no_progress_passes = 0
                if not args.precise and no_progress_passes >= 2:
                    # Avoid another blind push. Release and let the evaluator
                    # score the exact 15 mm / 1 s task condition; it does not
                    # feed back into any movement decision.
                    result["success"] = True
                    result["visual_stop_reason"] = "no_camera_observed_progress"
                    result["visual_final_pixel_xy"] = mouth_pixel.tolist()
                    result["visual_final_error_estimate_m"] = error_m
                    if node.visual_rows:
                        node.visual_rows.write(json.dumps(row) + "\n")
                        node.visual_rows.flush()
                    break
            if error_m > 0.075:
                raise RuntimeError("Mouth moved outside the visual controller's safe workspace region")
            if pass_index >= max_push_passes:
                raise RuntimeError(f"Camera visual controller exceeded {max_push_passes} continuous push passes")

            radial_direction = error_world / max(error_m,1e-9)
            mouth_xy_image = calibration.pixel_to_world(mouth_pixel)
            contact_center = mouth_xy_image.copy()
            if angular_only:
                radial_direction = np.array([math.cos(orientation_delta_yaw),math.sin(orientation_delta_yaw)])
                # Offset the force line to generate a restoring torque.
                lateral = np.array([-radial_direction[1],radial_direction[0]])
                contact_center += lateral * math.copysign(.018,orientation_delta_yaw)
            oriented_footprint = rotate_footprint(footprint, orientation_delta_yaw)
            tool_cursor, q_above, q_low, direction, offset, geometry_support = select_contact(
                node, kine, contact_center, radial_direction, config, oriented_footprint)
            support = support_radius(oriented_footprint, direction)
            alignment = float(np.dot(radial_direction, direction))
            parallel_error = float(np.dot(error_world, direction))
            perpendicular_error = math.sqrt(max(0.0, error_m ** 2 - parallel_error ** 2))
            threshold = result["visual_stop_threshold_m"]
            parallel_residual = math.sqrt(max(0.0, threshold ** 2 - perpendicular_error ** 2))
            desired_progress = max(0.0, parallel_error - parallel_residual)
            if args.precise:
                desired_progress = min(.0015,max(.00025,abs(orientation_delta_yaw)*.008)) if angular_only else min(.006,max(.0002,.5*desired_progress))
            if args.precise:
                # The distal gripper extends about 2mm ahead of the authored
                # TCP. The old point-support heuristic added a spurious 4mm
                # travel at every touch, causing near-goal oscillation.
                from fukuwarai_image_pose import rear_contact_distance
                line_support = rear_contact_distance(oriented_footprint,direction,contact_center-mouth_xy_image)
                gap = max(0.,float(np.dot(contact_center-tool_cursor,direction)-line_support-.002-contact_bias_m))
                row['force_line_support_m'] = line_support
            else:
                gap = max(0.0, float(np.dot(mouth_xy_image - tool_cursor, direction) - support))
            # Include the initial tool-to-part gap and leave a 12 mm visual
            # margin for slip and calibration error before the next check.
            push_distance = gap + desired_progress
            if push_distance <= (0.0001 if args.precise else 0.003):
                raise RuntimeError("Camera-derived continuous push is too short to execute safely")
            tool_next = tool_cursor + direction * push_distance
            row["tool_cursor_xy_m"] = tool_cursor.tolist()
            row["radial_direction_world"] = radial_direction.tolist()
            row["push_direction_world"] = direction.tolist()
            row["goal_alignment_cosine"] = alignment
            row["goal_perpendicular_error_m"] = perpendicular_error
            row["estimated_edge_gap_m"] = gap
            row["planned_push_m"] = push_distance
            row["planned_goal_progress_m"] = desired_progress
            row['angular_only'] = bool(angular_only)
            row['contact_center_xy_m'] = contact_center.tolist()
            row["control_mode"] = "single_continuous_camera_directed_push"
            if node.visual_rows:
                node.visual_rows.write(json.dumps(row) + "\n")
                node.visual_rows.flush()
            node.command(q_above, max(0.7, float(np.max(np.abs(q_above -
                                   np.array([node.joints[n] for n in ARM])))) / 1.0))
            node.cartesian_path(kine, tool_cursor, np.linspace(height + 0.06, height, 5)[1:], 0.7)
            push_seconds = max(1.0, push_distance / 0.014)
            node.cartesian_xy_path(kine, tool_cursor, tool_next, height, push_seconds)
            # Retract once after the complete pass, then use the overhead
            # camera to decide whether another pass is needed.
            node.cartesian_path(kine, tool_next, np.linspace(height, height + 0.06, 5)[1:], 0.7)
            home_time = max(1.0, float(np.max(np.abs(np.array([node.joints[n] for n in ARM]))) / 1.0))
            node.command(np.zeros(4), home_time)
            if args.precise:
                previous_push = dict(xy=mouth_xy_observed.copy(), yaw=orientation_delta_yaw,
                                     direction=direction.copy(), progress=desired_progress,
                                     angular=angular_only, error=error_m)
                result['adaptive_contact_gap_source'] = 'camera before/after displacement only'
            previous_error = error_m
            result["visual_push_passes"] = pass_index + 1
            result["approach_offset_m"] = float(offset)
            result["mouth_support_radius_m"] = float(geometry_support)
            result["visual_contact_support_estimate_m"] = support
            result["goal_alignment_cosine"] = alignment
            print(f"continuous camera push {pass_index + 1}: mouth_px={mouth_pixel.round(1).tolist()} "
                  f"error={error_m:.4f}m distance={push_distance:.4f}m", flush=True)

        if not result["success"]:
            raise RuntimeError("Camera visual controller did not reach the visual stop threshold")
        result["push_displacement_m"] = float(np.linalg.norm(node.mouth[:2] - initial_eval_push[:2]))
        node.stage("withdraw_up")
        # The last visual-servo cycle already returned the arm to home.
        node.stage("verify_released")
        start = node.sim
        goal = np.array(config["target_xy_m"], dtype=float)
        while node.sim - start < config["hold_time_sim_s"] + .2:
            node.wait(0.1)
            if node.sim-node.pose_sim > scorer.rules['max_telemetry_gap_s']:
                raise RuntimeError('Scoring pose is stale')
            placement = hold_scorer.update(node.sim,node.mouth,node.mouth_quaternion,
                                           float(kine.tool([node.joints[n] for n in ARM])[2]))
            result['final_placement_metrics'] = placement
            if not placement['instantaneous_pass']:
                result['success'] = False
                result['error'] = 'Released mouth failed outline/angle/position/height criteria v2'
                break
        result['success'] = bool(result['success'] and result.get('final_placement_metrics',{}).get('success',False))
        result["held_after_withdrawal_sim_s"] = result.get('final_placement_metrics',{}).get('held_sim_s',0.) if result['success'] else 0.
        if result["success"]:
            # Keep the arm retracted and record a stationary tail after the
            # required one-second score hold. This preserves the 100-frame
            # quality floor for otherwise short successful demonstrations.
            result["post_success_recording_sim_s"] = POST_SUCCESS_RECORDING_S
            node.stage("post_success_recording")
            node.wait(POST_SUCCESS_RECORDING_S)
        result["final_mouth_xyz_m"] = node.mouth.tolist()
        result["final_error_m"] = float(np.linalg.norm(node.mouth[:2] - goal))
        node.stage("success" if result["success"] else "failure")
        node.wait(0.3)
    except (Exception, KeyboardInterrupt) as exc:
        result["success"] = False
        result["error"] = str(exc) or "Interrupted by operator"
        print(f"FAILED: {exc}", flush=True)
    finally:
        if getattr(args, 'recovery_handoff', None):
            import evaluate_fukuwarai_act as recovery_runtime
            if recovery_runtime.CONTROL is not None:
                recovery_runtime.physics(False)
                recovery_runtime.CONTROL.terminate()
                recovery_runtime.CONTROL.wait(timeout=5)
                recovery_runtime.CONTROL = None
        if node.mouth is not None:
            goal = np.array(config["target_xy_m"], dtype=float)
            result["final_mouth_xyz_m"] = node.mouth.tolist()
            result["final_error_m"] = float(np.linalg.norm(node.mouth[:2] - goal))
        if rclpy.ok() and all(n in node.joints for n in ARM) and node.arm.get_subscription_count():
            hold = JointTrajectory(joint_names=ARM)
            point = JointTrajectoryPoint(positions=[float(node.joints[n]) for n in ARM])
            point.time_from_start.nanosec = 200_000_000
            hold.points = [point]
            node.arm.publish(hold)
            node.record(ARM_TOPIC, hold)
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=0.02)
        audit = node.close_bag()
        if node.visual_rows:
            node.visual_rows.close()
            node.visual_rows = None
        result["synchronization"] = audit
        result["physical_success"] = bool(result["success"] and
                                           result.get("held_after_withdrawal_sim_s", 0) >= 1.0)
        result["accepted_for_training"] = bool(result["physical_success"] and audit["quality_pass"])
        stream.close()
        node.calculations.shutdown(wait=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        (out / "scene_config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(json.dumps({"output": str(out), **result}), flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-path-projection', action='store_true')
    parser.add_argument('--shared-joint-path', action='store_true', help='Opt in to endpoint-preserving1rad/s paths with full sampled board checks')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--recovery-handoff', type=Path, default=None, help='Validated fresh model-prefix handoff; skip home/reset, preserve original start')
    parser.add_argument('--precise',action='store_true',help='Use image contour pose and short corrective pushes')
    parser.add_argument('--subpixel-pose',action='store_true',help='Opt in to supersampled local silhouette refinement; requires --precise')
    parser.add_argument('--stable-face-registration', action='store_true', help='Exclude movable-lip/guide features from image registration; strict fit gates unchanged')
    parser.add_argument('--registered-image-goal',action='store_true',help='Opt in to artwork/image registered goal; requires --precise')
    parser.add_argument("--output-root", default=str(ROOT / "data/expert_images"))
    raise SystemExit(run(parser.parse_args()))
