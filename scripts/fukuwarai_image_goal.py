"""Register the known face artwork to a camera image, without object poses."""
import math
import cv2
import numpy as np


def lip_plane_pixel(guide_pixel, config):
    camera = config['observation_camera']
    depth = camera['pose_m_rad'][2] - config['board_surface_z_m']
    lip_depth = depth - config['mouth_size_m'][2]
    if not 0 < lip_depth < depth:
        raise ValueError('Invalid camera/board/lip depth')
    principal = (np.array([camera['width'], camera['height']], dtype=float)-1)/2
    return principal + (np.asarray(guide_pixel)-principal)*depth/lip_depth


def register_goal(image, artwork, approved_face, config, mask_moving_regions=False):
    """Estimate the guide center with a bounded, image-only similarity fit.

    The approved lip bbox defines the target in artwork pixels. Only static
    camera/part dimensions enter the plane-height correction, never world xy.
    """
    if image.shape[:2] != (480, 640) or artwork.shape[:2] != approved_face.shape[:2]:
        raise ValueError('Unexpected camera or artwork dimensions')
    h, w = approved_face.shape[:2]
    roi = approved_face[int(h*.57):int(h*.67), int(w*.34):int(w*.66)].astype(float)
    b, g, r = cv2.split(roi)
    ys, xs = np.where((r > g*1.55) & (r > b*1.45))
    if len(xs) < 100:
        raise ValueError('Approved lip bbox missing')
    target = np.array([(xs.min()+xs.max())/2+int(w*.34),
                       (ys.min()+ys.max())/2+int(h*.57), 1.])
    detector = cv2.SIFT_create()
    source_mask = observed_mask = None
    if mask_moving_regions:
        from fukuwarai_registration_masks import registration_masks
        source_mask, observed_mask = registration_masks(artwork, image)
    k1, d1 = detector.detectAndCompute(cv2.cvtColor(artwork, cv2.COLOR_BGR2GRAY), source_mask)
    k2, d2 = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), observed_mask)
    if d1 is None or d2 is None:
        raise ValueError('Insufficient face image features')
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .7*pair[1].distance]
    if len(matches) < 30:
        raise ValueError('Insufficient face artwork matches')
    source = np.float32([k1[m.queryIdx].pt for m in matches])
    observed = np.float32([k2[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(source, observed, method=cv2.RANSAC,
                                                 ransacReprojThreshold=1., maxIters=3000)
    if matrix is None or inliers is None or int(inliers.sum()) < 30:
        raise ValueError('Face registration has insufficient inliers')
    error = np.linalg.norm(source @ matrix[:, :2].T + matrix[:, 2] - observed, axis=1)
    rmse = float(np.sqrt(np.mean(error[inliers[:, 0] > 0]**2)))
    angle = math.atan2(matrix[1, 0], matrix[0, 0])
    guide = matrix @ target
    pixel = lip_plane_pixel(guide, config)
    if not np.isfinite(pixel).all() or rmse > .5 or abs(angle) > math.radians(1):
        raise ValueError('Uncertain fixed face registration')
    if np.linalg.norm(pixel-np.array([320.,270.])) > 2.:
        raise ValueError('Registered image goal outside calibration neighborhood')
    return {'pixel': pixel.tolist(), 'guide_plane_pixel': guide.tolist(),
            'inliers': int(inliers.sum()), 'inlier_rmse_px': rmse,
            'similarity_matrix': matrix.tolist(),
            'source': 'approved artwork and live image features; static depth correction only',
            'simulator_object_pose_used': False,
            'feature_mask_version': 'localized_mutable_region_v1' if mask_moving_regions else None}
