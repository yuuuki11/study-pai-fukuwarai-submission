"""Image-estimated stopping margin; independent of simulator scoring poses."""
import math

CENTER_MARGIN_M = .0008
YAW_MARGIN_RAD = math.radians(1.)
OUTLINE_MARGIN_M = .0012


def visual_outline_bound(center_error_m, yaw_rad, radius_m):
    """Rigid-shape displacement bound from image translation and rotation.

    Uses the authored shape radius, not the observed simulator object pose.
    This controls the teacher only; the fixed v2 scorer remains authoritative.
    """
    if (not all(math.isfinite(x) for x in [center_error_m, yaw_rad, radius_m])
            or center_error_m < 0 or radius_m <= 0):
        raise ValueError('Invalid image pose estimate or shape radius')
    yaw = abs((yaw_rad + math.pi) % (2 * math.pi) - math.pi)
    return center_error_m + 2 * radius_m * math.sin(yaw / 2)


def visual_goal_pass(center_error_m, yaw_rad, radius_m):
    bound = visual_outline_bound(center_error_m, yaw_rad, radius_m)
    return (center_error_m <= CENTER_MARGIN_M
            and abs(yaw_rad) <= YAW_MARGIN_RAD and bound <= OUTLINE_MARGIN_M)
