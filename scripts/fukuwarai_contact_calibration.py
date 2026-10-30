"""Bounded teacher contact-gap adaptation from before/after camera motion."""
import math


def update_contact_bias(bias_m, planned_progress_m, observed_progress_m):
    if not all(math.isfinite(x) for x in [bias_m, planned_progress_m, observed_progress_m]):
        raise ValueError('Nonfinite camera contact calibration')
    if planned_progress_m < 0:
        raise ValueError('Negative planned push')
    # Ignore large or opposite motion rather than chasing an unreliable image.
    if not -.0001 <= observed_progress_m <= .012:
        return bias_m
    correction = max(-.001, min(.001, .5 * (observed_progress_m - planned_progress_m)))
    return max(-.002, min(.004, bias_m + correction))


def update_mode_bias(biases, angular, planned_progress_m, observed_progress_m):
    """Offset force lines and central translation have separate gap errors."""
    updated = dict(biases)
    key = 'angular' if angular else 'translation'
    updated[key] = update_contact_bias(updated[key], planned_progress_m, observed_progress_m)
    return updated
