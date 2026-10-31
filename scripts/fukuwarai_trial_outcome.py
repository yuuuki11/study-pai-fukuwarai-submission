"""Classify model failures; preserve original recording quality labels."""


def short_model_failure(result):
    a=result.get('recording_audit',{})
    n=a.get('frames_written',0)
    reason=result.get('reason','')
    return (0<n<100 and a.get('frames_received')==n and a.get('frames_rejected')==0
            and a.get('fps_requested')==10
            and a.get('maximum_frame_interval_sim_s') is not None
            and a['maximum_frame_interval_sim_s']<=.11
            and a.get('maximum_nearest_state_offset_ms') is not None
            and a['maximum_nearest_state_offset_ms']<=30
            and not result.get('success')
            and (reason=='Endpoint tracking timeout' or reason.startswith(('Safety stop:','Model command'))))


def countable_model_stop(result):
    reason=result.get('reason','')
    return ((bool(result.get('recording_audit',{}).get('quality_pass')) or short_model_failure(result))
            and (reason=='Endpoint tracking timeout' or reason.startswith(('Safety stop:','Model command'))))
