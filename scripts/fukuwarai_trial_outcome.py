"""Classify recorded model stops without treating infrastructure faults as trials."""


def countable_model_stop(result):
    reason = result.get('reason', '')
    return (bool(result.get('recording_audit', {}).get('quality_pass'))
            and (reason == 'Endpoint tracking timeout'
                 or reason.startswith(('Safety stop:', 'Model command'))))
