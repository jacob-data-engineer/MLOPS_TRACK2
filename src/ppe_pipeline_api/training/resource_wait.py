"""Status-only resource wait classification; never starts or stops AWS jobs."""


def is_resource_shortage(error):
    response = getattr(error, 'response', {})
    code = str(response.get('Error', {}).get('Code', '')).lower()
    message = str(error).lower()
    return code in ('resourcelimitexceeded', 'insufficientcapacityerror', 'insufficientinstancecapacity') or any(
        token in message for token in ('sagemaker_resource_limit', 'insufficientcapacity',
                                      'insufficient capacity', 'capacity unavailable',
                                      'resource limit exceeded', 'resourcelimitexceeded'))


def describe(queue, gpu, cpu):
    for resource, status in (('GPU', gpu), ('CPU', cpu)):
        if status.get('known'):
            waiting = [j for j in status.get('jobs', [])
                       if j.get('own_pipeline') and j.get('started') is False]
            if waiting:
                return {'waiting': True, 'detail': f"SageMaker has not allocated a {resource} instance for {waiting[0]['name']} yet."}
    if queue.get('started') and queue.get('pending'):
        recording = queue.get('submission_progress') or {}
        if recording.get('resource_wait'):
            return {'waiting': True, 'detail': recording.get('detail') or 'SageMaker capacity or quota is unavailable. The queue will retry.'}
        if gpu.get('known') and gpu.get('busy'):
            return {'waiting': True, 'detail': 'Queued training is waiting for the GPU slot currently occupied by another run.'}
    return {'waiting': False, 'detail': ''}
