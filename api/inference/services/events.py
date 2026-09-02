from inference.models import InferenceEvent

def record_event(
    inference,
    event_type,
    *,
    user=None,
    message="",
):
    return InferenceEvent.objects.create(
        inference=inference,
        event_type=event_type,
        created_by=user,
        message=message,
    )