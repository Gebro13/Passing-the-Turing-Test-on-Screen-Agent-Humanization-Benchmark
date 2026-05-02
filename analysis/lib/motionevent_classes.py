from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, TypedDict
import pandas as pd
from typing_extensions import TypeAlias

@dataclass
class GotEvent:
    timestamp_us: int # time in microseconds (\mu s) (1e-6 seconds)
    device: str # device name (e.g., /dev/input/event4)
    type: int       # event type in decimal (e.g., EV_ABS, EV_KEY)
    code: int       # event code in decimal (e.g., ABS_MT_POSITION_X, BTN_TOUCH)
    value: int      # event value in decimal

@dataclass
class FingerEvent:
    timestamp_us: int # time in microseconds (\mu s) (1e-6 seconds)
    x: int          # x coordinate in pixels
    y: int          # y coordinate in pixels

SingularActionType: TypeAlias = List[FingerEvent]  # list of Events for a single action (e.g., a tap, swipe, etc.)

END_SAMPLE_XY_IDX = -2 # the index of the last point that has valid x, y coordinates. This is used to handle the vanishing point issue where the last point may have None for x and y.

# Target: output write SingularActionType. input args: if write SingularActionType and the function require unwrapping the class, then better there will be a List[FingerEvent] inside the function that can trigger an error.

def is_integral(trace: SingularActionType) -> bool:
    """
        check if all events in the trace have integral x and y values.  
        If less than 2 events, raise AssertionError.
    """
    # 
    for event in trace:
        if not isinstance(event.x, int) or not isinstance(event.y, int) or not isinstance(event.timestamp_us, int):
            return False
    
    # assert that x and y same for [-2] and [-1]
    assert len(trace) >= 2, "Trace must have at least 2 events to check for integral upfinger event"
    if trace[-1].x != trace[-2].x or trace[-1].y != trace[-2].y:
        return False

    return True


SessionType = Tuple[str, List[SingularActionType]]  # (session_id, list of Events)

def get_session_singular_actions(session: SessionType) -> List[SingularActionType]:
    """
        Given a session, return the list of SingularActionType (list of list of Events) for that session.
    """
    session_id, singular_actions = session
    return singular_actions

@dataclass
class SwipeFeaturedSessionType:
    session_id: str
    # time-ordered sequence of feature dictionaries, or a DataFrame where each row corresponds to a swipe and columns correspond to features
    features: Sequence[Dict[str, float]] | pd.DataFrame

# a = SwipeFeaturedSessionType(session_id="session_1", features=[{"feature1": 0.5, "feature2": 1.0}, {"feature1": 0.3, "feature2": 0.8}])
