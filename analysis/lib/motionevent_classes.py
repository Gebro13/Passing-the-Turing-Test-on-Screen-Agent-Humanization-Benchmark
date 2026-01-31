from dataclasses import dataclass
from typing import List, Tuple

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

SessionType = Tuple[str, List[List[FingerEvent]]]  # (session_id, list of GotEvent)