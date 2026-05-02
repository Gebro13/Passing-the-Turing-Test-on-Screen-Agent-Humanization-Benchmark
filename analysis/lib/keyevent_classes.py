

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Union


class HardKeyType(Enum):
    BACK = auto()
    HOME = auto()
    RECENTS = auto()
    POWER = auto()
    VOLUME_UP = auto()
    VOLUME_DOWN = auto()
    VOLUME_MUTE = auto()
    ENTER = auto()

class IMESpecialKeyType(Enum):
    """
    TODO investigate what special keys apps receive as IME events.
    """
    DEL = auto()
    ENTER = auto()
    CLEAR_TEXT = auto()
    HOME = auto()
    BACK = auto()
    APP_SWITCH = auto()
    MENU = auto()

@dataclass
class MotionKeyEvent:
    down_us: int
    up_us: int
    key_type: HardKeyType

@dataclass
class IMEEvent:
    flush_time_s: float
    flushed_output: Union[str, IMESpecialKeyType]


