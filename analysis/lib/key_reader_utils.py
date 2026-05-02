import re
from typing import List, Optional
from re import Match

from analysis.lib.keyevent_classes import MotionKeyEvent, HardKeyType, IMEEvent, IMESpecialKeyType

# Map kernel key names to HardKeyType
_MOTION_EVENT_KEY_NAME_MAP = {
    "BACK": HardKeyType.BACK,
    "HOME": HardKeyType.HOME,
    "MENU": HardKeyType.RECENTS,      # APP_SWITCH / MENU → RECENTS
    "APP_SWITCH": HardKeyType.RECENTS,
    "POWER": HardKeyType.POWER,
    "VOLUMEUP": HardKeyType.VOLUME_UP,
    "VOLUMEDOWN": HardKeyType.VOLUME_DOWN,
    "MUTE": HardKeyType.VOLUME_MUTE,
    "ENTER": HardKeyType.ENTER,
}

_IME_KEY_NAME_MAP = {
    "KEYCODE_DEL": IMESpecialKeyType.DEL,
    "KEYCODE_ENTER": IMESpecialKeyType.ENTER,
    "ADB_CLEAR_TEXT": IMESpecialKeyType.CLEAR_TEXT,
    "KEYCODE_BACK": IMESpecialKeyType.BACK,
    "KEYCODE_HOME": IMESpecialKeyType.HOME,
    "KEYCODE_APP_SWITCH": IMESpecialKeyType.APP_SWITCH,
    "KEYCODE_MENU": IMESpecialKeyType.MENU,
}


def keys_generator_from_motionevent(adb_getevent_generator: List[str]) -> List[MotionKeyEvent]:
    """
        Parse adb getevent -lt output and extract only hardware key events
        (EV_KEY KEY_*), pairing DOWN/UP into KeyEvent objects.
    """
    time_pattern = re.compile(r'^\[ {0,8}([0-9]{0,8})\.([0-9]{6})\]')
    key_pattern = re.compile(r'EV_KEY       KEY_(\w+)             (DOWN|UP)')

    # pending_down[key_name] = timestamp_us of the DOWN event
    pending_down: dict[str, int] = {}
    result: List[MotionKeyEvent] = []

    for line in adb_getevent_generator:
        time_match = time_pattern.search(line)
        if time_match is None:
            continue

        key_match = key_pattern.search(line)
        if key_match is None:
            continue

        t_us = int(time_match.group(1)) * 1_000_000 + int(time_match.group(2))
        key_name = key_match.group(1)
        direction = key_match.group(2)

        if direction == "DOWN":
            pending_down[key_name] = t_us
        elif direction == "UP" and key_name in pending_down:
            down_us = pending_down.pop(key_name)
            try:
                key_type = _MOTION_EVENT_KEY_NAME_MAP[key_name]
            except KeyError:
                raise ValueError(f"Unrecognized key name in getevent line: {line}")
            result.append(MotionKeyEvent(down_us=down_us, up_us=t_us, key_type=key_type))

    return result

def IME_generator_from_IME_recording(ime_event_generator: List[str]) -> List[IMEEvent]:
    """
        Parse IME event log lines into IMEEventType objects.

        Supported line formats (all start with two float timestamps):
          wall_s uptime_s `char`                        — single character
          wall_s uptime_s `KEYCODE_*`                   — keycode as literal string
          wall_s uptime_s <SPECIAL_KEY>                 — special key (e.g. ADB_CLEAR_TEXT)
          wall_s uptime_s B64: `...` Decoded: `...`     — base64-encoded string

        flush_time_s is set to the first timestamp (wall time).


        Warning: in IME_event_20260125_021712.txt there was a `\n`. this cannot be parsed correctly.
    """
    # Patterns for the three payload variants
    b64_pattern = re.compile(r'^([\d.]+)\s+([\d.]+)\s+B64:\s+`([^`]*)`\s+Decoded:\s+`([^`]*)`')
    special_pattern = re.compile(r'^([\d.]+)\s+([\d.]+)\s+<([^>]+)>')
    char_pattern = re.compile(r'^([\d.]+)\s+([\d.]+)\s+`([^`]*)`')

    result: List[IMEEvent] = []

    for line in ime_event_generator:
        line = line.rstrip('\n')
        if not line:
            continue

        m = b64_pattern.match(line)
        if m:
            wall_s = float(m.group(1))
            decoded = m.group(4)
            result.append(IMEEvent(flush_time_s=wall_s, flushed_output=decoded))
            continue

        m = special_pattern.match(line)
        if m:
            wall_s = float(m.group(1))
            key_name = m.group(3)
            
            try:
                key_code = _IME_KEY_NAME_MAP[key_name]
            except KeyError:
                raise ValueError(f"Unrecognized special key in IME event line: {line}")

            result.append(IMEEvent(flush_time_s=wall_s, flushed_output=key_code))
            continue

        m = char_pattern.match(line)
        if m:
            wall_s = float(m.group(1))
            char = m.group(3)
            if len(char) != 1:
                try:
                    key_code = _IME_KEY_NAME_MAP[char]
                except KeyError:
                    raise ValueError(f"Expected single character or known keycode in IME event line, got: {line}")
            result.append(IMEEvent(flush_time_s=wall_s, flushed_output=char))
            continue

    return result

def IME_event_name_schema(timestamp: str) -> str:
    return f"IME_event_{timestamp}.txt"