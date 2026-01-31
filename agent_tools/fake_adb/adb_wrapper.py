import math
import sys
import subprocess
import os
import time
import random
import pickle
from typing import List, Optional
import shlex
import datetime
import sys
import os
import numpy as np
from pathlib import Path

# Try to import fcntl — if it fails, platform doesn't support it
try:
    import fcntl
except ImportError:
    sys.exit(
        "Error: fcntl is not available on this platform (e.g., Windows).\n"
        "This script requires a Unix-like system (Linux, macOS, etc.) for process locking."
    )

# Ensure the current directory is in python path to import local modules
proj_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


try:
    
    sys.path.append(proj_folder)
    from motionevent_classes import GotEvent, FingerEvent
    from fit_effort_provider import FitEffortProvider, bot_line_fit
except ImportError as e:
    print(f"Wrapper Error: Could not import dependencies: {e}", file=sys.stderr)
    sys.exit(1)

# Define the real ADB executable
REAL_ADB = str(Path.home() / "Android" / "Sdk" / "platform-tools" / "adb")
GLOBAL_TOUCH_DEVICE = "/dev/input/event4"
GLOBAL_EVENT_INTERVAL_US = 11000
GLOBAL_FAKE_HUMAN = False
IME_EVENT_PATH_STR = "~/IME_EVENT_PATH.txt"

# Constants from controller.py
EV_SYN             = 0
EV_KEY             = 1
EV_ABS             = 3
ABS_MT_TRACKING_ID = 57
ABS_MT_POSITION_X  = 53
ABS_MT_POSITION_Y  = 54
BTN_TOUCH          = 330
DOWN               = 1
UP                 = 0
SYN_REPORT         = 0
SYN_MT_REPORT      = 2
KEY_MENU           = 139 # 0x8b 
KEY_HOME           = 102 # 0x66
KEY_BACK           = 158 # 0x9e

# Unverfied constants for multi-touch events
ABS_MT_PRESSURE    = 58
ABS_MT_TOUCH_MAJOR = 48
ABS_MT_TOUCH_MINOR = 49
ABS_MT_ORIENTATION = 59
ABS_MT_WIDTH_MAJOR = 60
ABS_MT_WIDTH_MINOR = 61
ABS_MT_TOOL_TYPE   = 47
ABS_MT_BLOB_ID     = 64
ABS_MT_SLOT       = 47
ABS_MT_TOOL_X      = 65
ABS_MT_TOOL_Y      = 66
ABS_MT_TOOL_ORIENTATION = 67
ABS_MT_TOOL_WIDTH  = 68
ABS_MT_TOOL_MINOR  = 69
ABS_MT_TOOL_MAJOR  = 70
ABS_MT_TOOL_PRESSURE = 71

DEFAULT_PRESS_BTN_TIME_US = 110000 if GLOBAL_FAKE_HUMAN else 1000

def run_real_adb(adb_path: List[str], args: List[str]) -> int:
    """Executes the real adb command with the provided arguments."""
    cmd = adb_path + args
    return subprocess.run(cmd).returncode

def get_current_phone_timestamp() -> str:
    command = f"{REAL_ADB} shell cat /proc/uptime"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        return "0.0"
    return result.stdout.strip()

def log_char(char_or_str: str) -> None:
    # We use os.system here to match controller.py's behavior exactly
    # replace " and ' in char_or_str to avoid shell issues
    char_or_str = char_or_str.replace('"', '\\"').replace("'", "\\'")
    os.system(f"printf \"{get_current_phone_timestamp()} \\`{char_or_str}\\`\n\" >> \"$(cat {IME_EVENT_PATH_STR})\"")

class MotionGenerator:
    static_fit_effort_provider: Optional[FitEffortProvider] = None

    @staticmethod
    def init_provider():
        if MotionGenerator.static_fit_effort_provider is None:
            try:
                pkl_path = os.path.join(proj_folder, "swipe_data.pkl")
                with open(pkl_path, "rb") as f:
                    MotionGenerator.static_fit_effort_provider = FitEffortProvider(pickle.load(f))
            except Exception as e:
                print(f"Warning: Could not load swipe_data.pkl: {e}", file=sys.stderr)

    @staticmethod
    def generate_swipe_trace(x1, y1, x2, y2, duration_us = 500 * 1000, neighbor_time_delta_us: int = GLOBAL_EVENT_INTERVAL_US, fake_human: bool = False, verbose: bool = False) -> List[FingerEvent]:
        """
        Generate a list of (time_offset_us, x, y) for a linear swipe.
        duration_ms: total swipe time in milliseconds
        steps: number of intermediate “dots” (so total points = steps+1)
        """
        
        if fake_human and (MotionGenerator.static_fit_effort_provider is not None):
            trace = MotionGenerator.static_fit_effort_provider.fit(x1, y1, x2, y2)
        else: 
            trace = bot_line_fit(x1, y1, x2, y2, duration_us, neighbor_time_delta_us)
        return trace

    # https://www.kernel.org/doc/html/latest/input/multi-touch-protocol.html
    # describes two kind of protocols: Type A and Type B.
    # Type A: https://www.kernel.org/doc/html/latest/input/multi-touch-protocol.html#protocol-example-a
    # Type B: https://www.kernel.org/doc/html/latest/input/multi-touch-protocol.html#protocol-example-b

    # https://www.kernel.org/doc/html/latest/input/multi-touch-protocol.html#protocol-example-b:~:text=Note,use%20type%20B.
    # MT protocol type A is obsolete, all kernel drivers have been converted to use type B.
    
    # we just write a first version. We can only compile the event file of a total swipe at first then flush the total file into a single device.
    # Since: 
    # 1. since the time of the start of the whole swipe is not very important, we can tolerate a small latency in "agent wanting to send the event" versus "event displayed to screen".
    # 2. This flushing paradigm only use one command for the process of sending the swipe.
    # 3. If we encapsulate every os-interacting command in a single python function, the dirtiness and potential later scope of trying to speed up the command sending will be localized.
    # 4. We want our code to handle multiple potential devices, which may be  achievable.

    @staticmethod
    def single_command_that_may_take_time(command_str: str):
        os.system(command_str)

    @staticmethod
    def flush_event_sequence(adb_path: str, tosend_device: str, event_list: List[GotEvent], keep_temp_file=False):
        """
            a dirty function.
            os.system() seem quicker than subprocess.run() under naked eye: hardly any latency. os.system() is a blocking call, so it will wait for the command to finish.
            tosend_device: e.g. "/dev/input/event4"
        """
        # first, for each GotEvent, we convert it into a string that looks like:
        # [   11406.954013] 0003 0039 00000057
        # the device is assumed to be identical and thus omitted, and 
        # the numbers are in hexadecimal.

        # the event file looks like:

        file_name = f"event_sequence_{(random.randint(0, 0xffffffff)):08x}.txt"
        
        with open(file_name, "w") as f:
            for event in event_list:
                # convert the event to a string
                # be aware of precision loses, so we construct the timestamp not by division but by modulo.
                lower_6_digits = event.timestamp_us % 1000000
                higher_digits = event.timestamp_us // 1000000

                # higher digits need to occupy 8 digits, with padding left with space
                event_timestamp = f"{higher_digits:8d}.{lower_6_digits:06d}"

                # format the event line
                if (event.value == -1):
                    event.value = 0xFFFFFFFF

                line = f"[{event_timestamp}] {event.type:04x} {event.code:04x} {event.value:08x}\n" # follow `android-touch-record-replay` format
                f.write(line)


        # Then we send the file to the device and execute it.
        sendc = MotionGenerator.single_command_that_may_take_time
        # Use the adb_path provided (which is REAL_ADB in this wrapper context)
        sendc(f"{adb_path} push {file_name} /sdcard/")
        sendc(f"{adb_path} shell /data/local/tmp/mysendevent-arm64 {tosend_device} /sdcard/{file_name}")

        if not keep_temp_file:
            # remove the temporary file
            os.remove(file_name)
            sendc(f"{adb_path} shell rm /sdcard/{file_name}")

    @staticmethod
    def swipe_to_event_trace(trace: List[FingerEvent], end_upfinger_time_us = 50000, evdev=GLOBAL_TOUCH_DEVICE, fake_pressure=False) -> List[GotEvent]:
        """
        Generate a Type-B swipe event sequence.
        Returns the generated event: list of `GotEvent`s.

        Params:
        duration_ms  total swipe duration,
        steps        how many intermediate points,
        evdev        the device node for your touchscreen. May differ across devices.
        fake_pressure: if True, injects fake pressure values (not always needed or supported by device).

        You can log or inspect the returned trace to see each "dot"'s timestamp.
        """

        # … inside your swipe() function replace the placeholder with:

        # --- touch down (begin contact) ---
        # assign a tracking ID 0
        first_event = trace[0]

        event_list: List[GotEvent] = []
        # add the first event to the list
        event_list += [
            GotEvent(timestamp_us=first_event.timestamp_us, device=evdev, type=EV_ABS, code=ABS_MT_TRACKING_ID, value=0),
            GotEvent(timestamp_us=first_event.timestamp_us, device=evdev, type=EV_ABS, code=ABS_MT_POSITION_X, value=first_event.x),
            GotEvent(timestamp_us=first_event.timestamp_us, device=evdev, type=EV_ABS, code=ABS_MT_POSITION_Y, value=first_event.y),
            GotEvent(timestamp_us=first_event.timestamp_us, device=evdev, type=EV_KEY, code=BTN_TOUCH, value=DOWN),
            GotEvent(timestamp_us=first_event.timestamp_us, device=evdev, type=EV_SYN, code=SYN_REPORT, value=0)
        ]

        # --- move events ---
        for finger_event in trace[1:]:
            # add the event to the list
            event_list += [
                GotEvent(timestamp_us=finger_event.timestamp_us, device=evdev, type=EV_ABS, code=ABS_MT_POSITION_X, value=finger_event.x),
                GotEvent(timestamp_us=finger_event.timestamp_us, device=evdev, type=EV_ABS, code=ABS_MT_POSITION_Y, value=finger_event.y),
                GotEvent(timestamp_us=finger_event.timestamp_us, device=evdev, type=EV_SYN, code=SYN_REPORT, value=0)
            ]

        finger_uplift_time = trace[-1].timestamp_us + end_upfinger_time_us  # add a small delay after the last move; see the docs for details

        event_list += [
            GotEvent(timestamp_us=finger_uplift_time, device=evdev, type=EV_ABS, code=ABS_MT_TRACKING_ID, value=0xFFFFFFFF),
            GotEvent(timestamp_us=finger_uplift_time, device=evdev, type=EV_KEY, code=BTN_TOUCH, value=UP),
            GotEvent(timestamp_us=finger_uplift_time, device=evdev, type=EV_SYN, code=SYN_REPORT, value=0)
        ]

        return event_list

    @staticmethod
    def swipe(adb_path, x1, y1, x2, y2, duration_ms=500, evdev=GLOBAL_TOUCH_DEVICE, fake_pressure=False, fake_human=False):
        """
        Perform a swipe gesture on the device.
        x1, y1: start coordinates
        x2, y2: end coordinates
        duration_ms: total swipe duration in milliseconds
        steps: number of intermediate points (dots)
        evdev: the device node for your touchscreen (may differ across devices)
        fake_pressure: if True, injects fake pressure values (not always needed or supported by device).
        """
        MotionGenerator.init_provider()
        trace_abstract_finger = MotionGenerator.generate_swipe_trace(x1, y1, x2, y2, duration_ms * 1000, neighbor_time_delta_us=GLOBAL_EVENT_INTERVAL_US, fake_human=fake_human)
        trace_gotevent = MotionGenerator.swipe_to_event_trace(trace=trace_abstract_finger, end_upfinger_time_us=50000, evdev=evdev, fake_pressure=fake_pressure)
        MotionGenerator.flush_event_sequence(adb_path, evdev, trace_gotevent)

    @staticmethod
    def tap(adb_path, x, y, evdev=GLOBAL_TOUCH_DEVICE, duration_us = 100, fake_pressure=False):
        """
        Perform a tap gesture on the device.
        x, y: coordinates to tap
        duration_ms: duration of the tap in milliseconds
        evdev: the device node for your touchscreen (may differ across devices)
        fake_pressure: if True, injects fake pressure values (not always needed or supported by device).
        """
        MotionGenerator.init_provider()
        trace_abstract_finger = [
            FingerEvent(timestamp_us=0, x=x, y=y),
        ]
        trace_gotevent = MotionGenerator.swipe_to_event_trace(trace=trace_abstract_finger, end_upfinger_time_us=duration_us, evdev=evdev, fake_pressure=fake_pressure)
        MotionGenerator.flush_event_sequence(adb_path, evdev, trace_gotevent)




    @staticmethod
    def custom_fake_action_1(adb_path, width = 1080, height=1920, evdev=GLOBAL_TOUCH_DEVICE):
        """
        A short press for 40ms, then move right by 20 pixel in 11ms, then release.
        
        :param adb_path: Description
        :param width: Description
        :param height: Description
        :param evdev: Description
        """
        # choose a random point in the screen center area
        start_x = random.randint(width//4, 3*width//4)
        start_y = random.randint(height//4, 3*height//4)

        fake_action_trace = [
            FingerEvent(timestamp_us=0, x=start_x, y=start_y),
            FingerEvent(timestamp_us=33000, x=start_x, y=start_y),
            FingerEvent(timestamp_us=44000, x=start_x + 200, y=start_y),
            FingerEvent(timestamp_us=55000, x=start_x, y=start_y),
        ]

        trace_gotevent = MotionGenerator.swipe_to_event_trace(trace=fake_action_trace, end_upfinger_time_us=10000, evdev=evdev, fake_pressure=False)
        MotionGenerator.flush_event_sequence(adb_path, evdev, trace_gotevent)

    @staticmethod
    def custom_fake_action_2(adb_path, width = 1080, height=1920, evdev=GLOBAL_TOUCH_DEVICE):
        """
        Draw a square with four edges of length 200 pixels.
        
        :param adb_path: Description
        :param width: Description
        :param height: Description
        :param evdev: Description
        """
        # choose a random point in the screen center area
        start_x = random.randint(width//4, 3*width//4)
        start_y = random.randint(height//4, 3*height//4)


        side = 500
        fake_action_trace = [
            FingerEvent(timestamp_us=0, x=start_x, y=start_y),
            FingerEvent(timestamp_us=22000, x=start_x + side, y=start_y),
            FingerEvent(timestamp_us=33000, x=start_x + side, y=start_y),
            FingerEvent(timestamp_us=55000, x=start_x, y=start_y),
            FingerEvent(timestamp_us=66000, x=start_x, y=start_y),
        ]

        trace_gotevent = MotionGenerator.swipe_to_event_trace(trace=fake_action_trace, end_upfinger_time_us=12000, evdev=evdev, fake_pressure=False)
        MotionGenerator.flush_event_sequence(adb_path, evdev, trace_gotevent)

    tap_position_record_file = str(Path(proj_folder) / "fake_adb" / "tap_position_record.txt") # PASTBUG: was relative path, causing issues when running from other directories e.g. mobile-agent-e
    @staticmethod
    def record_tap_position(x, y):
        with open(MotionGenerator.tap_position_record_file, "w") as f:

            f.write(f"{x},{y}\n")
    
    @staticmethod
    def read_tap_positions() -> List[tuple]:
        positions = []
        try:
            with open(MotionGenerator.tap_position_record_file, "r") as f:
                for line in f:
                    x_str, y_str = line.strip().split(",")
                    positions.append((int(x_str), int(y_str)))
        except FileNotFoundError:
            pass
        return positions

    @staticmethod
    def custom_fake_action_3(adb_path, width = 1080, height=1920, evdev=GLOBAL_TOUCH_DEVICE):
        """
        Draw a circle with radius 200 pixels.
        
        :param adb_path: Description
        :param width: Description
        :param height: Description
        :param evdev: Description
        """
        # choose a random point in the screen center area

        radius = 50
        # center_x = random.randint(width//4 + radius, 3*width//4 - radius)
        # center_y = random.randint(height//4 + radius, 3*height//4 - radius)
        center_x, center_y = MotionGenerator.read_tap_positions()[-1]

        fake_action_trace: List[FingerEvent] = []
        num_points = 36
        time = num_points * GLOBAL_EVENT_INTERVAL_US

        # randomly sample clockwise or counter-clockwise
        angle_direction = random.choice([-1, 1])
        # randomly choose a starting angle
        angle_base = random.uniform(0, 2 * math.pi)

        for i in range(num_points + 1):
            angle = (2 * math.pi / num_points) * i * angle_direction + angle_base
            x = int(center_x + radius * round(math.cos(angle), 5))
            y = int(center_y + radius * round(math.sin(angle), 5))
            timestamp_us = int((GLOBAL_EVENT_INTERVAL_US) * i)
            fake_action_trace.append(FingerEvent(timestamp_us=timestamp_us, x=x, y=y))
        for i in range(5):
            fake_action_trace.append(FingerEvent(timestamp_us=time + (i+1)*GLOBAL_EVENT_INTERVAL_US, x=center_x + radius, y=center_y))            
        trace_gotevent = MotionGenerator.swipe_to_event_trace(trace=fake_action_trace, end_upfinger_time_us=15000, evdev=evdev, fake_pressure=False)
        MotionGenerator.flush_event_sequence(adb_path, evdev, trace_gotevent)
# --- Action Functions ---

def do_type(adb_path: str, text: str, use_adb_keyboard_for_all_keys: bool = False, enable_logging: bool = True):
    """
    The text is typed AS IS, no further escaping.
    
    :param adb_path: a string that may either be adb or adb combined with the device serial thru -s.
    :param text: text to be printed As Is.
    :param use_adb_keyboard_for_all_keys: force use adb keyboard input for all keys, including those that can be sent by adb shell input text.
    :param enable_logging: log each character to IME log with phone timestamp which may not strictly equal the time when the character is received by the destination app.
    """
    for char in text:
        if enable_logging:
            print(f"Logging... {char}")
            log_char(char)
            print(f"Logged {char}")
        
        if use_adb_keyboard_for_all_keys:
            # check whether it should be escaped
            if char in ["\"", "\\", "`"]:
                char = "\\\\\\" + char
            elif char in "\'":
                char = "\\" + char
            elif char == ' ':
                char = "\\ "
            # $ doesn't need to be escaped
            # PASTBUG: did not treat " and else carefully, sometime leading to that the input never gets executed because certain shell thinks you haven't finished typing.
            command = adb_path + f" shell am broadcast -a ADB_INPUT_TEXT --es msg \"{char}\""

        # deal with special characters first
        elif char == "\"":
            # type adb shell input text "\\\"" should output a single double quote for adb; adb shell input text "\"" doesn't work
            # no keycode for double quote???
            command = adb_path + " shell input text \"\\\\\\\"\""
        elif char == "\\":
            command = adb_path + " shell input keyevent 73"  # KEYCODE_BACKSLASH
        elif char == "`":
            command = adb_path + " shell input keyevent 68"  # KEYCODE_GRAVE
        elif char == "\'":
            command = adb_path + " shell input keyevent 75"  # KEYCODE_APOSTROPHE
        elif char == ' ':
            command = adb_path + f" shell input text %s"
        elif char == '\n': # now we require literacy. We also want _ to be correctly printed.
            command = adb_path + f" shell input keyevent 66" # KEYCODE_ENTER 
        # elif char == '':  # backspace; not thoroughly tested. Observed mobile-agent-e trying to send  thru am broadcast; view logger for details. This functionality is actually ill-defined; whether we want to input a backspace character or want to delete a character before the cursor? theoretically, the agent should use this text to input a literal backspace, and use a special function backspac() to actually delete something. Since passing the backspace character to am broadcast actually have no effect, we are right to ignore this.
        #     command = adb_path + f" shell input keyevent 67" # KEYCODE_DEL. The delete key we know better is represented by 112 KEYCODE_FORWARD_DEL

        elif 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char.isdigit():
            command = adb_path + f" shell input text {char}"
        elif char in '-.,!?@\'°/:;()_':
            command = adb_path + f" shell input text \"{char}\""
        else:
            command = adb_path + f" shell am broadcast -a ADB_INPUT_TEXT --es msg \"{char}\""
            
        subprocess.run(command, capture_output=True, text=True, shell=True)

def do_enter(adb_path: str = REAL_ADB):
    log_char("KEYCODE_ENTER")
    command = adb_path + f" shell input keyevent KEYCODE_ENTER"
    MotionGenerator.single_command_that_may_take_time(command)

def do_tap(adb_path: str, x, y, fake_human=GLOBAL_FAKE_HUMAN):
    duration_us = max(int(random.normalvariate(80000, 10000)), 20000) if fake_human else 100
    MotionGenerator.tap(adb_path, x, y, duration_us=duration_us)

def do_swipe(adb_path: str, x1, y1, x2, y2, fake_human=GLOBAL_FAKE_HUMAN, time_ms = None,verbose=False):
    if verbose:
        print(f"Swiping from ({x1}, {y1}) to ({x2}, {y2}), fake_human={fake_human}")
    if not fake_human:
        if time_ms is None:
            raise ValueError("time_ms must be provided when fake_human is False")
        else:
            duration = time_ms
    else:
        duration = 200

    MotionGenerator.swipe(adb_path, x1, y1, x2, y2, duration_ms=duration, evdev=GLOBAL_TOUCH_DEVICE, fake_pressure=False, fake_human=GLOBAL_FAKE_HUMAN)

def do_key_sequence(adb_path: str, code, pressure_duration_us=DEFAULT_PRESS_BTN_TIME_US):
    # HOME key and switch_app_key cannot be captured by jd.
    # command = adb_path + f" shell am start -a android.intent.action.MAIN -c android.intent.category.HOME"
    # command = adb_path + f" shell input keyevent KEYCODE_HOME"
    # subprocess.run(command, capture_output=True, text=True, shell=True)
    device_press_btn = GLOBAL_TOUCH_DEVICE
    sequence = [
        GotEvent(timestamp_us=0, device=device_press_btn, type=EV_KEY, code=code, value=DOWN),
        GotEvent(timestamp_us=0, device=device_press_btn, type=EV_SYN, code=SYN_REPORT, value=0),
        GotEvent(timestamp_us=pressure_duration_us, device=device_press_btn, type=EV_KEY, code=code, value=UP),
        GotEvent(timestamp_us=pressure_duration_us, device=device_press_btn, type=EV_SYN, code=SYN_REPORT, value=0),
    ]
    MotionGenerator.flush_event_sequence(adb_path, device_press_btn, sequence)

def text_restore(arg_s: str) -> str:
    """
    echos the arg_s in terminal and obtain the output to de-escape.
    
    :param arg_s: Description
    :type arg_s: str
    :return: Description
    :rtype: str
    """
    # result = os.popen(f'echo -e "{arg_s.replace(\'"\', \'\\\\"\')}"').read()
    result = os.popen("echo " + arg_s).read()
    return result[:-1] # remove trailing newline

def main():
    args: List[str] = sys.argv[1:]
    adb_path = REAL_ADB
    adb_path_list = [REAL_ADB]
    # print(args)
    try:
        log_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(log_dir, "logger.txt")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Reconstruct the exact invocation
        # try:
        #     invocation = shlex.join(sys.argv)
        # except AttributeError:
            # Fallback for very old Python versions
        if True:
            invocation = " ".join(sys.argv)
        with open(log_path, "a") as lf:
            lf.write(f"{timestamp} {invocation}\n")
    except Exception:
        # Logging must not interfere with normal execution
        pass
    if not args: 
        # args being [] or None
        sys.exit(run_real_adb(adb_path_list, []))
    if len(args) <= 1:
        sys.exit(run_real_adb(adb_path_list, args))
    if args[0] == "-s": # PASTBUG: got -s at the wrong place, thought 1. Phenomenon: with -s, got delegated to real adb.
        # device specified
        if len(args) <= 2: # [-s, serial, command] command length <= 1
            sys.exit(run_real_adb(adb_path_list, args))
        device_serial = args[1]
        adb_path = REAL_ADB + f" -s {device_serial}"
        adb_path_list = [REAL_ADB, "-s", device_serial]
        args = args[2:]

    if len(args) >= 2 and args[0] == "shell" and args[1] == "input":
        if len(args) < 3:
            sys.exit(run_real_adb(adb_path_list, args))
        
        input_type = args[2]
        
        if input_type == "text":
            if len(args) >= 4:
                text = args[3]
                if text == "%s": 
                    text = " " # %s is interpreted as space by adb input
                else:
                    text = text_restore(text)
                do_type(adb_path, text)
                return

        elif input_type == "keyevent":
            if len(args) == 4:
                keycode = args[3]
                if keycode in ["KEYCODE_ENTER", "66"]:
                    do_enter(adb_path)
                elif keycode in ["KEYCODE_BACK", "4"]:
                    do_key_sequence(adb_path, KEY_BACK)
                elif keycode in ["KEYCODE_HOME", "3", "HOME"]:
                    do_key_sequence(adb_path, KEY_HOME)
                elif keycode in ["KEYCODE_APP_SWITCH", "187", "APP_SWITCH", "KEYCODE_MENU"]:
                    do_key_sequence(adb_path, KEY_MENU)
                else:
                    try:
                        keycode_int = int(keycode)
                    except ValueError as e:
                        raise ValueError(f"Unsupported keycode: {keycode}") from e
                    do_key_sequence(adb_path, keycode_int)
            else:
                raise ValueError("keyevent requires one keycode argument; the received args is `{args}`")

        elif input_type == "tap":
            try:   
                if len(args) == 5:
                    x, y = int(args[3]), int(args[4])
                    do_tap(adb_path, x, y)
                    MotionGenerator.record_tap_position(x, y)
                else:
                    raise ValueError
            except ValueError as e: 
                raise ValueError(f"tap requires two integer arguments; received args: `{args}`, {e}")

        elif input_type == "swipe":
            try:    
                if len(args) == 7 or len(args) == 8:
                    x1, y1, x2, y2 = int(args[3]), int(args[4]), int(args[5]), int(args[6])
                    if (len(args) == 8) and not GLOBAL_FAKE_HUMAN:
                        time_ms = int(args[7])
                    else:
                        time_ms = None
                    do_swipe(adb_path, x1, y1, x2, y2, fake_human=GLOBAL_FAKE_HUMAN, time_ms=time_ms)
                else:
                    raise ValueError
            except ValueError as e:
                raise ValueError(f"swipe requires four to five integer arguments; received args: `{args}`, error: {e}")
        elif input_type == "fake":
            try:
                if len(args) == 4:
                    if args[3] == "custom_fake_action_1":
                        MotionGenerator.custom_fake_action_1(adb_path)
                    elif args[3] == "custom_fake_action_2":
                        MotionGenerator.custom_fake_action_2(adb_path)
                    elif args[3] == "custom_fake_action_3":
                        MotionGenerator.custom_fake_action_3(adb_path)
                    elif args[3] == "custom_fake_action_4":
                        MotionGenerator.custom_fake_action_4(adb_path)
                    else:
                        raise ValueError
                else:
                    raise ValueError
            except ValueError as e:
                raise ValueError(f"Unsupported fake action: `{args}`")
        else:
            sys.exit(run_real_adb(adb_path_list, args))
    elif len(args) >= 4 and args[0:7] == ["shell", "am", "broadcast", "-a", "ADB_INPUT_TEXT", "--es", "msg"]:
        # command = adb_path + f" shell am broadcast -a ADB_INPUT_TEXT --es msg \"{char}\""
        text = args[7] # the quotes are automatically stripped by the shell
        # print(text)

        text = text_restore(text)
        do_type(adb_path, text, use_adb_keyboard_for_all_keys=True)
    else:
        time.sleep(2.0) # slight delay to wait for random clicks to settle
        sys.exit(run_real_adb(adb_path_list, args))


LOCK_FILE = str(Path(proj_folder) / "fake_adb" / "adb_wrapper.lock")

if __name__ == "__main__":
    # Open lock file
    with open(LOCK_FILE, "w") as lock_file:
        try:
            # Acquire an exclusive, blocking lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            # Now safe to run main()
            main()
        finally:
            # Lock is automatically released when file is closed,
            # but we ensure cleanup
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)