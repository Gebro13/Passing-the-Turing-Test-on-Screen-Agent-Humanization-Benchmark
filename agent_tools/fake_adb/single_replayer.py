# input arg1 is a .txt like result from adb shell -t -t getevent -lt


import adb_wrapper
from pathlib import Path

from analysis.lib.gesture_log_reader_utils import single_trace_generator, file_reader_yield

def replay_single_trace_from_file(file_path: Path):
    trace_generator = single_trace_generator(file_reader_yield(file_path))
    # print(trace_generator)
    for trace in trace_generator:
        event_list = adb_wrapper.MotionGenerator.swipe_to_event_trace(trace=trace, evdev=adb_wrapper.GLOBAL_TOUCH_DEVICE)
        adb_wrapper.MotionGenerator.flush_event_sequence("adb", adb_wrapper.GLOBAL_TOUCH_DEVICE, event_list)

if __name__ == "__main__":
    replay_single_trace_from_file(Path("single_trace.txt"))