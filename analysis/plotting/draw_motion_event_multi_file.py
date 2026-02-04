from analysis.lib.gesture_log_reader_utils import filtered_gesture_generator_from_files
import argparse
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from analysis.lib.motionevent_classes import FingerEvent
from analysis.processing.extract_feature_of_swipes import cleanse_into_swipe
from analysis.processing.tap_duration_extract import leave_taps
from analysis.processing.interval_extract import filter_null
from analysis.plotting.draw_motion_event2 import throw_away_timestamp, plot_gestures
from functools import partial
import matplotlib.pyplot as plt

def chain_gesture_iterators_filtered(filter_type: str, gesture_iterator: List[Tuple[str, str, List[List[FingerEvent]]]]) -> List[List[FingerEvent]]:
    result: List[List[FingerEvent]] = []
    for gesture_name, session_timestamp, gesture in gesture_iterator:
        if filter_type in gesture_name:
            result += gesture
    return result

def filterless_gesture_iterators(gesture_iterator: List[Tuple[str, str, List[List[FingerEvent]]]]) -> List[List[FingerEvent]]:
    result: List[List[FingerEvent]] = []
    for gesture_name, session_timestamp, gesture in gesture_iterator:
        result += gesture
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", dest="type", type=str, default=None)
    parser.add_argument("--index_csv", type=str, default=str(Path(__file__).with_name("logs_inf.csv")), help="Path to logs_inf.csv")
    parser.add_argument("--logs_dir", type=str, default=str(Path(__file__).with_name("logs")), help="Directory containing gesture_recording_*.log files")
    parser.add_argument("--gesture_target", type=str, choices=["swipe", "tap"], default=None, help="Type of gesture to plot, e.g., 'swipe' or 'tap'. If not specified, plots all gestures.")

    args = parser.parse_args()

    if (args.gesture_target == "swipe"):
        gesture_filter = cleanse_into_swipe
    elif (args.gesture_target == "tap"):
        gesture_filter = partial(leave_taps, tap_len_max=3)
    else:
        gesture_filter = filter_null

    gesture_generator = filtered_gesture_generator_from_files(
        pd.read_csv(args.index_csv),
        Path(args.logs_dir),
        gesture_filter
    )

    if (args.type is None):
        chained_gesture_generator = filterless_gesture_iterators(gesture_generator)
    else:
        chained_gesture_generator = chain_gesture_iterators_filtered(args.type, gesture_generator)

    plot_gestures(throw_away_timestamp(chained_gesture_generator), have_legend=False)
    
    # save the image
    save_str_name = f"gestures_{args.type}_{args.gesture_target}.png"
    plt.savefig(save_str_name)
    print(f"Saved figure as {save_str_name}")