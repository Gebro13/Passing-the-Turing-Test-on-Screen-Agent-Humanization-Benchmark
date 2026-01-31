from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd

from motionevent_classes import FingerEvent
from gesture_log_reader_utils import filtered_gesture_generator_from_files_no_timestamp
from functools import partial

def null_object_warning() -> None:
    print("Null item detected. These guards are useful after all.")

def _gesture_start_end_us(gesture: List[FingerEvent]) -> Tuple[int, int]:
    """Return (start_us, end_us) for a gesture, ignoring events without timestamps.
    Removes the artificially appended disappearing point if present (last point),
    and keeps taps by not enforcing a minimum length filter.
    """
    return gesture[0].timestamp_us, gesture[-1].timestamp_us

def is_tap(gesture: List[FingerEvent], tap_len_max: int = 5) -> bool:
    """Check if a gesture is a tap, defined as length <= tap_len_max."""
    return len(gesture) <= tap_len_max


def leave_taps(gesture: List[FingerEvent], tap_len_max: int = 5) -> Optional[List[FingerEvent]]: 
    """Leave only taps, defined as gestures with length <= tap_len_max. Does not remove last point."""
    if (is_tap(gesture, tap_len_max=tap_len_max)):
        return gesture # actually nothing needs to be done
    else:
        return None

def build_durations_dataframe(filtered_gesture_generator: List[Tuple[str, List[List[FingerEvent]]]]) -> pd.DataFrame:
    """For each log in logs_inf.csv, compute time_duration_us between consecutive gestures.

    - time_duration_us = start_time_us(current_gesture) - end_time_us(previous_gesture)
    - Keeps taps (length filter)
    - Returns a DataFrame with columns: type (label), duration_us (int, microseconds)
    """
    rows: List[Dict[str, int | str]] = []

    for label, gesture_iterator in filtered_gesture_generator:
        for gesture in gesture_iterator:
            start_us, end_us = _gesture_start_end_us(gesture)
            dt = end_us - start_us
            rows.append({"type": label, "duration_us": int(dt)})
    if not rows:
        null_object_warning()
        return pd.DataFrame(columns=["type", "duration_us"])  # empty

    return pd.DataFrame(rows, columns=["type", "duration_us"])  # enforce column order


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract time durations (microseconds) between gestures from logs.")
    parser.add_argument("--index_csv", type=str, default=str(Path(__file__).with_name("logs_inf.csv")), help="Path to logs_inf.csv")
    parser.add_argument("--logs_dir", type=str, default=str(Path(__file__).with_name("logs")), help="Directory containing gesture_recording_*.log files")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save the resulting CSV with columns [type, duration_us]")
    args = parser.parse_args()

    logs_index_csv = Path(args.index_csv)
    logs_dir = Path(args.logs_dir)

    df_idx = pd.read_csv(logs_index_csv)
    filtered_gesture_generator = filtered_gesture_generator_from_files_no_timestamp(df_idx=df_idx, logs_dir=logs_dir, filtering_and_modification_function=partial(leave_taps, tap_len_max=3))

    df = build_durations_dataframe(filtered_gesture_generator)
    print(f"Built durations DataFrame with shape {df.shape}")

    if args.save_csv:
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
