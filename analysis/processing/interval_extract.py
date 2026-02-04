from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Literal, Iterator

import pandas as pd

from analysis.lib.motionevent_classes import FingerEvent, SessionType
from analysis.lib.gesture_log_reader_utils import filtered_gesture_generator_from_files
import random

def null_object_warning() -> None:
    print("Null item detected. These guards are useful after all.")

def _gesture_start_end_us(gesture: List[FingerEvent]) -> Tuple[int, int]:
    """Return (start_us, end_us) for a gesture, ignoring events without timestamps.
    Removes the artificially appended disappearing point if present (last point),
    and keeps taps by not enforcing a minimum length filter.
    """
    return gesture[0].timestamp_us, gesture[-1].timestamp_us

def filter_null(gesture: List[FingerEvent]) -> Optional[List[FingerEvent]]:
    """currently an identity function."""
    return gesture # actually nothing needs to be done

def poisson_generator(target_interval_us: float) -> Iterator[int]:
    """Generate timestamps starting from 0 with intervals drawn from an exponential distribution with mean target_interval_us."""
    t: int = 0
    lambd = 1.0 / target_interval_us
    while True:
        step = int(random.expovariate(lambd))
        t += step
        yield t

def fake_timestamps_between_start_and_end(start_us: int, end_us: int, target_interval_us: int) -> List[int]:
    """Generate fake timestamps between start_us and end_us such that the average interval is target_interval_us.

    Uses exponentially distributed steps (Poisson process) until end_us.
    """
    timestamps: List[int] = []
    generator = poisson_generator(target_interval_us)
    upper_limit = end_us - start_us
    for timestamp in generator:
        if timestamp >= upper_limit:
            break
        timestamps.append(timestamp + start_us)
    return timestamps

def faking_intervals_between_gestures(gesture_generator: List[List[FingerEvent]], target_expected_interval_us: int, fake_method: Literal["ghost_operation"] = "ghost_operation") -> List[List[FingerEvent]]:
    """Given a gesture list, return a new list with faked intervals between them.
    Each gesture's timestamps are shifted such that there is `interval_us` microseconds
    between the end of one gesture and the start of the next.
    """
    result: List[List[FingerEvent]] = []
    gestures = gesture_generator
    if (len(gestures) == 0):
        return []
    start_ends: List[Tuple[int, int]] = [_gesture_start_end_us(gesture) for gesture in gestures]
    result.append(gestures[0])
    for i in range(1, len(start_ends)):
        prev_end = start_ends[i - 1][1]
        current_start = start_ends[i][0]
        # generate many fake timestamps to fill the gap in a poisson-distribution manner so that the average interval is target_expected_interval_us

        time_stamp_list = fake_timestamps_between_start_and_end(prev_end, current_start, target_expected_interval_us)

        for time_stamp in time_stamp_list:
            result.append([FingerEvent(timestamp_us=time_stamp, x=None, y=None),
                           FingerEvent(timestamp_us=time_stamp + 11000, x=None, y=None)])  # fill in x, y not appropriately because it has no meaning here 
        result.append(gestures[i])
    return result
    

def build_intervals_dataframe(filtered_gesture_generator: List[Tuple[str, List[SessionType]]]) -> pd.DataFrame:
    """
       For each labeled gesture list, compute time_interval_us between consecutive gestures.
    
    :param filtered_gesture_generator: Description
    :type filtered_gesture_generator: List[Tuple[str, List[SessionType]]]
    :return: A dataframe with columns ["type", "interval_us"]
    :rtype: DataFrame
    """
    rows: List[Dict[str, int | str]] = []

    for label, session_generator in filtered_gesture_generator:
        for session_timestamp, gestures in session_generator:
            prev_end_us: Optional[int] = None
            for gesture in gestures:
                start_us, end_us = _gesture_start_end_us(gesture)
                if start_us is None or end_us is None:
                    null_object_warning()
                    continue
                if prev_end_us is not None:
                    dt = start_us - prev_end_us
                    if dt >= 0:
                        rows.append({"type": label, "interval_us": int(dt)})
                prev_end_us = end_us

    if not rows:
        null_object_warning()
        return pd.DataFrame(columns=["type", "interval_us"])  # empty

    return pd.DataFrame(rows, columns=["type", "interval_us"])  # enforce column order


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract time intervals (microseconds) between gestures from logs.")
    parser.add_argument("--index_csv", type=str, default=str(Path(__file__).with_name("logs_inf.csv")), help="Path to logs_inf.csv")
    parser.add_argument("--logs_dir", type=str, default=str(Path(__file__).with_name("logs")), help="Directory containing gesture_recording_*.log files")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save the resulting CSV with columns [type, interval_us]")
    args = parser.parse_args()

    logs_index_csv = Path(args.index_csv)
    logs_dir = Path(args.logs_dir)


    df_idx = pd.read_csv(logs_index_csv)
    filtered_gesture_generator = filtered_gesture_generator_from_files(df_idx=df_idx, logs_dir=logs_dir, filtering_and_modification_function=filter_null)
    raise NotImplementedError("filtered_gesture_generator_from_files's participant labels need to be aggregated to work.")
    df = build_intervals_dataframe(filtered_gesture_generator)
    print(f"Built intervals DataFrame with shape {df.shape}")

    if args.save_csv:
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
