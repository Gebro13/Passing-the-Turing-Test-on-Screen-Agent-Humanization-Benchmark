# input and output specified in --help.


# from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd

from motionevent_classes import FingerEvent
from feature_library import extract_features
from gesture_log_reader_utils import filtered_gesture_generator_from_files_no_timestamp

def cleanse_into_swipe(swipe: List[FingerEvent]) -> Optional[List[FingerEvent]]:
    """
        Keep only the swipes and remove the last point.
    """
    # Remove points with None coordinates
    if swipe is None:
        return None
    swipe = swipe[:-1] if len(swipe) > 1 else swipe
    # Remove points with None coordinates
    swipe = [e for e in swipe if (e.x is not None and e.y is not None)]
    if len(swipe) > 5:
        return swipe
    return None


def build_features_dataframe(one_gesture_generator_for_each_file: List[Tuple[str, List[List[FingerEvent]]]]) -> pd.DataFrame:
    """
    iterate swipes, extract swipe features, and return a DataFrame.
    
    :param one_gesture_generator_for_each_file: (label_to_insert_into_dataframe, swipe_iterator) for each file.
    :return: The resulting DataFrame has a 'type' column followed by feature columns.
    """
    all_rows: List[pd.DataFrame] = []

    for label, swipe_iterator in one_gesture_generator_for_each_file:
        features: List[Dict[str, float]] = [
            extract_features(swipe) for swipe in swipe_iterator
        ]
        df_feat = pd.DataFrame(features) # a df for each file, containing all features but no label
        df_feat.insert(0, "type", label)
        all_rows.append(df_feat)

    if not all_rows:
        return pd.DataFrame(columns=["type"])  # empty

    df = pd.concat(all_rows, ignore_index=True)
    # Ensure 'type' is the first column
    cols = ["type"] + [c for c in df.columns if c != "type"]
    return df[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract swipe features into a DataFrame from logs and labels.")
    parser.add_argument("--index_csv", type=str, default=str(Path(__file__).with_name("logs_inf.csv")), help="Path to logs_inf.csv")
    parser.add_argument("--logs_dir", type=str, default=str(Path(__file__).with_name("logs")), help="Directory containing gesture_recording_*.log files")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save the resulting CSV")
    args = parser.parse_args()

    logs_index_csv = Path(args.index_csv)
    logs_dir = Path(args.logs_dir)

    df = build_features_dataframe(
        filtered_gesture_generator_from_files_no_timestamp(
            pd.read_csv(logs_index_csv), logs_dir, cleanse_into_swipe)
    )
    print(f"Built DataFrame with shape {df.shape}")

    if args.save_csv:
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
