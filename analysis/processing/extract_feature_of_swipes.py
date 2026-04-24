# input and output specified in --help.


# from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Tuple, TypedDict

import pandas as pd

from analysis.lib.motionevent_classes import FingerEvent, SingularActionType, is_integral, SessionType, SwipeFeaturedSessionType
from analysis.lib.feature_library import extract_features, PhysicallyCorrectSingleSwipeType, transform_to_physically_correct_single_swipe_type, is_tap
from analysis.lib.gesture_log_reader_utils import filtered_gesture_generator_from_files_no_timestamp


def deprecated_cleanse_into_swipe(swipe: SingularActionType) -> Optional[PhysicallyCorrectSingleSwipeType]:
    """
        Keep only the swipes and remove the last point.
        Deprecated: When extraction features from SingularActionTypes, feature library's extract_features will automatically handle the vanishing point.  
    """
    # Remove points with None coordinates
    if swipe is None:
        return None
    if not is_integral(swipe):
        raise NotImplementedError("Currently only integral swipes are supported. Found non-integral swipe. Perhaps only the last point is None; You should examine this.")
    # Remove points with None coordinates
    if not is_tap(gesture=swipe):
        return transform_to_physically_correct_single_swipe_type(swipe)
    return None

def keep_swipe(gesture: SingularActionType) -> Optional[SingularActionType]:
    """
        Keep only the swipes. No change to content.
    """
    if is_tap(gesture=gesture):
        return None
    return gesture


def build_features_dataframe(one_gesture_generator_for_each_file: List[Tuple[str, List[SingularActionType]]]) -> pd.DataFrame:
    """
    iterate swipes, extract swipe features, and return a DataFrame.  
    Ensure that the order of rows in the resulting DataFrame is the same as the order of swipes in the input generators. This is important for any time-series analysis that may be performed later, as well as for correctly associating labels with features if needed.
    
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


def build_features_dataframe_for_sessions(one_swipe_generator_for_each_file: SessionType) -> SwipeFeaturedSessionType:
    """
    iterate swipes, extract swipe features, and return a DataFrame.
    
    :param one_swipe_generator_for_each_file: (session_id, label_to_insert_into_dataframe, swipe_iterator) for each file.
    :return: The resulting DataFrame component consist of swipe features vector for each row and strictly increase index as time progresses.
    """
    session_id, swipe_iterator = one_swipe_generator_for_each_file
    features: List[Dict[str, float]] = [
        extract_features(swipe) for swipe in swipe_iterator
    ]
    df_feat = pd.DataFrame(features) # a df for each file, containing all features but no label
    return SwipeFeaturedSessionType(session_id=session_id, features=df_feat) #.to_dict(orient="list"))



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
            pd.read_csv(logs_index_csv), logs_dir, keep_swipe)
    )
    print(f"Built DataFrame with shape {df.shape}")

    if args.save_csv:
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
