from typing import Optional, List, Dict, Tuple, Callable
import re
from re import Match
from analysis.lib.motionevent_classes import FingerEvent, SessionType
from pathlib import Path
import pandas as pd
import os
from functools import partial

def hex_to_dec(hex_str):
    return int(hex_str, 16)

def file_finder(search_scope: Path, file_name: str) -> Path:
    """
    Recursively search for files named exactly `file_name` under `search_scope`.

    Returns:
        The single matching pathlib.Path.

    Raises:
        FileNotFoundError: If search_scope doesn't exist or no file is found.
        NotADirectoryError: If search_scope is not a directory.
        FileExistsError: If multiple files with the same name are found.
    """
    scope = search_scope
    if not scope.exists():
        raise FileNotFoundError(f"Search scope does not exist: {scope}")
    if not scope.is_dir():
        raise NotADirectoryError(f"Search scope is not a directory: {scope}")

    target_name = Path(file_name).name
    matches: List[Path] = []
    seen_dirs = set()

    for root, dirnames, filenames in os.walk(scope, topdown=True, followlinks=True):
        real_root = os.path.realpath(root)
        if real_root in seen_dirs:
            # prevent infinite loops caused by symlink cycles
            dirnames[:] = []
            continue
        seen_dirs.add(real_root)

        for fname in filenames:
            if fname == target_name:
                p = Path(root) / fname
                try:
                    if p.is_file() or p.is_symlink():
                        matches.append(p)
                except Exception:
                    # skip paths that cannot be accessed
                    pass

    if not matches:
        raise FileNotFoundError(f"No file named '{target_name}' found under {scope}")
    if len(matches) > 1:
        listed = "\n".join(str(m) for m in matches)
        raise FileExistsError(f"Multiple files named '{target_name}' found under {scope}:\n{listed}")
    return matches[0]

def file_reader_yield(file_path: str) -> List[str]:
    """
    Generator to read a file line by line.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def single_trace_generator(adb_getevent_generator: List[str], thres_after_1st_menu_s: Optional[int] = None) -> List[List[FingerEvent]]:
    """disappearing point is included, which actually disrupts the trace"""
    current_gesture = []
    current_id = None
    x, y, t_us = None, None, None

    id_pattern = re.compile(r'ABS_MT_TRACKING_ID\s+([0-9a-f]+)')
    x_pattern = re.compile(r'ABS_MT_POSITION_X\s+([0-9a-f]+)')
    y_pattern = re.compile(r'ABS_MT_POSITION_Y\s+([0-9a-f]+)')
    time_pattern = re.compile(r'^\[ {0,8}([0-9]+)\.([0-9]+)\]')
    
    time_start_us = None

    result: List[List[FingerEvent]] = []

    for line in adb_getevent_generator:
        id_match = id_pattern.search(line)
        x_match = x_pattern.search(line)
        y_match = y_pattern.search(line)
        time_match: Match[str] = time_pattern.search(line)
        if (thres_after_1st_menu_s is not None) and (time_start_us is None) and ("MENU" in line):
            time_start_us = int(time_match.group(1)) * int(1e6) + int(time_match.group(2))
            time_end_us = time_start_us + thres_after_1st_menu_s * int(1e6)
            print(f"First MENU detected at {time_start_us} us, threshold set to {time_end_us} microseconds.")
        if (time_start_us is not None) and (time_match is not None):
            t_us = int(time_match.group(1)) * int(1e6) + int(time_match.group(2))
            if t_us > time_end_us:
                # If the current event is beyond the threshold, we stop the gesture.
                if current_gesture:
                    result.append(current_gesture)
                current_gesture = []
                break

        if id_match:
            tracking_id = id_match.group(1)
            if tracking_id != 'ffffffff':
                # Finger pressed down — start a new gesture
                current_id = tracking_id
                current_gesture = []
            else:
                # Finger lifted — current gesture ends
                if current_gesture:
                    if time_match:
                        # add disappearing point, which actually disrupts the trace.
                        t_us = int(time_match.group(1)) * int(1e6) + int(time_match.group(2))
                        current_gesture.append(FingerEvent(timestamp_us=t_us, x=x, y=y))
                    result.append(current_gesture)
                current_id = None
                current_gesture = []

        if current_id is not None:
            if x_match:
                x: int = hex_to_dec(x_match.group(1))
            if y_match:
                y: int = hex_to_dec(y_match.group(1))
            if time_match:
                t_us: int = int(time_match.group(1)) * int(1e6) + int(time_match.group(2))

            # SYN_REPORT indicates the current sample is completed
            if "SYN_REPORT" in line: # and x is not None and y is not None:
                current_gesture.append(FingerEvent(timestamp_us=t_us, x=x, y=y))
                # x, y = None, None shouldn't reset the saved values
    return result



def gesture_generator_from_files(df_idx: pd.DataFrame, logs_dir: Path) -> List[Tuple[str, str, List[List[FingerEvent]]]]:
    """Read df_idx(the catalog of logs), iterate logs, extract unmodified swipes, and return a List of Lists.
    The resulting DataFrame has a 'type' column followed by feature columns.
    resuiting list have: (participant, session_time_stamp, gestures)
    """
    result: List[Tuple[str, str, List[List[FingerEvent]]]] = []
    for _, row in df_idx.iterrows():
        log_num: str = str(row["log_num"])  # e.g., 20250714_162513
        label: str = str(row["type"])      # class label
        # Construct file name like draw_motion_event2 defaults
        log_file = logs_dir / f"gesture_recording_{log_num}.log"
        if not log_file.exists():
            print(f"Missing log: {log_file}")
            continue

        # Compute threshold if both timestamps are present
        total_len = row.get("total_video_length_s", None)
        first_tap = row.get("first_tap_time_s", None)
        thres: Optional[int] = None
        if pd.notnull(total_len) and pd.notnull(first_tap):
            try:
                thres = int(total_len) - int(first_tap)
            except Exception:
                thres = None
        result.append((label, log_num, single_trace_generator(file_reader_yield(str(log_file)), thres_after_1st_menu_s=thres)))
    return result

def filtered_gestures_generation(
        gesture_iterator: List[List[FingerEvent]], 
        filtering_and_modification_function: Callable[[List[FingerEvent]], Optional[List[FingerEvent]]]
    ) -> List[List[FingerEvent]]:
    """Apply filtering_and_modification_function to each swipe list and return a filtered list. If None, skip."""
    result: List[List[FingerEvent]] = []
    for gesture in gesture_iterator:
        filtered_gesture = filtering_and_modification_function(gesture)
        if filtered_gesture is not None:
            result.append(filtered_gesture)
    return result


def filtered_gesture_generator_from_files(
    df_idx: pd.DataFrame,
    logs_dir: Path,
    filtering_and_modification_function: Callable[[List[FingerEvent]], Optional[List[FingerEvent]]],
) -> List[Tuple[str, str, List[List[FingerEvent]]]]:
    """
    for df_idx, generate filtered gestures from files in logs_dir using filtering_and_modification_function.
    
    :param df_idx: 
    :type df_idx: pd.DataFrame
    :param logs_dir: 
    :type logs_dir: Path
    :param filtering_and_modification_function: Description
    :type filtering_and_modification_function: Callable[[List[FingerEvent]], Optional[List[FingerEvent]]]
    :return: A list of tuples containing participant tag, session timestamp, and filtered gestures; one tuple per file.
    :rtype: List[Tuple[str, str, List[List[FingerEvent]]]]
    """

    result: List[Tuple[str, str, List[List[FingerEvent]]]] = []
    for participant, session_timestamp, swipe_iterator in gesture_generator_from_files(df_idx, logs_dir):
        # skip the nullified swipes
        filtered_swipe_iterator = filtered_gestures_generation(swipe_iterator, filtering_and_modification_function)
        result.append((participant, session_timestamp, filtered_swipe_iterator))
    return result

def filtered_gesture_generator_from_files_no_timestamp(
    df_idx: pd.DataFrame,
    logs_dir: Path,
    filtering_and_modification_function: Callable[[List[FingerEvent]], Optional[List[FingerEvent]]],
) -> List[Tuple[str, List[List[FingerEvent]]]]:
    return [(tupling[0], tupling[2]) for tupling in filtered_gesture_generator_from_files(
        df_idx, logs_dir, filtering_and_modification_function
    )]

def check_integrity_of_logs(df_idx: pd.DataFrame, logs_dir: Path) -> None:
    # Check if all log files listed in df_idx exist in logs_dir.
    missing_logs = []
    for _, row in df_idx.iterrows():
        session_timestamp: str = str(row["log_num"])  # e.g., 20250714_162513
        log_file = logs_dir / f"gesture_recording_{session_timestamp}.log"
        if not log_file.exists():
            missing_logs.append(log_file)
    if missing_logs:
        print("Missing log files:")
        for log in missing_logs:
            print(f" - {log}")
    else:
        print("All log files are present.")

    # check that every yielded swipe is not None
    for participant_tag, session_timestamp, swipe_iterator in gesture_generator_from_files(df_idx, logs_dir):
        for swipe in swipe_iterator:
            if swipe is None:
                raise ValueError(f"Null swipe detected in label {participant_tag}. These guards are useful after all.")

            # check that every FingerEvent has no None attributes
            for event in swipe:
                if event is None or event.timestamp_us is None or event.x is None or event.y is None:
                    raise ValueError(f"Invalid FingerEvent detected in label {participant_tag}: {event}")

def stack_iterator_with_str(label: str, items: List[List[FingerEvent]]) -> List[Tuple[str, List[FingerEvent]]]:
    """Stack a label alongside each item in a list."""
    return [(label, item) for item in items]

def ranged_batched_modified_generator_with_session_timestamp(
        formated_data_timestamps: pd.DataFrame, 
        participants: List[str], 
        index_range: Optional[List[int]] = None, 
        filtering_and_modification_function: Optional[
            Callable[[List[List[FingerEvent]]], List[List[FingerEvent]]]
        ] = None, 
        humanity_disturbance: Optional[
            Callable[[List[FingerEvent]], List[FingerEvent]]
        ] = None
        ) -> List[SessionType]:
    """
    formated_data_timestamps: DataFrame with participant names as columns and timestamps as values
    """
    result: List[SessionType] = []
    for participant_name in participants:
        for index, timestamp in enumerate(formated_data_timestamps[participant_name]):
            #print(participant_name, index, timestamp)
            if (index_range is not None) and (index not in index_range):
                continue
            stripped_timestamp = str(timestamp).strip()
            if (stripped_timestamp == ""):
                continue
            #print(stripped_timestamp)
            #continue
            try:
                file_path = file_finder(Path("logs/"), "gesture_recording_" + stripped_timestamp + ".log")
            except FileNotFoundError as e:
                print(e)
                continue
            line_generator = file_reader_yield(str(file_path))
            single_trace_generated = single_trace_generator(
                line_generator, None
            )
            if filtering_and_modification_function is not None:
                swipe_generated = filtering_and_modification_function(single_trace_generated)
            else:
                swipe_generated = single_trace_generated
            if humanity_disturbance is not None:
                swipe_generated = [
                    humanity_disturbance(swipe_without_final_point)
                    for swipe_without_final_point in swipe_generated
                ]
            result.append((stripped_timestamp, swipe_generated))
    return result


def ranged_batched_modified_generator_without_session_timestamp(
        formated_data_timestamps: pd.DataFrame, 
        participants: List[str], 
        index_range: Optional[List[int]] = None, 
        filtering_and_modification_function: Optional[
            Callable[[List[List[FingerEvent]]], List[List[FingerEvent]]]
        ] = None, 
        humanity_disturbance: Optional[
            Callable[[List[FingerEvent]], List[FingerEvent]]
        ] = None
        ) -> List[List[FingerEvent]]:
    """
    formated_data_timestamps: DataFrame with participant names as columns and timestamps as values
    """
    result_with_timestamps = ranged_batched_modified_generator_with_session_timestamp(
            formated_data_timestamps=formated_data_timestamps,
            participants=participants,
            index_range=index_range,
            filtering_and_modification_function=filtering_and_modification_function,
            humanity_disturbance=humanity_disturbance
        )
    result: List[List[FingerEvent]] = []
    for timestamp, swipes in result_with_timestamps:
        result += swipes
    return result

def ranged_modified_generator_without_session_timestamp(
        formated_data_timestamps: pd.DataFrame, 
        participants: List[str], 
        index_range: Optional[List[int]] = None, 
        filtering_and_modification_function: Optional[
            Callable[[List[FingerEvent]], Optional[List[FingerEvent]]]
        ] = None, 
        humanity_disturbance: Optional[
            Callable[[List[FingerEvent]], List[FingerEvent]]
        ] = None
        ) -> List[List[FingerEvent]]:
    """
    formated_data_timestamps: DataFrame with participant names as columns and timestamps as values
    """
    if filtering_and_modification_function is not None:
        batched_modifying_function = partial(
            filtered_gestures_generation, 
            filtering_and_modification_function=filtering_and_modification_function
        )
    else:
        batched_modifying_function = None
        
    return ranged_batched_modified_generator_without_session_timestamp(
            formated_data_timestamps=formated_data_timestamps,
            participants=participants,
            index_range=index_range,
            filtering_and_modification_function=batched_modifying_function,
            humanity_disturbance=humanity_disturbance
        )

def ranged_modified_generator_with_session_timestamp(
        formated_data_timestamps: pd.DataFrame, 
        participants: List[str], 
        index_range: Optional[List[int]] = None, 
        filtering_and_modification_function: Optional[
            Callable[[List[FingerEvent]], Optional[List[FingerEvent]]]
        ] = None, 
        humanity_disturbance: Optional[
            Callable[[List[FingerEvent]], List[FingerEvent]]
        ] = None
        ) -> List[SessionType]:
    """
    formated_data_timestamps: DataFrame with participant names as columns and timestamps as values
    """
    if filtering_and_modification_function is not None:
        batched_modifying_function = partial(
            filtered_gestures_generation, 
            filtering_and_modification_function=filtering_and_modification_function
        )
    else:
        batched_modifying_function = None
        
    return ranged_batched_modified_generator_with_session_timestamp(
            formated_data_timestamps=formated_data_timestamps,
            participants=participants,
            index_range=index_range,
            filtering_and_modification_function=batched_modifying_function,
            humanity_disturbance=humanity_disturbance
        )

def rectify_timestamp_idx(
        formatted_data_timestamps_df: pd.DataFrame,
        participants: List[str],
        orthodox_data_regex: str = r'\d{8}_\d{6}',
    ) -> Dict[str, List[Tuple[int, str]]]:
    """
    Convert a DataFrame of formatted data timestamps into a dictionary mapping participant names to lists of (task_idx, timestamp) tuples.
    For participants not found in the DataFrame columns, an empty list is assigned.
    For errorneous entries (e.g., NaN or empty strings or illegal strings), those entries are skipped.
    
    :param formatted_data_timestamps_df: dataframe with participant names as columns and timestamps as values
    :type formatted_data_timestamps_df: pd.DataFrame
    :param participants: List of participant names to include; used to extract the columns of formatted_data_timestamps_df
    :return:  {participant: [(task_idx, timestamp), ...] }
    :rtype: Dict[str, List[Tuple[int, str]]]
    """
    
    result: Dict[str, List[Tuple[int, str]]] = {}
    for participant in participants:
        if participant not in formatted_data_timestamps_df.columns:
            result[participant] = []
            continue
            # raise ValueError(f"Participant {participant} not found in DataFrame columns.")
        timestamps = formatted_data_timestamps_df[participant].dropna().astype(str)
        # read timestamps as a series
        result_participant: List[Tuple[int, str]] = [(idx, ts.strip()) for idx, ts in timestamps.items() if (re.fullmatch(orthodox_data_regex, ts.strip()) is not None)]
        result[participant] = result_participant
    return result

if __name__ == "__main__":
    print("Temporarily using this file's main execution as log sanity check.")
    check_integrity_of_logs(
        pd.read_csv(Path(__file__).with_name("logs_inf.csv")),
        Path(__file__).parent / "logs"
    )

