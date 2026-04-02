from typing import Final, Optional, List, Dict, Tuple, Callable, Union
import re
from re import Match
from unittest import result
from analysis.lib.motionevent_classes import FingerEvent, SessionType, SingularActionType, is_integral
from pathlib import Path
import pandas as pd
import os
from functools import partial

def hex_to_dec(hex_str: str) -> int:
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

def file_reader_yield(file_path: Union[str, Path]) -> List[str]:
    """
    Generator to read a file line by line.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

def single_trace_generator(adb_getevent_generator: List[str]) -> List[SingularActionType]:
    """
        Parse the output of adb getevent -lt and generate a list of traces, where each trace is a list of FingerEvent objects.
        Only support single-finger traces.
        Only support format B.
        disappearing point is included, which actually disrupts the trace.
        However, the disappearing point(the last point) is existent in the app collection with same x and y so we keep it.  
    """
    current_gesture: List[FingerEvent] = []
    x: Optional[int] = None
    y: Optional[int] = None
    t_us: Optional[int] = None
    TERMINATION_ID: Final[str] = 'ffffffff' # the id indicating the finger is lifted, which is inherited by the next trace if not replaced by a new id.
    inherited_id: str = TERMINATION_ID # the id inherited from the previous trace, which is TERMINATION_ID if no previous trace or the previous trace has been terminated. This is used to determine whether a new trace starts or the current trace ends when we encounter a SYN_REPORT line.
    last_is_syn_report: bool = True # only used to examine timestamp discrepancies.
    last_is_key_event: bool = False # only used to detect unexpected key events in the log, which are not disruptive but may indicate some unexpected log format.
    fleeting_id: str = "" # the id in the current syn_report block

    exist_incomplete_sample: bool = False

    id_pattern = re.compile(r'EV_ABS       ABS_MT_TRACKING_ID   ([0-9a-f]{8,8})')
    x_pattern  = re.compile(r'EV_ABS       ABS_MT_POSITION_X    ([0-9a-f]{8,8})')
    y_pattern  = re.compile(r'EV_ABS       ABS_MT_POSITION_Y    ([0-9a-f]{8,8})')
    time_pattern = re.compile(r'^\[ {0,8}([0-9]{0,8})\.([0-9]{6,6})\]')

    # currently useless pattern
    button_pattern = re.compile(r'EV_KEY       BTN_TOUCH            (DOWN|UP)') 

    # strange patterns that we should discard
    key_pattern = re.compile(r'EV_KEY       KEY_([0-9A-Z]+)') # we only care about ABS_MT events, so lines with KEY_ are unexpected and should be discarded.
    multitouch_pattern = re.compile(r'ABS_MT_SLOT\s+([0-9]+)') # we only support single-finger traces, so lines with ABS_MT_SLOT are unexpected and should be discarded.

    result: List[SingularActionType] = []

    for line in adb_getevent_generator:

        time_match: Optional[Match[str]] = time_pattern.search(line)
        if time_match is None:
            continue # not a valid line, which is actually rather common at the geginning of a log.

        id_match = id_pattern.search(line)
        x_match = x_pattern.search(line)
        y_match = y_pattern.search(line)
        syn_report_in_line = "SYN_REPORT" in line

        button_match = button_pattern.search(line)

        key_match = key_pattern.search(line)
        syn_config_in_line = "SYN_CONFIG" in line # SYN_CONFIG lines are unexpected in our logs and may indicate some unexpected log format, but we don't raise an error for them because they don't disrupt the trace generation.
        multitouch_match = multitouch_pattern.search(line)


        if key_match is not None:
            # print(f"Warning: EV_KEY event found in line, which is unexpected but not disruptive: {line}")
            pass
        if syn_config_in_line:
            print(f"Warning: SYN_CONFIG found in line, which is unexpected but not disruptive: {line}")
        if multitouch_match is not None:
            raise NotImplementedError(f"Multitouch event found in line, which is unexpected and currently not supported: {line}")


        # assert exactly one matches in each line, otherwise the log format is unexpected and we raise an error.
        total_legal_matches = sum(match is not None for match in [id_match, x_match, y_match, button_match, key_match]) + sum(exist is True for exist in [syn_report_in_line, syn_config_in_line])
        if total_legal_matches != 1:
            raise ValueError(f"Unexpected log format: {total_legal_matches} matches in line: {line}")

        if time_match is not None:
            t_us_tmp = int(time_match.group(1)) * int(1e6) + int(time_match.group(2))
            
            if not last_is_syn_report and (t_us != t_us_tmp):
                raise ValueError(f"Timestamp discrepancy detected. Previous timestamp: {t_us}, current timestamp: {t_us_tmp}. This may indicate an unexpected log format or missing SYN_REPORT lines.")
            t_us = t_us_tmp

        if id_match is not None:
            fleeting_id = id_match.group(1)
            
            # exactly one of fleeting_id and inherit_id should be TERMINATION_ID, otherwise the log format is unexpected and we raise an error.
            if (fleeting_id == TERMINATION_ID) == (inherited_id == TERMINATION_ID):
                raise ValueError(f"Unexpected log format: invalid id inheritance. fleeting_id: {fleeting_id}, inherited_id: {inherited_id}, line: {line}")
            inherited_id = fleeting_id

            last_is_syn_report = False

        elif x_match is not None:
            x = hex_to_dec(x_match.group(1))
            
            last_is_syn_report = False

        elif y_match is not None:
            y = hex_to_dec(y_match.group(1))
            
            last_is_syn_report = False

        elif button_match is not None:

            # only some error checking
            button_stat = button_match.group(1)
            if button_stat == "DOWN":
                if (inherited_id == fleeting_id) and (inherited_id != TERMINATION_ID):
                    pass
                else:
                    raise ValueError(f"Unexpected log format: button DOWN event without proper id inheritance. inherited_id: {inherited_id}, fleeting_id: {fleeting_id}, line: {line}")
            elif button_stat == "UP":
                if (inherited_id == fleeting_id) and (inherited_id == TERMINATION_ID):
                    pass
                else:
                    raise ValueError(f"Unexpected log format: button UP event without proper id inheritance. inherited_id: {inherited_id}, fleeting_id: {fleeting_id}, line: {line}")
            else:
                raise ValueError(f"Unexpected log format: button event with unknown status: {button_stat}, line: {line}")

            last_is_syn_report = False
        
        elif key_match is not None:
            last_is_key_event = True

            last_is_syn_report = False

        elif syn_report_in_line:
            # SYN_REPORT indicates the current sample is completed
            last_is_syn_report = True
            fleeting_id = ""

            if last_is_key_event:
                last_is_key_event = False
                continue # skip SYN_REPORT lines that come after key events, which are unexpected but not disruptive.
            
            # if any of t_us, x, y is None, it means the current sample is incomplete, which may be caused by unexpected log format or missing lines. We raise an error in this case.
            if (t_us is None):
                raise ValueError(f"Unexpected log format: timestamp is missing for this SYN_REPORT block: {line}")
            
            if (x is None) or (y is None):
                exist_incomplete_sample = True
                # This means that the resulting trace may have none as result. 
                # raise ValueError(f"Unexpected log format: incomplete sample before SYN_REPORT: {line}")

            current_gesture.append(FingerEvent(timestamp_us=t_us, x=x, y=y))
            # x, y = None, None shouldn't reset the saved values

            if not isinstance(inherited_id, str):
                raise ValueError(f"Unexpected log format: inherited_id is not set before SYN_REPORT: {line}")

            if inherited_id == TERMINATION_ID:
                # Finger lifted — current gesture ends
                # currently current_gesture must have length > 0.
                result.append(current_gesture)
                current_gesture = []    
            
    if (not last_is_syn_report) or (not inherited_id == TERMINATION_ID) or (len(current_gesture) > 0):
        raise ValueError(f"Unexpected log format: log ended without proper termination. last_is_syn_report: {last_is_syn_report}, inherited_id: {inherited_id}, len(current_gesture): {len(current_gesture)}", result, current_gesture)

    if exist_incomplete_sample:
        raise ValueError(f"Unexpected log format: at least one incomplete sample detected. This may indicate missing lines or some other log format issue. Please check the logs for details. The generated trace may contain incomplete samples with None values.", result)

    return result


def null_object_warning() -> None:
    print("Null item detected. These guards are useful after all.")

def gesture_generator_from_files(df_idx: pd.DataFrame, logs_dir: Path) -> List[Tuple[str, str, List[SingularActionType]]]:
    """Read df_idx(the catalog of logs), iterate logs, extract unmodified swipes, and return a List of Lists.
    The resulting DataFrame has a 'type' column followed by feature columns.
    resuiting list have: (participant, session_time_stamp, gestures)
    """
    result: List[Tuple[str, str, List[SingularActionType]]] = []
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
        result.append((label, log_num, single_trace_generator(file_reader_yield(str(log_file)))))
    return result

def filtered_gestures_generation(
        gesture_iterator: List[SingularActionType], 
        filtering_and_modification_function: Callable[[SingularActionType], Optional[SingularActionType]]
    ) -> List[SingularActionType]:
    """Apply filtering_and_modification_function to each swipe list and return a filtered list. If None, skip."""
    result: List[SingularActionType] = []
    for gesture in gesture_iterator:
        filtered_gesture = filtering_and_modification_function(gesture)
        if filtered_gesture is not None:
            result.append(filtered_gesture)
    return result


def filtered_gesture_generator_from_files(
    df_idx: pd.DataFrame,
    logs_dir: Path,
    filtering_and_modification_function: Callable[[SingularActionType], Optional[SingularActionType]],
) -> List[Tuple[str, str, List[SingularActionType]]]:
    """
    for df_idx, generate filtered gestures from files in logs_dir using filtering_and_modification_function.
    
    :param df_idx: 
    :type df_idx: pd.DataFrame
    :param logs_dir: 
    :type logs_dir: Path
    :param filtering_and_modification_function: Description
    :type filtering_and_modification_function: Callable[[SingularActionType], Optional[SingularActionType]]
    :return: A list of tuples containing participant tag, session timestamp, and filtered gestures; one tuple per file.
    :rtype: List[Tuple[str, str, List[SingularActionType]]]
    """

    result: List[Tuple[str, str, List[SingularActionType]]] = []
    for participant, session_timestamp, swipe_iterator in gesture_generator_from_files(df_idx, logs_dir):
        # skip the nullified swipes
        filtered_swipe_iterator = filtered_gestures_generation(swipe_iterator, filtering_and_modification_function)
        result.append((participant, session_timestamp, filtered_swipe_iterator))
    return result

def filtered_gesture_generator_from_files_no_timestamp(
    df_idx: pd.DataFrame,
    logs_dir: Path,
    filtering_and_modification_function: Callable[[SingularActionType], Optional[SingularActionType]],
) -> List[Tuple[str, List[SingularActionType]]]:
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
    for participant_tag, session_timestamp, gesture_iterator in gesture_generator_from_files(df_idx, logs_dir):
        for gesture in gesture_iterator:
            if gesture is None:
                raise ValueError(f"Null swipe detected in label {participant_tag}. These guards are useful after all.")

            # check that every gesture has no None attributes
            if not is_integral(gesture):
                raise ValueError(f"Non-integral gesture detected in label {participant_tag}. These guards are useful after all.")

def stack_iterator_with_str(label: str, items: List[SingularActionType]) -> List[Tuple[str, SingularActionType]]:
    """Stack a label alongside each item in a list."""
    return [(label, item) for item in items]

def ranged_batched_modified_generator_with_session_timestamp(
        formated_data_timestamps: pd.DataFrame, 
        participants: List[str], 
        index_range: Optional[List[int]] = None, 
        filtering_and_modification_function: Optional[
            Callable[[List[SingularActionType]], List[SingularActionType]]
        ] = None, 
        humanity_disturbance: Optional[
            Callable[[SingularActionType], SingularActionType]
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
            
            if pd.isnull(timestamp):
                # print(f"Warning: Null timestamp found for participant {participant_name} at index {index}. Skipping this entry.")
                continue

            stripped_timestamp = str(timestamp).strip()
            if (stripped_timestamp == ""):
                continue
            #print(stripped_timestamp)
            #continue

            # assert that the timestamp follow the yyyymmdd_hhmmss format, otherwise the log format is unexpected and we raise an error.
            if not re.match(r'^\d{8}_\d{6}$', stripped_timestamp):
                print(f"Unexpected timestamp format for participant {participant_name} at index {index}: '{timestamp}'")
                continue
            
            try:
                file_path = file_finder(Path("logs/"), "gesture_recording_" + stripped_timestamp + ".log")
            except FileNotFoundError as e:
                print(participant_name, e)
                raise e
            line_generator = file_reader_yield(str(file_path))

            try:
                single_trace_generated = single_trace_generator(line_generator)
                
            except ValueError as e:
                if "incomplete sample" in e.args[0]:
                    print(f"Warning: Incomplete sample detected in file {file_path}. This may indicate an unexpected log format or missing lines. Keeping the traces with complete properties.")
                    thingy_to_filter = e.args[1]
                    accumulator: List[SingularActionType] = []
                    for item in reversed(thingy_to_filter):
                        if not is_integral(item):
                            break
                        accumulator.append(item)
                    single_trace_generated = list(reversed(accumulator))
                    print(f"Kept {len(single_trace_generated)} complete samples from all {len(thingy_to_filter)} samples in the generated trace.")

                elif "without proper termination" in e.args[0]:
                    print(f"Warning: Error processing file {file_path}: {e.args[0]}. Discarding the last incomplete trace.")
                    single_trace_generated = e.args[1] # compact stacked traces are contact.

                elif "invalid id inheritance" in e.args[0]:
                    print(f"Error processing file {file_path}: {e}. Skipping this file.")
                    continue

                else:
                    print(f"Error processing file {file_path}: {e}")
                    raise e

            except NotImplementedError as e:
                if "Multitouch event found in line" in e.args[0]:
                    print(f"Warning: Multitouch event found in file {file_path}, which is unexpected and currently not supported. Skipping this file.")
                    continue
                else:
                    print(f"Error processing file {file_path}: {e}")
                    raise e

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
            Callable[[List[SingularActionType]], List[SingularActionType]]
        ] = None, 
        humanity_disturbance: Optional[
            Callable[[SingularActionType], SingularActionType]
        ] = None
        ) -> List[SingularActionType]:
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
    result: List[SingularActionType] = []
    for timestamp, swipes in result_with_timestamps:
        result += swipes
    return result

def ranged_modified_generator_without_session_timestamp(
        formated_data_timestamps: pd.DataFrame, 
        participants: List[str], 
        index_range: Optional[List[int]] = None, 
        filtering_and_modification_function: Optional[
            Callable[[SingularActionType], Optional[SingularActionType]]
        ] = None, 
        humanity_disturbance: Optional[
            Callable[[SingularActionType], SingularActionType]
        ] = None
        ) -> List[SingularActionType]:
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
            Callable[[SingularActionType], Optional[SingularActionType]]
        ] = None, 
        humanity_disturbance: Optional[
            Callable[[SingularActionType], SingularActionType]
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

