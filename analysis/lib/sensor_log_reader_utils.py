from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict
from analysis.lib.gesture_log_reader_utils import file_finder, file_reader_yield
from analysis.plotting.draw_sensor_event import parse_as_df
import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    tqdm = lambda x: x  # type: ignore
    HAS_TQDM = False

class SensorSessionType(TypedDict):
    session_timestamp: str
    time_offset_ns: int
    sensor_data_for_each_sensor_type: Dict[str, pd.DataFrame]

def sensor_recording_name_schema(timestamp: str) -> str:
    return f"sensor_recording_{timestamp}.txt"

def ranged_batched_modified_sensor_generator_with_session_timestamp(
        formated_data_timestamps_rectified: Dict[str, List[Tuple[int, str]]], 
        participants: List[str], 
        index_range: Optional[List[int]] = None, 
        filtering_and_modification_function = None,
        humanity_disturbance = None,
        use_tqdm_for_each_participant: bool = False
        ) -> List[SensorSessionType]:
    """
    formated_data_timestamps: Dict[str(participant_name), List[Tuple[int(task_id), str(timestamp)]]]
    participants: List of participant names to include. Can possibly not appear in formated_data_timestamps.
    index_range: Optional range of task indices to include
    """
    if filtering_and_modification_function is not None or humanity_disturbance is not None:
        raise NotImplementedError("filtering_and_modification_function and humanity_disturbance are not supported in this function.")
    if use_tqdm_for_each_participant and not HAS_TQDM:
        raise ImportError("tqdm is not installed, cannot use tqdm for progress bar.")

    result: List[SensorSessionType] = []
    for participant in participants:
        if participant not in formated_data_timestamps_rectified:
            continue
        task_timestamps = formated_data_timestamps_rectified[participant]
        if index_range is not None:
            filtered_task_timestamps = [task_timestamp for task_timestamp in task_timestamps if task_timestamp[0] in index_range]
        else:
            filtered_task_timestamps = task_timestamps
        for task_id, timestamp in (tqdm if use_tqdm_for_each_participant else lambda x: x)(filtered_task_timestamps):
            try:
                sensor_file_path = file_finder(search_scope=Path("logs/"), file_name=sensor_recording_name_schema(timestamp))
            except FileNotFoundError as fe:
                print(f"Warning: {fe}. timestamp: {timestamp} is not found.")
                continue
            
            if sensor_file_path is None:
                raise FileNotFoundError(f"Sensor log file not found for participant {participant}, task {task_id}, timestamp {timestamp}")
            lines = file_reader_yield(sensor_file_path)
            time_offset_ns, all_sensor_data = parse_as_df(lines)
            
            result.append(SensorSessionType(
                session_timestamp = timestamp,
                time_offset_ns = time_offset_ns,
                sensor_data_for_each_sensor_type = all_sensor_data
            ))
    return result