
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from analysis.lib.motionevent_classes import SingularActionType
from analysis.lib.keyevent_classes import MotionKeyEvent, IMEEvent, IMESpecialKeyType

from analysis.lib.gesture_log_reader_utils import file_finder, rectify_timestamp_idx, session_noexcept_handling, limited_file_reader_yield, file_reader_yield, gesture_log_schema
from analysis.lib.sensor_log_reader_utils import parse_as_df, sensor_recording_name_schema
from analysis.lib.key_reader_utils import keys_generator_from_motionevent, IME_generator_from_IME_recording, IME_event_name_schema
import numpy as np
from pathlib import Path    
import re
import subprocess
# import cv2
# set up logging
import logging
import base64
from argparse import ArgumentParser
from agent_tools.fake_adb.adb_wrapper import MotionGenerator
from concurrent.futures import ThreadPoolExecutor

# get log level from environment variable, default to ERROR
import os
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "ERROR").upper(), logging.ERROR),
    format="%(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)


from analysis.lib.feature_library import is_tap, swipe_judger, startT_us

LOGS_FOLDER = Path("logs")

# CV2_VIDEO_BACKEND = cv2.CAP_GSTREAMER

def extract_starting_time_ns_of_sensor(timestamp: str) -> int:
    """
        extreme case 20251214_180650 where there is only Light and Proximity, with Proximity's timestamp smaller.
        However, Light and Proximity timestamps seem to be "early accumulations" rather than real-time, so don't use them.
    """
    # get the sensor event path
    sensor_event_path = file_finder(LOGS_FOLDER, f"{sensor_recording_name_schema(timestamp)}")

    _, sensor_df = parse_as_df(limited_file_reader_yield(sensor_event_path, head_lines=200))
    # if (sensor_df["Proximity"]['t'].iloc[0] >= sensor_df["Accelerometer"]['t'].iloc[0]) or (sensor_df["Proximity"]['t'].iloc[0] >= sensor_df["Gyroscope"]['t'].iloc[0]):
    #     raise ValueError(f"Unexpected sensor event file format: the first timestamp in Proximity sensor events is smaller than that in Accelerometer or Gyroscope events, which is not expected since Proximity sensor events should be triggered by the tap action at the very beginning of the session. Please check the sensor event file {sensor_event_path} for details.")
    first_ns = sensor_df["Accelerometer"]['t'].iloc[0]

    logger.debug(f"Extracted first timestamp from sensor event file {sensor_event_path}: {first_ns} ns, which is {first_ns // 1000} us")
    return first_ns

def _ffmpeg_extract_frame_to_file(video_path: Path, time_in_s: float, output_path: Path) -> bool:
    """Use ffmpeg -ss (input seeking) to extract a single frame directly to a jpg file."""
    result = subprocess.run(
        ["ffmpeg", "-ss", str(time_in_s), "-i", str(video_path),
         "-frames:v", "1", "-q:v", "2", "-y",
         "-v", "error", str(output_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not output_path.exists():
        logger.debug(f"ffmpeg failed for time {time_in_s}s: {result.stderr.strip()}")
        return False
    return True


def extract_frames_with_designated_time_in_s(video_path: Path, designated_time_in_s: List[float], output_folder: Path, start_idx: int = 0) -> Tuple[int, float]:
    """
        Extract frames at designated times using ffmpeg, writing jpgs directly.
        Returns (number of frames written, biggest available time in seconds).
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    error_frames: List[Tuple[float, int]] = []
    written = 0
    biggest_available_time_in_s = 0.0

    for i, time_in_s in enumerate(designated_time_in_s):
        out_path = output_folder / f"{start_idx + written}.jpg"
        if _ffmpeg_extract_frame_to_file(video_path, time_in_s, out_path):
            biggest_available_time_in_s = max(biggest_available_time_in_s, time_in_s)
            written += 1
        else:
            error_frames.append((time_in_s, i))

    logger.info(f"Extracted {written} frames from video at designated times.")
    if error_frames:
        logger.warning(f"Failed to extract {len(error_frames)} frames at designated times: {error_frames}")
    return written, biggest_available_time_in_s

def _ffmpeg_extract_frame_sseof(video_path: Path, offset_s: float, output_path: Path) -> bool:
    """Try to extract a frame at -offset_s from end of video. Returns True if a non-empty jpg was produced."""
    result = subprocess.run(
        ["ffmpeg", "-sseof", f"-{offset_s}", "-i", str(video_path),
         "-frames:v", "1", "-q:v", "2", "-y", "-update", "1",
         "-v", "error", str(output_path)],
        capture_output=True, text=True
    )
    return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0


def get_last_frame_of_video(video_path: Path, output_path: Path) -> bool:
    """
        Extract the last frame of a video using ffmpeg -sseof.
        Returns True if successful.
    """
    if _ffmpeg_extract_frame_sseof(video_path, 0.1, output_path):
        logger.info("Extracted last frame from video.")
        return True
    raise IOError(f"Cannot extract last frame from video {video_path}")


def get_last_readable_frame_of_video(video_path: Path, output_path: Path) -> bool:
    """
        Find the latest readable frame using ffmpeg, with exponential backoff
        from the end followed by binary search to narrow down the boundary.
        Writes the result directly to output_path as jpg.
        Returns True if successful.
    """
    # Phase 0: try the very end first
    if _ffmpeg_extract_frame_sseof(video_path, 0.1, output_path):
        logger.info("Extracted last readable frame from video (offset 0.1s).")
        return True

    # Phase 1: exponential backoff — find a readable offset
    step = 0.5
    high_unreadable = 0.1  # smallest offset known to fail
    low_readable = -1.0    # will be set when we find a readable offset
    max_offset = 60.0      # don't search beyond 60s from end

    while step <= max_offset:
        if _ffmpeg_extract_frame_sseof(video_path, step, output_path):
            low_readable = step
            break
        high_unreadable = step
        step *= 2

    if low_readable < 0:
        raise IOError(f"Cannot read any frame from video {video_path} (searched up to {max_offset}s from end)")

    # Phase 2: binary search between [high_unreadable, low_readable] to find
    # the smallest offset (closest to end) that still produces a frame.
    left = high_unreadable
    right = low_readable
    best_offset = low_readable

    while right - left > 0.05:  # ~50ms precision
        mid = (left + right) / 2
        if _ffmpeg_extract_frame_sseof(video_path, mid, output_path):
            best_offset = mid
            right = mid
        else:
            left = mid

    # Final extraction at best offset (may already be on disk from last successful probe)
    if best_offset != right or not (output_path.exists() and output_path.stat().st_size > 0):
        _ffmpeg_extract_frame_sseof(video_path, best_offset, output_path)

    logger.info(f"Extracted latest readable frame from video (sseof=-{best_offset:.2f}s).")
    return True


"""
def save_frames_to_folder(frames: List[cv2.typing.MatLike], output_folder: Path):
    output_folder.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        output_path = output_folder / f"{idx}.jpg"
        cv2.imwrite(str(output_path), frame)
    logger.info(f"Saved {len(frames)} frames to folder {output_folder}")
"""



def extract_nonfake_actions(gesture_events: List[SingularActionType]) -> List[SingularActionType]:
    results: List[SingularActionType] = []
    for action in gesture_events:
        if is_tap(action) or swipe_judger(action):
            if not MotionGenerator.is_custom_fake_action_3(action):
                results.append(action)
    return results
            
def extract_us_timestamps_from_gesture(gesture_events: List[SingularActionType]) -> List[int]:
    return [startT_us(action) for action in gesture_events]

def extract_key_events_seconds(time_stamp: str) -> List[float]:
    adb_getevent_generator = file_reader_yield(file_finder(LOGS_FOLDER, f"{gesture_log_schema(time_stamp)}"))
    keys = keys_generator_from_motionevent(adb_getevent_generator)
    
    key_time = [event.down_us / 1e6 for event in keys]

    return key_time

def extract_ime_events_seconds(time_stamp: str) -> List[float]:
    
    ime_recording_path = file_reader_yield(file_finder(LOGS_FOLDER, f"{IME_event_name_schema(time_stamp)}"))
    ime_events = IME_generator_from_IME_recording(ime_recording_path)

    # selection logic
    ime_time = [event.flush_time_s for event in ime_events if isinstance(event.flushed_output, IMESpecialKeyType)]
    return ime_time

SUCCESS_FILE = "success.txt"
def exist_success(folder_path: Path) -> bool:
    # check if success.txt exists in the folder
    success_file = folder_path / SUCCESS_FILE
    return success_file.exists()

def write_success(folder_path: Path):
    success_file = folder_path / SUCCESS_FILE
    success_file.touch()

def extract_step_frames(time_stamp: str, created_output_folder: Path):

    # get the video path
    video_path = file_finder(LOGS_FOLDER, f"screen_recording_{time_stamp}.mp4")

    logger.debug(f"Processing timestamp {time_stamp} with video path {video_path}")

    first_ns = extract_starting_time_ns_of_sensor(time_stamp)
    first_us = first_ns / 1000


    # get the gesture event path
    gesture_events = session_noexcept_handling(time_stamp)
    if gesture_events is None:
        logger.error(f"No gesture events extracted for timestamp {time_stamp}. Skipping frame extraction.")
        gesture_events = []
    else:
        gesture_events = gesture_events[1]
    gesture_start_time_uss = extract_us_timestamps_from_gesture(extract_nonfake_actions(gesture_events))


    debug_path = file_finder(LOGS_FOLDER, gesture_log_schema(time_stamp))
    logger.debug(f"Extracted gesture start timestamps in microseconds: {debug_path}: {gesture_start_time_uss}")

    # calculate the designated time in seconds for frame extraction, by aligning the gesture events with the sensor event first timestamp, and then converting to seconds
    gesture_start_time_s = [us / 1e6 for us in gesture_start_time_uss]

    supplementary_key_time_s = extract_key_events_seconds(time_stamp)

    try:
        supplementary_ime_time_s = extract_ime_events_seconds(time_stamp)
    except FileNotFoundError as e:
        logger.warning(f"Key/IME event file not found for timestamp {time_stamp} during key/IME event extraction: {e}")
        supplementary_ime_time_s = []

    # merge gesture_start_time_s with supplementary_key_ime_time_s, and sort them, to get the final designated_time_in_s for frame extraction.
    designated_time_in_s = sorted(gesture_start_time_s + supplementary_key_time_s + supplementary_ime_time_s)
    # reduce by the offset
    designated_time_in_s_offset_corrected = [t - first_us / 1e6 for t in designated_time_in_s]

    logger.debug(f"Calculated designated times in seconds for frame extraction: {designated_time_in_s_offset_corrected}")

    # extract the frames from the video at the designated times
    num_successful_frames, biggest_available_time_in_s = extract_frames_with_designated_time_in_s(video_path, designated_time_in_s_offset_corrected, created_output_folder, start_idx=0)

    try:
        last_frame_path = created_output_folder / f"{num_successful_frames}.jpg"
        get_last_readable_frame_of_video(video_path, last_frame_path)
    except IOError as e:
        logger.warning(f"Failed to extract any later readable frame from video {video_path}: {e}")


    write_success(created_output_folder)

@dataclass
class SPAMetadataClass:
    task_identifier: str
    task_requirement: str


def process_timestamp_task(agent_name: str, task_idx: int, timestamp: str, task_requirement: str, skip_existing_folder: bool) -> Optional[SPAMetadataClass]:
    output_folder_to_create = OUTPUT_PATH / timestamp
    result_metadata = SPAMetadataClass(
        task_identifier=timestamp,
        task_requirement=task_requirement,
    )
    
    if skip_existing_folder:
        if output_folder_to_create.exists():
            if exist_success(output_folder_to_create):
                logger.info(f"Output folder {output_folder_to_create} already exists with successful extraction. Skipping extraction for timestamp {timestamp}.")
                return result_metadata
    else:
        output_folder_to_create.mkdir(parents=False, exist_ok=False)

    try:
        extract_step_frames(timestamp, output_folder_to_create)
    except FileNotFoundError as e:
        logger.warning(f"File not found for agent {agent_name} task {task_idx} with timestamp {timestamp}: {e}")
        # remove the folder if it was created, to avoid leaving empty folders for failed extractions
        return None
    except IOError as e:
        logger.warning(f"IO error for agent {agent_name} task {task_idx} with timestamp {timestamp}: {e}")
        return None
    except IndexError as e:
        logger.warning(f"Index error (possibly due to truncated sensor events) for agent {agent_name} task {task_idx} with timestamp {timestamp}: {e}")
        return None

    return result_metadata

def construct_csv(metadata_list: List[SPAMetadataClass]) -> pd.DataFrame:
    "task_identifier,task_app,task_language,task_description,task_difficulty,golden_steps,key_component_final,adb_app,adb_home_page,is_cross_app"
    # create this pandas dataframe
    # results_csv = pd.DataFrame(columns=["task_identifier", "task_app", "task_language", "task_description", "task_difficulty", "golden_steps", "key_component_final", "adb_app", "adb_home_page", "is_cross_app"])
    # <mytask_0001>,unknown,CHN,<requirements>,1,0,"[]",,,N
    # now, construct the dataframe.
    # first construct a List[Dict[str, Any]] where each dict corresponds to a row in the dataframe, and then convert it to a dataframe.
    rows = []
    for metadata in metadata_list:
        row = {
            "task_identifier": metadata.task_identifier,
            "task_app": "unknown",
            "task_language": "CHN",
            "task_description": metadata.task_requirement,
            "task_difficulty": 1,
            "golden_steps": 0,
            "key_component_final": "[]",
            "adb_app": "",
            "adb_home_page": "",
            "is_cross_app": "N",
        }
        rows.append(row)
    results_csv = pd.DataFrame(rows, columns=["task_identifier", "task_app", "task_language", "task_description", "task_difficulty", "golden_steps", "key_component_final", "adb_app", "adb_home_page", "is_cross_app"])
    return results_csv


if __name__ == "__main__":

    parser = ArgumentParser(description="Extract frames from videos based on gesture events and sensor events, and save them in a structured way for SPA evaluation.")
    # 1. only extract first n sessions
    parser.add_argument("--formatted_data_excel", type=str, required=True, help="The path to the formatted data Excel file that contains the timestamps and task requirements. By default, it is set to 'Formated_Data_Renamed.xlsx'.")
    parser.add_argument("--output_folder", type=str, required=True, help="The output folder to save the extracted frames and metadata CSV. The script will create a subfolder for each session (timestamp)")
    parser.add_argument("--only_extract_n_sessions", type=int, default=None, help="Only extract the first n sessions for testing and debugging purposes.")
    parser.add_argument("--work_on_existent_folder", action="store_true", default=False, help="Whether to work on an already existent output folder. By default, the script will create a new output folder and throw an error if it already exists, to avoid accidental overwriting of existing data.")
    parser.add_argument("--only_test_single", type=str, default=None, help="Only test the extraction on a single specified timestamp (session), for testing and debugging purposes.")
    parser.add_argument("--only_test_sensor_and_ime", default=False, action="store_true", help="Only test the extraction of sensor and IME events without processing videos, for testing and debugging purposes.")
    args = parser.parse_args()

    n = int(args.only_extract_n_sessions) if args.only_extract_n_sessions is not None else float('inf')
    work_on_existent_folder: bool = args.work_on_existent_folder

    OUTPUT_PATH = Path(args.output_folder)
    OUTPUT_PATH.mkdir(parents=False, exist_ok=work_on_existent_folder)

    if args.only_test_single is not None:
        process_timestamp_task("test_agent", 0, args.only_test_single, "test_requirement", work_on_existent_folder)
        exit(0)

    formated_data_timestamps = pd.read_excel(args.formatted_data_excel, header=0, index_col=None, dtype=str)
    TASK_COUNT = 116

    columns_list = list(formated_data_timestamps.columns)

    from operator import itemgetter

    task_list: str = columns_list[1]


    humans: List[str] = list(itemgetter(2, 3, 4, 33, 34, 35, 36)(columns_list))
    humanized_agents: List[str] = list(itemgetter(5, 7, 9, 20, 39)(columns_list))
    non_humanized_agents: List[str] = list(itemgetter(12, 14, 16, 19, 37)(columns_list))
    rot_humanized_agents: List[str] = list(itemgetter(23, 25)(columns_list))
    fake_rot_tap_humanized_agents: List[str] = list(itemgetter(28, 30, 41)(columns_list))


    agents = humanized_agents + non_humanized_agents + rot_humanized_agents + fake_rot_tap_humanized_agents

    metadata_list_maybe_none: List[Optional[SPAMetadataClass]] = []

    timestamp_list = rectify_timestamp_idx(
        formated_data_timestamps,
        participants=agents
    )

    pending_tasks: List[Tuple[str, int, str, str, bool]] = []
    for agent_name, task_tuples in timestamp_list.items():
        for task_idx, timestamp in task_tuples:
            task_requirement = str(formated_data_timestamps.loc[task_idx, task_list])
            pending_tasks.append((agent_name, task_idx, timestamp, task_requirement, work_on_existent_folder))

    logger.info("Totally pending tasks to process: %d", len(pending_tasks))



    if args.only_test_sensor_and_ime:
        for agent_name, task_idx, timestamp, task_requirement, _ in tqdm(pending_tasks):
            logger.info(f"Testing sensor and IME extraction for agent {agent_name} task {task_idx} with timestamp {timestamp}")
            try:
                important_seconds = extract_key_events_seconds(timestamp)
                important_seconds += extract_ime_events_seconds(timestamp)
                logger.info(f"Extracted important seconds for timestamp {timestamp}")
            except FileNotFoundError as e:
                logger.warning(f"File not found during sensor and IME extraction test for timestamp {timestamp}: {e}")
            except Exception as e:
                logger.error(f"Error during sensor and IME extraction test for timestamp {timestamp}: {type(e).__name__}: {e}")
                raise e
        
        for agent_name, task_idx, timestamp, task_requirement, _ in tqdm(pending_tasks):
            logger.info(f"Testing sensor event starting time extraction for agent {agent_name} task {task_idx} with timestamp {timestamp}")
            try:
                starting_time_ns = extract_starting_time_ns_of_sensor(timestamp)
                logger.info(f"Extracted starting time of sensor events: {starting_time_ns} ns for timestamp {timestamp}")
            except FileNotFoundError as e:
                logger.warning(f"File not found during sensor starting time extraction test for timestamp {timestamp}: {e}")
            except Exception as e:
                logger.error(f"Error during sensor starting time extraction test for timestamp {timestamp}: {type(e).__name__}: {e}")

        exit(0)


    with ThreadPoolExecutor() as executor:
        metadata_list_maybe_none.extend(executor.map(lambda task: process_timestamp_task(*task), pending_tasks))
    # execute sequentially for first n sessions
    # for agent_name, task_idx, timestamp, task_requirement in pending_tasks[:n]:
    #     metadata = process_timestamp_task(agent_name, task_idx, timestamp, task_requirement)
    #     if metadata is not None:
    #         metadata_list.append(metadata)

        #     if len(metadata_list) >= n:
        #         logger.info(f"Reached the limit of {n} sessions to extract. Stopping further extraction.")
        #         break
        # if len(metadata_list) >= n:
        #     break
            
    metadata_list: List[SPAMetadataClass] = [metadata for metadata in metadata_list_maybe_none if metadata is not None]

    results_csv = construct_csv(metadata_list)
    csv_name = "results.csv"

    results_csv.to_csv(OUTPUT_PATH / csv_name, index=False)
    logger.info(f"Saved metadata CSV to {OUTPUT_PATH / csv_name}")

    





