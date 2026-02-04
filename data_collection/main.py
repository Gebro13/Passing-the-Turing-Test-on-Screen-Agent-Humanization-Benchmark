from pathlib import Path
import sys

this_path = Path(__file__).resolve()
PROJ_FOLDER = this_path.parent.parent

# assert the running folder is PROJ_FOLDER, otherwise throw error
if Path.cwd() != PROJ_FOLDER:
    raise NotImplementedError("Please run this script from the project root folder.")
    print(f"Changing working directory from {Path.cwd()} to project folder {PROJ_FOLDER}")
    os.chdir(PROJ_FOLDER)

sys.path.insert(0, str(PROJ_FOLDER))

import data_collection.controller as controller
import argparse
import os
import time
import data_collection.automations as automations

from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gesture and sensor data collection script.")
    parser.add_argument("--only_adb_getevent", dest="only_adb_getevent", action="store_true", default=False,
                        help="If set, only runs the adb getevent command without starting gesture recording.")
    parser.add_argument("--automatic_exit_app_and_reset", dest="automatic_exit_app_and_reset", action="store_true", default=False,
                        help="If set, exits the app to the home screen, clear it and return to app screen after recording.")
    parser.add_argument("--automatic_switch_app", dest="automatic_switch_app", action="store_true", default=False,
                        help="If set, automatically returns to already running app from sensor recorder")
    # add argument: 1. current user
    parser.add_argument("--user", type=str, default="default_user",
                        help="Specify the current user for data organization.")
    parser.add_argument("--task_provide_file", type=str, default=None,
                        help="CSV file that provides the list of tasks to be performed.")
    args = parser.parse_args()

    # if task_provide_file is given, read the file, find the user name column, find the first empty block, and print the app_name and app_description column of the row
    can_log = (args.task_provide_file is not None) and (args.user is not None)
    if can_log:
        result = automations.load_not_done_task_pair(attendee_name=args.user, task_file_path=args.task_provide_file)
        if result is None:
            print(f"All tasks for user {args.user} are already completed.")
            exit(0)
        first_empty_row_idx, app_name, task_description = result

    basepath = Path("logs")
    if not basepath.exists():
        basepath.mkdir(parents=True)
    absolute_base_path = basepath.resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(basepath / f"gesture_recording_{timestamp}.log")
    output_video = str(basepath / f"screen_recording_{timestamp}.mp4")
    absolute_output_IME_path = str(absolute_base_path / f"IME_event_{timestamp}.txt")

    output_sensor_path = str(basepath / f"sensor_recording_{timestamp}.txt")
    sensor_apk_name = "com.example.motionlogger"
    input_sensor_path = f"/sdcard/Android/data/{sensor_apk_name}/Files/motion_log.txt"


    proc = controller.start_gesture_recording("adb", output_path=output_path)

    if not args.only_adb_getevent:
        video_proc = controller.start_recording("adb", temp_file_name="screenrecord2")
        controller.start_sensor_recording("adb", sensor_apk_name=sensor_apk_name, input_sensor_path=input_sensor_path)
        controller.setup_IMEevent_capturer(absolute_IME_path=absolute_output_IME_path)
        if args.automatic_switch_app:
            automations.switch_app_from_sensorevent_to_appscreen()

    try:
        input(f"Press Enter to stop recording...   {timestamp}")
    except KeyboardInterrupt:
        pass
    print("Stopping main gesture recording...")
    controller.stop_gesture_recording(proc)

    if not args.only_adb_getevent:
        controller.stop_sensor_recording(sensor_apk_name=sensor_apk_name, input_sensor_path=input_sensor_path, output_sensor_path=output_sensor_path)
        controller.end_recording("adb", video_proc, output_video, temp_file_name="screenrecord2")
        controller.nullify_IMEevent_capturer()

    if args.automatic_exit_app_and_reset:
        automations.prepare_apps_screen()

    if can_log:
        automations.write_timestamp_to_idx(
            task_file_path=args.task_provide_file,
            attendee_name=args.user,
            empty_row_idx=first_empty_row_idx,
            timestamp=timestamp
        )