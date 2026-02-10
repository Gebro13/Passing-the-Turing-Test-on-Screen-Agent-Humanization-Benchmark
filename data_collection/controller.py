import os
from pathlib import Path
import time
import subprocess
from PIL import Image
from time import sleep
import threading
import random
import psutil
import numpy.typing as npt
import numpy as np

from typing import List

def start_recording(adb_path, temp_file_name: str = "screenrecord"):
    print(f"Remove existing {temp_file_name}.mp4")
    command = adb_path + f" shell rm /sdcard/{temp_file_name}.mp4"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    print("Start!")
    # Use subprocess.Popen to allow terminating the recording process later
    command = adb_path + f" shell /data/local/tmp/screenrecord_modified /sdcard/{temp_file_name}.mp4"
    process = subprocess.Popen(command, shell=True)
    return process

def end_recording(adb_path, video_proc: subprocess.Popen, output_recording_path, temp_file_name: str = "screenrecord"):
    print("Stopping recording...")
    # Send SIGINT to stop the screenrecord process gracefully
    stop_command = adb_path + " shell pkill -l SIGINT screenrecord" # strangely, changing the name doesn't change the process name
    subprocess.run(stop_command, capture_output=True, text=True, shell=True)
    sleep(1)  # Allow some time to ensure the recording is stopped
    video_proc.terminate()
    video_proc.wait()
    sleep(1)  # Allow some time to ensure the recording is stopped

    print("Pulling recorded file from device...")
    pull_command = f"{adb_path} pull /sdcard/{temp_file_name}.mp4 {output_recording_path}"
    subprocess.run(pull_command, capture_output=True, text=True, shell=True)
    print(f"Recording saved to {output_recording_path}")


# get a string from adb shell cat /proc/uptime
def get_current_phone_timestamp() -> str:
    """
        have maximally 40ms latency. This is not real time, after all. JD can obtain the "real time" on its own.
    """
    command = "adb shell cat /proc/uptime"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error: Failed to get current phone timestamp. {result.stderr}")
    return result.stdout.strip()

COLLECTION_FOLDER_ABSOLUTE: Path = Path(__file__).resolve().parent
PROJ_FOLDER_ABSOLUTE: Path = COLLECTION_FOLDER_ABSOLUTE.parent
FAKE_ADB_PATH_ABSOLUTE: Path = PROJ_FOLDER_ABSOLUTE / "agent_tools" / "fake_adb"
BIN_IME_PATH: Path = FAKE_ADB_PATH_ABSOLUTE / "ime_bin_event_capturer.txt"

IME_EVENT_PATH_STORAGE =  PROJ_FOLDER_ABSOLUTE / "IME_EVENT_PATH.txt"

def setup_IMEevent_capturer(absolute_IME_path: str) -> None:
    """
        Inject the path as a file that can be read by other running scripts.
    """
    with open(IME_EVENT_PATH_STORAGE, "w") as f:
        f.write(absolute_IME_path + "\n")
    print("IME event capturer set up successfully.")

def nullify_IMEevent_capturer(bin_IME_path: Path = BIN_IME_PATH) -> None:
    """
        Nullify the IME event capturer by redirecting to global bin.
    """
    with open(IME_EVENT_PATH_STORAGE, "w") as f:
        f.write(str(bin_IME_path) + "\n")


def terminate_process_gracefully(proc: subprocess.Popen):
    """Terminate a process and all its children."""
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        # Terminate children first
        for child in children:
            try:
                child.terminate()  # or .kill() for force kill
            except psutil.NoSuchProcess:
                pass
        # Then terminate parent (the shell)
        parent.terminate()
        # Wait for a moment
        _, alive = psutil.wait_procs(children, timeout=3)
        # Kill any remaining ones
        for p in alive:
            p.kill()
        proc.wait(timeout=5)
    except psutil.NoSuchProcess:
        pass  # already gone


def start_gesture_recording(adb_path, output_path="gestures.log"):
    """
    Begins capturing raw touch input events from the Android device.
    Returns a tuple (process, file_handle) you can use to stop later.
    """

    # open local file for writing raw events (line-buffered text mode)
    cmd = f"{adb_path} shell -t -t getevent -lt > {output_path}"
    # capture stdout so we can flush each line immediately
    # set stdin to be /dev/null
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL, text=True)

    return proc

def stop_gesture_recording(record_proc: subprocess.Popen):
    """
    Stops the running getevent process and closes the output file.
    """
    # file_handle.flush()
    terminate_process_gracefully(record_proc)
    print("Stopped gesture recording.")

def start_sensor_recording(adb_path, sensor_apk_name, input_sensor_path):
    print(f"Starting sensor recording with APK: {sensor_apk_name}")
    subprocess.run(f"{adb_path} shell rm {input_sensor_path}", capture_output=True, text=True, shell=True)
    subprocess.run(f"{adb_path} shell am start -S {sensor_apk_name}/.MainActivity", capture_output=True, text=True, shell=True)
    print(f"Started? sensor recording with APK: {sensor_apk_name}")

def stop_sensor_recording(sensor_apk_name, input_sensor_path, output_sensor_path):
    print(f"Stopping sensor recording for APK: {sensor_apk_name}")
    # Stop the sensor app
    subprocess.run(f"adb shell am force-stop {sensor_apk_name}", capture_output=True, text=True, shell=True)
    
    # Pull the sensor data file from the device
    subprocess.run(f"adb pull {input_sensor_path} {output_sensor_path}", capture_output=True, text=True, shell=True)
    
    print(f"Sensor data saved to: {output_sensor_path}")
