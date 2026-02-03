import pandas as pd
from pathlib import Path
from analysis.lib.gesture_log_reader_utils import filtered_gesture_generator_from_files
from draw_sensor_event import sensor_generator_from_files
from analysis.processing.tap_duration_extract import leave_taps, _gesture_start_end_us
import argparse
import matplotlib.pyplot as plt
from typing import List

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract both sensor and gesture from logs.")
    parser.add_argument("--index_csv", type=str, default=str(Path(__file__).with_name("logs_inf.csv")), help="Path to logs_inf.csv")
    parser.add_argument("--logs_dir", type=str, default=str(Path(__file__).with_name("logs")), help="Directory containing gesture_recording_*.log files")
    parser.add_argument("--temp_img_output", type=str, default=str(Path(__file__).with_name("temp_sensor_gesture_img.png")), help="Directory to save the temporary image")
    
    interested_sensor_name: str = "Accelerometer"
    interested_components_name: List[str] = ["x", "y", "z"]
    interested_label: str = "user3-manual-frequent-nop"
    # parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save the resulting CSV with columns [type, interval_us]")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    df_idx = pd.read_csv(args.index_csv)
    temp_img_output: str = args.temp_img_output

    gesture_generator = filtered_gesture_generator_from_files(df_idx, logs_dir, leave_taps)
    sensor_generator = sensor_generator_from_files(df_idx, logs_dir)

    for (participant_tag, session_timestamp, file_gesture_generator), \
        (sensor_label, (time_offset, sensor_pd_dict)) \
            in zip(gesture_generator, sensor_generator):
        assert participant_tag == sensor_label, f"Label mismatch: {participant_tag} vs {sensor_label}"
        if interested_label != participant_tag:
            continue
        print("time offset: ", time_offset, " ns")
        offset_ns = 0 if time_offset < 2000000 else time_offset
        interested_data = sensor_pd_dict[interested_sensor_name]
        for gesture in file_gesture_generator:
            start_us, end_us = _gesture_start_end_us(gesture)
            start_ns, end_ns = start_us * 1000 + offset_ns, end_us * 1000 + offset_ns
            delta_edge = (end_ns - start_ns) * 10
            interested_data_extraction = interested_data[
                (interested_data['t'] >= start_ns - delta_edge) &
                (interested_data['t'] <= end_ns   + delta_edge)].copy()
            # interested_data_extraction["t"] = (interested_data_extraction["t"] - start_ns) / (end_ns - start_ns)  # normalize to [0, 1]
            for interested_component_name in interested_components_name:
                plt.plot(interested_data_extraction['t'], interested_data_extraction[interested_component_name], label=f"{interested_sensor_name}-{interested_component_name}")

            plt.axvline(start_ns, color='r', linestyle='--')
            plt.axvline(end_ns, color='r', linestyle='--')
            plt.legend()
            plt.title(f"{participant_tag} total duration {(end_ns - start_ns) / 1e6}ms")
            plt.savefig(temp_img_output)
            input("Next plot:")
            plt.clf()

    quit()
    # Check integrity of sensor logs
    sensor_generator = sensor_generator_from_files(df_idx, logs_dir)
    for label, (first_line_number, sensor_data) in sensor_generator:
        assert isinstance(first_line_number, int), f"First line number is not int in label {label}"
        assert isinstance(sensor_data, dict), f"Sensor data is not dict in label {label}"
        for sensor_type in ['Accelerometer', 'Gyroscope', 'RotationVector']:
            assert sensor_type in sensor_data, f"Missing {sensor_type} data in label {label}"
            for axis in ['t', 'x', 'y', 'z']:
                assert axis in sensor_data[sensor_type], f"Missing {axis} axis in {sensor_type} data for label {label}"
                assert isinstance(sensor_data[sensor_type][axis], list), f"{axis} data is not list in {sensor_type} for label {label}"

    print("All sensor logs are valid.")