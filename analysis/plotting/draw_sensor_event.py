import re
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import pandas as pd
from pathlib import Path
import argparse


def parse_sensor_file(file_path: str
) -> Tuple[int,        Dict[str,    Dict[str,               List[float]]]]:
    """                      ^            ^                      ^
a single timestamp offset and             t/x/y/z
                    object [sensor_type][component_name] -> ordered_series 
    """
    # every sensor has a dict, keys are 'x', 'y', 'z' or something else
    sensor_data = {
        'Accelerometer': {'t': [], 'x': [], 'y': [], 'z': []},
        'Gyroscope': {'t': [], 'x': [], 'y': [], 'z': []},
        'RotationVector': {'t': [], 'i': [], 'j': [], 'k': [], "w": [], "accuracy": []},
        "Gravity": {'t': [], 'x': [], 'y': [], 'z': []},
        "LinearAcceleration": {'t': [], 'x': [], 'y': [], 'z': []},
        "MagneticField": {'t': [], 'x': [], 'y': [], 'z': []},
        "Light": {'t': [], 'value': []},
        "MotionDetector": {"t": [], "value": []},
        "Pressure": {"t": [], "value": []},
        "Proximity": {"t": [], "value": []},
        "StepCounter": {"t": [], "value": []},
        "StepDetector": {"t": [], "value": []},
    }

    first_line_number: int = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        first_line = lines[0]
        first_line_number_matcher = re.match(r"^(\d+)$", first_line)
        if first_line_number_matcher and (first_line_number == 0):
            first_line_number = int(first_line_number_matcher.group(1))
        else:
            raise ValueError(f"Unexpected first line format: {first_line} in file {file_path}; should be an offset in nanoseconds")
        
        for line in lines[1:]:

            
            
            # first determine the type:
            m_type = re.match(r"(\d+)\s+(Accelerometer|Gyroscope|RotationVector|Gravity|LinearAcceleration|MagneticField|Light|MotionDetector|Pressure|Proximity|StepCounter|StepDetector):\s*", line)
            if m_type is None:
                continue
            t_str, sensor_type = m_type.groups()
            t = int(t_str)

            if sensor_type in ["Light", "MotionDetector", "Pressure", "Proximity", "StepCounter", "StepDetector"]:
                m_value = re.match(r"(\d+)\s+(Light|MotionDetector|Pressure|Proximity|StepCounter|StepDetector):\s*value=([-\d.]+)", line)
                if m_value is None:
                    raise ValueError(f"Cannot parse line: {line}, which is expected to have single value for sensor {sensor_type}")
                t, sensor, value = m_value.groups()
                sensor_data[sensor]['t'].append(int(t))
                sensor_data[sensor]['value'].append(float(value))

            elif sensor_type == "RotationVector":
                m_value = re.match(r"(\d+)\s+RotationVector:\s*i=([-\d.]+),\s*j=([-\d.]+),\s*k=([-\d.]+),\s*w=([-\d.]+),\s*accuracy=([-\d.]+)", line)
                if m_value is None:
                    raise ValueError(f"Cannot parse line: {line}, which is expected to have i/j/k/w/accuracy for sensor RotationVector")
                t, i, j, k, w, accuracy = m_value.groups()
                sensor_data['RotationVector']['t'].append(int(t))
                sensor_data['RotationVector']['i'].append(float(i))
                sensor_data['RotationVector']['j'].append(float(j))
                sensor_data['RotationVector']['k'].append(float(k))
                sensor_data['RotationVector']['w'].append(float(w))
                sensor_data['RotationVector']['accuracy'].append(float(accuracy))

            elif sensor_type in ["Accelerometer", "Gyroscope", "Gravity", "LinearAcceleration", "MagneticField"]:
                m_value = re.match(r"(\d+)\s+(Accelerometer|Gyroscope|Gravity|LinearAcceleration|MagneticField):\s*x=([-\d.]+)\s*y=([-\d.]+)\s*z=([-\d.]+)", line)
                if m_value is None:
                    raise ValueError(f"Cannot parse line: {line}, which is expected to have x/y/z for sensor {sensor_type}")
                t, sensor, x, y, z = m_value.groups()
                sensor_data[sensor]['t'].append(int(t))
                sensor_data[sensor]['x'].append(float(x))
                sensor_data[sensor]['y'].append(float(y))
                sensor_data[sensor]['z'].append(float(z))
            else:
                raise ValueError(f"Unknown sensor type in line: {line}")
            
    return first_line_number, sensor_data

def parse_as_df(file_path: str) -> Tuple[int, Dict[str, pd.DataFrame]]:
    first_line_number, sensor_data = parse_sensor_file(file_path)
    sensor_data_df: Dict[str, pd.DataFrame] = {}
    for sensor_type in sensor_data.keys():
        pseudo_dataframe = sensor_data[sensor_type]
        sensor_data_df[sensor_type] = pd.DataFrame(data=pseudo_dataframe)
    return first_line_number, sensor_data_df

def sensor_generator_from_files(df_idx: pd.DataFrame, logs_dir: Path
        ) -> List[Tuple[str, Tuple[int, Dict[str, pd.DataFrame]]]]:
    """Read df_idx(the catalog of logs), iterate logs, extract unmodified swipes, and return a list.    
    The resulting DataFrame has a 'type' column followed by feature columns.
    """
    result: List[Tuple[str, Tuple[int, Dict[str, pd.DataFrame]]]] = []
    for _, row in df_idx.iterrows():
        log_num: str = str(row["log_num"])  # e.g., 20250714_162513
        label: str = str(row["type"])      # class label
        # Construct file name like draw_motion_event2 defaults
        log_file = logs_dir / f"sensor_recording_{log_num}.txt"
        if not log_file.exists():
            print(f"Missing log: {log_file}")
            continue
        result.append((label, parse_as_df(str(log_file))))
    return result

def plot_sensor_data(sensor_data: Dict[str, Union[Dict[str, List[float]], pd.DataFrame]]):
    plt.figure(figsize=(16, 10))
    xyz_sensors = ['Accelerometer', 'Gyroscope']

    ijkwacc_sensors = ['RotationVector']
    single_sensors = ['Light', 'MotionDetector', 'Pressure', 'Proximity', 'StepCounter', 'StepDetector'][-2:-1]

    total_plots = len(xyz_sensors) + len(ijkwacc_sensors) + len(single_sensors)


    ax0 = None  # To store the first axis for sharing x-axis

    for i, sensor in enumerate(xyz_sensors, 1):
        if ax0 is None:
            ax0 = plt.subplot(total_plots, 1, i)
        else:
            plt.subplot(total_plots, 1, i, sharex=ax0)

        t = sensor_data[sensor]['t']
        x = sensor_data[sensor]['x']
        y = sensor_data[sensor]['y']
        z = sensor_data[sensor]['z']
        if len(t) == 0:
            continue
        # use relative time (ms) for x-axis
        t0 = t[0]
        t_rel = [(tt - t0) / 1e6 for tt in t]  # from nanoseconds to milliseconds

        plt.plot(t_rel, x, label='x', color='tab:blue')
        plt.plot(t_rel, y, label='y', color='tab:orange')
        plt.plot(t_rel, z, label='z', color='tab:green')
        plt.title(sensor)
        # plt.xlabel('Time (ms)') # Only show label on bottom plot if desired, or keep for all
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    
    for i, sensor in enumerate(ijkwacc_sensors, len(xyz_sensors) + 1):
        if ax0 is None:
            ax0 = plt.subplot(total_plots, 1, i)
        else:
            plt.subplot(total_plots, 1, i, sharex=ax0)

        t = sensor_data[sensor]['t']
        i_comp = sensor_data[sensor]['i']
        j_comp = sensor_data[sensor]['j']
        k_comp = sensor_data[sensor]['k']
        w_comp = sensor_data[sensor]['w']
        accuracy_comp = sensor_data[sensor]['accuracy']
        if len(t) == 0:
            continue
        # use relative time (ms) for x-axis
        t0 = t[0]
        t_rel = [(tt - t0) / 1e6 for tt in t]  # from nanoseconds to milliseconds

        plt.plot(t_rel, i_comp, label='i', color='tab:blue')
        plt.plot(t_rel, j_comp, label='j', color='tab:orange')
        plt.plot(t_rel, k_comp, label='k', color='tab:green')
        plt.plot(t_rel, w_comp, label='w', color='tab:red')
        plt.plot(t_rel, accuracy_comp, label='accuracy', color='tab:purple')
        plt.title(sensor)
        # plt.xlabel('Time (ms)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

    for i, sensor in enumerate(single_sensors, len(xyz_sensors) + len(ijkwacc_sensors) + 1):
        if ax0 is None:
            ax0 = plt.subplot(total_plots, 1, i)
        else:
            plt.subplot(total_plots, 1, i, sharex=ax0)

        t = sensor_data[sensor]['t']
        value = sensor_data[sensor]['value']
        if len(t) == 0:
            continue
        # use relative time (ms) for x-axis
        t0 = t[0]
        t_rel = [(tt - t0) / 1e6 for tt in t]  # from nanoseconds to milliseconds

        plt.plot(t_rel, value, label='value', color='tab:blue')
        plt.title(sensor)
        plt.xlabel('Time (ms)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and plot sensor data from log file.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the sensor log file. e.g. 'logs/sensor_recording_20251209_070431.txt'")
    args = parser.parse_args()
    file_path = args.file_path
    first_line_number, data = parse_sensor_file(file_path)
    plot_sensor_data(data)
