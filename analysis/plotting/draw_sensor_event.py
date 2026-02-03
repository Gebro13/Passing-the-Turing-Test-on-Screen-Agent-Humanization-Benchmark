import re
import matplotlib.pyplot as plt

def parse_sensor_file(file_path):
    # 每种传感器一个dict，键为 'x', 'y', 'z'
    sensor_data = {
        'Accelerometer': {'t': [], 'x': [], 'y': [], 'z': []},
        'Gyroscope': {'t': [], 'x': [], 'y': [], 'z': []},
        'RotationVector': {'t': [], 'x': [], 'y': [], 'z': []},
    }

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 时间戳
            m = re.match(r"(\d+)\s+(Accelerometer|Gyroscope|RotationVector):\s*x=([-\d.]+)\s*y=([-\d.]+)\s*z=([-\d.]+)", line)
            if m:
                t, sensor, x, y, z = m.groups()
                sensor_data[sensor]['t'].append(int(t))
                sensor_data[sensor]['x'].append(float(x))
                sensor_data[sensor]['y'].append(float(y))
                sensor_data[sensor]['z'].append(float(z))
    return sensor_data

def plot_sensor_data(sensor_data):
    plt.figure(figsize=(16, 10))
    sensors = ['Accelerometer', 'Gyroscope', 'RotationVector']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for i, sensor in enumerate(sensors, 1):
        plt.subplot(3, 1, i)
        t = sensor_data[sensor]['t']
        x = sensor_data[sensor]['x']
        y = sensor_data[sensor]['y']
        z = sensor_data[sensor]['z']
        if len(t) == 0:
            continue
        # 横坐标转为相对时间（ms），更直观
        t0 = t[0]
        t_rel = [(tt - t0) / 1e6 for tt in t]  # 转为毫秒

        plt.plot(t_rel, x, label='x', color='tab:blue')
        plt.plot(t_rel, y, label='y', color='tab:orange')
        plt.plot(t_rel, z, label='z', color='tab:green')
        plt.title(sensor)
        plt.xlabel('Time (ms)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = './logs/sensor_recording_20250714_162513.txt'  # 替换为你的传感器日志文件路径
    data = parse_sensor_file(file_path)
    plot_sensor_data(data)
