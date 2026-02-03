import re
import matplotlib.pyplot as plt
import numpy as np

def hex_to_dec(hex_str):
    """16进制转10进制"""
    return int(hex_str, 16)

def parse_motionevent_log(file_path):
    """解析 motionevent 日志，提取XY坐标"""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    x_pattern = re.compile(r'ABS_MT_POSITION_X\s+([0-9a-f]+)')
    y_pattern = re.compile(r'ABS_MT_POSITION_Y\s+([0-9a-f]+)')
    
    x_coords = []
    y_coords = []

    x_current = None
    y_current = None

    for line in lines:
        x_match = x_pattern.search(line)
        y_match = y_pattern.search(line)
        
        if x_match:
            x_current = hex_to_dec(x_match.group(1))
        if y_match:
            y_current = hex_to_dec(y_match.group(1))
        
        # 每次 SYN_REPORT 时确认记录当前坐标
        if "SYN_REPORT" in line and x_current is not None and y_current is not None:
            x_coords.append(x_current)
            y_coords.append(y_current)

    return np.array(x_coords), np.array(y_coords)

def fit_line(x, y):
    """最小二乘法拟合直线 y = kx + b"""
    k, b = np.polyfit(x, y, 1)
    return k, b

def plot_results(x, y, k, b):
    """绘制触摸点及拟合直线"""
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Touch Points')
    
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = k * x_fit + b
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label=f'Fit Line: y={k:.2f}x+{b:.2f}')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Touchscreen MotionEvent Trajectory')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()  # y轴翻转，以符合屏幕坐标系
    plt.show()

if __name__ == "__main__":
    file_path = "./logs/gesture_recording_20250710_150007.log"
    x, y = parse_motionevent_log(file_path)
    print(x,y)
    k, b = fit_line(x, y)

    print(f"拟合直线方程: y = {k:.4f}x + {b:.4f}")

    plot_results(x, y, k, b)
