import matplotlib.pyplot as plt
from controller import FingerEvent
from typing import Iterator, Optional
import argparse
import numpy as np
from gesture_log_reader_utils import single_trace_generator, file_reader_yield

def throw_away_timestamp(gesture_generator: Iterator[list[FingerEvent]]) -> list[list[tuple[int, int]]]:
    """
    Convert FingerEvent traces to a list of (x, y) tuples, discarding timestamps.
    """
    gestures = []
    for gesture in gesture_generator:
        gestures.append([(event.x, event.y) for event in gesture])
    return gestures

def plot_gestures(gestures):
    plt.figure(figsize=(10, 8))

    for i, gesture in enumerate(gestures):
        x_coords, y_coords = zip(*gesture)
        if x_coords[0] is None or y_coords[0] is None:
            # this is a problem of getevent, so not fixable.
            # but we can skip this point.
            print("Gesture {i+1} begin with None coordinates, skipping this point.")
            x_coords = x_coords[1:]
            y_coords = y_coords[1:]
        # https://www.tutorialspoint.com/how-do-i-specify-an-arrow-like-linestyle-in-matplotlib
        x, y = np.array(x_coords), np.array(y_coords)
        plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, color=np.random.rand(3,), label=f'Gesture {i+1}', alpha=0.6)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Detected Gestures from MotionEvent Log')
    plt.legend()
    plt.grid(True)
    # x window 0-1078, y window 0-1918
    plt.tight_layout()
    plt.xlim(0, 1078)
    plt.ylim(0, 1918)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().xaxis.set_ticks_position('top')
    plt.show()

def save_gestures(gestures, output_path):
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gestures, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    #     file_path = "./logs/gesture_recording_20250714_162513.log"
    #     output_path = "gestures.json"
    parser = argparse.ArgumentParser(description="Process ADB events and save gestures.")
    parser.add_argument('--file_path', type=str, help='Path to the ADB event log file', default='./logs/gesture_recording_20250714_162513.log')
    parser.add_argument('--output_path', type=str, help='Path to save the gestures JSON file', default='gestures.json')
    parser.add_argument("--thres_after_1st_menu_s", type=int, default=None, help="Legacy parameter to deal with getevents that didn't terminate properly.")
    args = parser.parse_args()


    gestures = throw_away_timestamp(single_trace_generator(file_reader_yield(args.file_path), args.thres_after_1st_menu_s))

    print(f"总共识别到 {len(gestures)} 个手势操作")
    for idx, gesture in enumerate(gestures, 1):
        print(f"手势 {idx} 有 {len(gesture)} 个坐标点")

    save_gestures(gestures, args.output_path)
    print(f"手势数据已保存到 {args.output_path}")

    plot_gestures(gestures)
