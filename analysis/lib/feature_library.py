import os
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Any, List, Dict, Tuple, Union
from analysis.lib.motionevent_classes import FingerEvent

# https://www.britannica.com/story/whats-the-difference-between-speed-and-velocity
# speed is a scalar value while velocity is with a direction

def pickling(fname: str, obj: Any) -> None:
    f = open(fname, "wb")
    pickle.dump(obj, f)
    f.close()

def unpickling(fname: str) -> Any:
    f = open(fname, 'rb')
    g = pickle.load(f) 
    f.close()
    return g

# Function to convert a given time string to linux time
# Input format: 'Y-m-d H:M:S.MS'
def convert_time_to_linux(time_str: str) -> int:
    pos = time_str.rfind(':')
    time_str = "".join((time_str[:pos], '.', time_str[pos+1:])) 
    return pd.to_datetime(time_str).value//10**6

def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return np.sqrt((x1-x2)**2 + (y1 - y2)**2)

def length_of_swipe(swipe: List[FingerEvent]) -> float:
    d = 0
    for i in range(1,len(swipe)):
        d += euclidean_distance(swipe[i-1].x, swipe[i-1].y, swipe[i].x, swipe[i].y)
    return d

def magnitude_of_average_velocity_of_swipe(swipe: List[FingerEvent]) -> float:
    displacement = euclidean_distance(swipe[0].x, swipe[0].y, swipe[-1].x, swipe[-1].y)
    time = swipe[-1].timestamp_us - swipe[0].timestamp_us
    velocity = displacement/time
    return velocity

"""
@DeprecationWarning
def area_of_swipe(swipe: List[FingerEvent]) -> float:
    # Area of each point and averaging it per swipe
    area_per_point = 0
    for i in range(0,len(swipe)-1):
        area_per_point += np.pi * swipe[i]['touchMajor'] * swipe[i]['touchMinor']
    avg_area = area_per_point/len(swipe)
    return avg_area

@DeprecationWarning
def median_area_of_swipe(swipe: List[FingerEvent]) -> float:
    # Area of each point and averaging it per swipe
    area_per_point = []
    for i in range(0, len(swipe) - 1):
        area_per_point.append(np.pi * swipe[i]['touchMajor'] * swipe[i]['touchMinor'])
    med_area = np.median(area_per_point)
    return med_area
"""

def get_direction(x1: float, y1: float, x2: float, y2: float) -> float:
    return np.arctan2(y2 - y1, x2 - x1) # using tan is erroneous; use arctan2 to exact TouchAlytics

def acceleration_of_swipe(swipe: List[FingerEvent]) -> float:
    # BUG GY
    velocity = magnitude_of_average_velocity_of_swipe(swipe)
    if (swipe[-1].timestamp_us - swipe[0].timestamp_us) == 0:
        return 0
    return velocity/(swipe[-1].timestamp_us - swipe[0].timestamp_us)

def get_pairwise_velocities_X(swipe: List[FingerEvent]) -> List[float]:
    v = [0.0]
    for i in range(1, len(swipe)):
        d = swipe[i].x - swipe[i-1].x
        t = swipe[i].timestamp_us - swipe[i-1].timestamp_us
        if(t == 0):
            v.append(v[-1])
            continue
        v.append(d/t)
    return v

def get_pairwise_velocities_Y(swipe: List[FingerEvent]) -> List[float]:
    v = [0.0]
    for i in range(1, len(swipe)):
        d = swipe[i].y - swipe[i-1].y
        t = swipe[i].timestamp_us - swipe[i-1].timestamp_us
        if(t == 0):
            v.append(v[-1])
            continue
        v.append(d/t)
    return v

def get_pairwise_accelerations(swipe: List[FingerEvent], velocities: List[float]) -> List[float]:
    a = [0.0]
    for i in range(1, len(velocities)):
        dv = velocities[i] - velocities[i-1]
        dt = swipe[i].timestamp_us - swipe[i-1].timestamp_us
        if(dt == 0):
            a.append(a[-1])
            continue
        a.append(dv/dt)
    return a

def get_average_acceleration(swipe: List[FingerEvent]) -> float:
    # Getting acceleration for each point and averaging it per swipe
    avg = 0
    res = 0
    for i in range(1,len(swipe)):
        displacement = euclidean_distance(swipe[i-1].x, swipe[i-1].y, swipe[i].x, swipe[i].y)
        time = swipe[i].timestamp_us - swipe[i-1].timestamp_us
        if time == 0:
            continue
        res += displacement/(time **2)
    avg = res/len(swipe)
    return avg

def get_initial_acceleration(swipe: List[FingerEvent]) -> float:
    # Considering inital 5 percent of points per swipe
    n = 0.05 * len(swipe)
    displacement = euclidean_distance(swipe[0].x, swipe[0].y, swipe[int(n)].x, swipe[int(n)].y)
    time = swipe[int(n)].timestamp_us - swipe[0].timestamp_us
    if time == 0:
        return 0
    acc = displacement/(time ** 2)
    return acc

def get_acceleration_percentile_25(swipe: List[FingerEvent]) -> float:
    # Considering inital 5 percent of points per swipe TYPO BUG?
    n = 0.25 * len(swipe)
    displacement = euclidean_distance(swipe[0].x, swipe[0].y, swipe[int(n)].x, swipe[int(n)].y)
    time = swipe[int(n)].timestamp_us - swipe[0].timestamp_us
    if time == 0:
        return 0
    acc = displacement/(time ** 2)
    return acc

def get_acceleration_percentile_50(swipe: List[FingerEvent]) -> float:
    # Considering inital 5 percent of points per swipe TYPO BUG?
    n = 0.5 * len(swipe)
    displacement = euclidean_distance(swipe[0].x, swipe[0].y, swipe[int(n)].x, swipe[int(n)].y)
    time = swipe[int(n)].timestamp_us - swipe[0].timestamp_us
    if time == 0:
        return 0
    acc = displacement/(time ** 2)
    return acc

def get_acceleration_percentile_75(swipe: List[FingerEvent]) -> float:
    # Considering inital 5 percent of points per swipe TYPO BUG?
    n = 0.75 * len(swipe)
    displacement = euclidean_distance(swipe[0].x, swipe[0].y, swipe[int(n)].x, swipe[int(n)].y)
    time = swipe[int(n)].timestamp_us - swipe[0].timestamp_us
    if time == 0:
        return 0
    acc = displacement/(time ** 2)
    return acc

def get_final_acceleration(swipe: List[FingerEvent]) -> float:
    # Considering final 5 percent of points per swipe TYPO BUG?
    n = 0.05 * len(swipe)
    displacement = euclidean_distance(swipe[int(-n)].x, swipe[int(-n)].y, swipe[-1].x, swipe[-1].y)
    time = swipe[-1].timestamp_us - swipe[int(-n)].timestamp_us
    if time == 0:
        return 0
    acc = displacement/(time ** 2)
    return acc

def get_magnitude_of_average_initial_velocity(swipe: List[FingerEvent]) -> float:
    # Considering inital 5 percent of points per swipe
    n = 0.05 * len(swipe)
    displacement = euclidean_distance(swipe[0].x, swipe[0].y, swipe[int(n)].x, swipe[int(n)].y)
    time = swipe[int(n)].timestamp_us - swipe[0].timestamp_us
    if time == 0:
        return 0
    velocity = displacement/time
    return velocity

def get_velocity_percentile_25(swipe: List[FingerEvent]) -> float:
    # Considering inital 5 percent of points per swipe TYPO BUG?
    n = 0.25 * len(swipe)
    displacement = euclidean_distance(swipe[0].x, swipe[0].y, swipe[int(n)].x, swipe[int(n)].y)
    time = swipe[int(n)].timestamp_us - swipe[0].timestamp_us
    if time == 0:
        return 0
    vel = displacement/time
    return vel

def get_velocity_percentile_50(swipe: List[FingerEvent]) -> float:
    # Considering inital 5 percent of points per swipe TYPO BUG?
    n = 0.5 * len(swipe)
    displacement = euclidean_distance(swipe[0].x, swipe[0].y, swipe[int(n)].x, swipe[int(n)].y)
    time = swipe[int(n)].timestamp_us - swipe[0].timestamp_us
    if time == 0:
        return 0
    vel = displacement/time
    return vel

def get_velocity_percentile_75(swipe: List[FingerEvent]) -> float:
    # Considering inital 5 percent of points per swipe TYPO BUG?
    n = 0.75 * len(swipe)
    displacement = euclidean_distance(swipe[0].x, swipe[0].y, swipe[int(n)].x, swipe[int(n)].y)
    time = swipe[int(n)].timestamp_us - swipe[0].timestamp_us
    if time == 0:
        return 0
    vel = displacement/time
    return vel

def get_magnitude_of_average_final_velocity(swipe: List[FingerEvent]) -> float:
    # Considering final 5 percent of points per swipe
    n = 0.05 * len(swipe)
    displacement = euclidean_distance(swipe[int(-n)].x, swipe[int(-n)].y, swipe[-1].x, swipe[-1].y)
    time = swipe[-1].timestamp_us - swipe[int(-n)].timestamp_us
    if time == 0:
        return 0
    velocity = displacement/time
    return velocity

def get_average_speed(swipe: List[FingerEvent]) -> float:
    # Getting velocity for each point and averaging it per swipe
    avg = 0
    res = 0
    for i in range(1,len(swipe)):
        displacement = euclidean_distance(swipe[i-1].x, swipe[i-1].y, swipe[i].x, swipe[i].y)
        time = swipe[i].timestamp_us - swipe[i-1].timestamp_us
        if time == 0:
            continue
        res += displacement/time
    avg = res/len(swipe)
    return avg

def speed_of_swipe(swipe: List[FingerEvent]) -> float:
    distance = length_of_swipe(swipe)
    time = swipe[-1].timestamp_us - swipe[0].timestamp_us
    if time == 0:
        return 0
    speed = distance/time
    return speed

def get_final_speed(swipe: List[FingerEvent]) -> float:
    # Considering inital 5 percent of points per swipe TYPO BUG?
    n = 0.05 * len(swipe)
    distance = 0
    time = 0
    for i in range(1,int(n)+1):
        distance += euclidean_distance(swipe[-i-1].x, swipe[-i-1].y, swipe[-i].x, swipe[-i].y)
        time += swipe[-i].timestamp_us - swipe[-i-1].timestamp_us
    if time == 0:
        return 0
    speed = distance/time
    return speed

def get_initial_speed(swipe: List[FingerEvent]) -> float:
    # Considering inital 5 percent of points per swipe TYPO BUG?
    n = 0.05 * len(swipe)
    distance = 0
    time = 0
    for i in range(1,int(n)+1):
        distance += euclidean_distance(swipe[i-1].x, swipe[i-1].y, swipe[i].x, swipe[i].y)
        time += swipe[i].timestamp_us - swipe[i-1].timestamp_us
    if time == 0:
        return 0
    speed = distance/time
    return speed 

def get_speed_percentile_25(swipe: List[FingerEvent]) -> float:
    # Considering inital 5 percent of points per swipe TYPO BUG?
    n = 0.25 * len(swipe)
    distance = 0
    time = 0
    for i in range(1,int(n)+1):
        distance += euclidean_distance(swipe[i-1].x, swipe[i-1].y, swipe[i].x, swipe[i].y)
        time += swipe[i].timestamp_us - swipe[i-1].timestamp_us
    if time == 0:
        return 0
    speed = distance/time
    return speed 

def get_speed_percentile_50(swipe: List[FingerEvent]) -> float:
    # Considering inital 5 percent of points per swipe TYPO BUG?
    n = 0.50 * len(swipe)
    distance = 0
    time = 0
    for i in range(1,int(n)+1):
        distance += euclidean_distance(swipe[i-1].x, swipe[i-1].y, swipe[i].x, swipe[i].y)
        time += swipe[i].timestamp_us - swipe[i-1].timestamp_us
    if time == 0:
        return 0
    speed = distance/time
    return speed

def get_speed_percentile_75(swipe: List[FingerEvent]) -> float:
    # Considering inital 5 percent of points per swipe TYPO BUG?
    n = 0.75 * len(swipe)
    distance = 0
    time = 0
    for i in range(1,int(n)+1):
        distance += euclidean_distance(swipe[i-1].x, swipe[i-1].y, swipe[i].x, swipe[i].y)
        time += swipe[i].timestamp_us - swipe[i-1].timestamp_us
    if time == 0:
        return 0
    speed = distance/time
    return speed

def get_deviations(swipe: List[FingerEvent]) -> List[float]:
    devs = []
    if(swipe[0].x == swipe[-1].x):
        for i in swipe:
            devs.append(abs(i.x - swipe[0].x))
        return devs
    if(swipe[0].y == swipe[-1].y):
        for i in swipe:
            devs.append(abs(i.y - swipe[0].y))
        return devs
    p1 = np.array([swipe[0].x, swipe[0].y])
    p2 = np.array([swipe[-1].x, swipe[-1].y])
    for i in swipe:
        p3 = np.array([i.x, i.y])
        d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
        devs.append(d)
    return devs



# ---------------------------
# Touchalytics-aligned helpers
# ---------------------------

def _pairwise_speeds(swipe: List[FingerEvent]) -> List[float]:
    """
    Instantaneous speed magnitudes between successive points: vi = dist(i-1,i) / dt.
    
    """

    # """Skips zero-dt intervals by repeating last valid speed."""
    if len(swipe) < 2:
        return []
    speeds: List[float] = []
    # last_v = 0.0
    for i in range(1, len(swipe)):
        dx = swipe[i].x - swipe[i-1].x
        dy = swipe[i].y - swipe[i-1].y
        dt = swipe[i].timestamp_us - swipe[i-1].timestamp_us
        if dt <= 0:
            raise IndexError("Non-positive time difference between successive points.")
            speeds.append(last_v)
            continue
        v = float(np.hypot(dx, dy)) / float(dt)
        speeds.append(v)
        # last_v = v
    return speeds

def _pairwise_accelerations_from_speeds(swipe: List[FingerEvent], speeds: List[float]) -> List[float]:
    """
    Signed acceleration between successive speed samples: ai = (v_i - v_{i-1}) / dt.
    Uses the same dt(i) as between the corresponding points; """ # repeats last valid a if dt<=0.
    # """
    if len(swipe) < 3 or len(speeds) < 2:
        return []
    accs: List[float] = []
    # last_a = 0.0
    # speeds aligns to segments [1..n-1]; acceleration aligns to segments [2..n-1]
    for i in range(2, len(swipe)):
        dv = speeds[i-1] - speeds[i-2]
        dt = swipe[i].timestamp_us - swipe[i-1].timestamp_us
        if dt <= 0:
            raise IndexError("Non-positive time difference between successive points.")
            accs.append(last_a)
            continue
        a = dv / float(dt)
        accs.append(a)
        # last_a = a
    return accs

def _segment_directions(swipe: List[FingerEvent]) -> List[float]:
    """
    Angles (radians) of each segment i-1 -> i via atan2(dy, dx).
    """
    if len(swipe) < 2:
        return []
    thetas: List[float] = []
    for i in range(1, len(swipe)):
        dy = swipe[i].y - swipe[i-1].y
        dx = swipe[i].x - swipe[i-1].x
        thetas.append(float(np.arctan2(dy, dx)))
    return thetas

def _circular_mean(angles: List[float]) -> float:
    """
    Circular mean of angles in radians, in (-pi, pi].
    """
    if not angles:
        return 0.0
    C = float(np.mean(np.cos(angles)))
    S = float(np.mean(np.sin(angles)))
    return float(np.arctan2(S, C))

def _mean_resultant_length(angles: List[float]) -> float:
    """
    Mean resultant length R in [0,1] for a set of angles.
    """
    if not angles:
        return 0.0
    C = float(np.mean(np.cos(angles)))
    S = float(np.mean(np.sin(angles)))
    return float(np.hypot(C, S))

def _signed_deviations_from_line(swipe: List[FingerEvent]) -> np.ndarray:
    """
    Signed perpendicular deviation of each point to the end-to-end line.
    Positive sign corresponds to left side of the vector p1->p2 (right-handed).
    """
    n = len(swipe)
    if n == 0:
        return np.array([], dtype=float)
    p1 = np.array([swipe[0].x, swipe[0].y], dtype=float)
    p2 = np.array([swipe[-1].x, swipe[-1].y], dtype=float)
    v = p2 - p1
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return np.zeros(n, dtype=float)
    # 2D signed perpendicular distance can be computed via cross product z-component
    devs: List[float] = []
    for i in range(n):
        p = np.array([swipe[i].x, swipe[i].y], dtype=float)
        w = p - p1
        cross_z = v[0]*w[1] - v[1]*w[0]
        devs.append(cross_z / v_norm)
    return np.asarray(devs, dtype=float)

def _percentile(values: List[float], p: float) -> float:
    """
    Percentile p in [0,100] for a list; returns 0.0 if empty.
    """
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))

# ------------------------------------------
# Touchalytics feature functions (F10, F14+)
# ------------------------------------------

def mean_resultant_length(swipe: List[FingerEvent]) -> float:
    """
    F10: Mean resultant length (MRL) of segment directions.
    """
    return _mean_resultant_length(_segment_directions(swipe)) # DONE check circ_r

def velocity_percentile(swipe: List[FingerEvent], p: float) -> float:
    """
    F14/F15/F16: Percentiles (p = 20/50/80) of instantaneous speed magnitudes.
    """
    v = _pairwise_speeds(swipe)
    return _percentile(v, p)

def acceleration_percentile(swipe: List[FingerEvent], p: float) -> float:
    """
    F17/F18/F19: Percentiles (p = 20/50/80) of signed accelerations (dv/dt).
    """
    v = _pairwise_speeds(swipe)
    a = _pairwise_accelerations_from_speeds(swipe, v)
    return _percentile(a, p)

def f20_median_velocity_last3(swipe: List[FingerEvent]) -> float:
    """
    F20: Median of the last three instantaneous speeds.
    """
    v = _pairwise_speeds(swipe)
    if not v:
        return 0.0
    tail = v[-3:] if len(v) >= 3 else v
    return float(np.median(np.asarray(tail, dtype=float)))

def f21_largest_signed_deviation(swipe: List[FingerEvent]) -> float:
    """
    F21: Signed deviation value at the point of maximum absolute deviation from end-to-end line.
    """
    devs = _signed_deviations_from_line(swipe)
    if devs.size == 0:
        return 0.0
    return np.max(np.abs(devs))
    # why copilot write this way? The example code calculates the 
    # signed value at max abs dev, not the absolute value. This
    # is the same as the claim in the TouchAlytics paper, 
    # which states "largest **absolute** perpendicular distance between 
    # the end-to-end connection and the trajectory" BUT We project each 
    # vector on a perpendicular vector with a **defined direction** to 
    # distinguish if the largest deviation is on the left side or 
    # the right side of the end-to-end connection. This might be an
    #  indicator on whether the user is left-handed or right-handed.
    idx = int(np.argmax(np.abs(devs)))
    return float(devs[idx])

def f22_24_deviation_percentile(swipe: List[FingerEvent], p: float) -> float:
    """
    F22/F23/F24: Percentiles (p = 20/50/80) of signed deviations to end-to-end line.
    """
    devs = _signed_deviations_from_line(swipe).tolist()
    return _percentile(devs, p)

def f25_circular_mean_direction(swipe: List[FingerEvent]) -> float:
    """
    F25: Circular mean of segment directions (radians).
    """
    return _circular_mean(_segment_directions(swipe))

def f27_ratio_end_to_length(swipe: List[FingerEvent]) -> float:
    """
    F27: Ratio of end-to-end distance to trajectory length.
    """
    length = length_of_swipe(swipe)
    if length == 0:
        return 0.0
    disp = euclidean_distance(swipe[0].x, swipe[0].y, swipe[-1].x, swipe[-1].y)
    return float(disp / length)

def f29_median_initial_acc_first5pnt(swipe: List[FingerEvent]) -> float:
    """
    F29: Median acceleration over the initial window."""
#   WARNING: Adapted from "first 5 points" to "first 5% of points" as requested.
#   Computed from instantaneous speeds: a = dv/dt, take first ceil(5% * (#segments - 1)) samples.
    
    v = _pairwise_speeds(swipe)
    a = _pairwise_accelerations_from_speeds(swipe, v)
    if not a:
        return 0.0
    k = 5 # max(1, int(np.ceil(0.05 * len(a))))
    return float(np.median(np.asarray(a[:k], dtype=float)))




























































# Feature Engineering
def extract_features(swipe: List[FingerEvent]) -> Dict[str, float]:
    X: Dict[str, float] = {}
    """
    vx = get_pairwise_velocities_X(swipe)
    vy = get_pairwise_velocities_Y(swipe)
    ax = get_pairwise_accelerations(swipe, vx)
    ay = get_pairwise_accelerations(swipe, vy)
    deviations = get_deviations(swipe)
    """
    # Fix this line at line .*03
    X['duration'] = swipe[-1].timestamp_us - swipe[0].timestamp_us #i
    X['startX'] = swipe[0].x #i
    X['startY'] = swipe[0].y #i
    X['endX'] = swipe[-1].x #i
    X['endY'] = swipe[-1].y #i
    X['displacement'] = euclidean_distance(swipe[0].x, swipe[0].y, swipe[-1].x, swipe[-1].y) #d
    X['meanResultantLength'] = mean_resultant_length(swipe)  # F10
    # F11 UDLR flag not implemented
    X['direction'] = get_direction(swipe[0].x, swipe[0].y, swipe[-1].x, swipe[-1].y) #d
    # F13 phone id not implemented
    X['v20'] = velocity_percentile(swipe, 20.0)  # F14 # F14–F16: 20/50/80% percentiles of instantaneous speed magnitudes
    X['v50'] = velocity_percentile(swipe, 50.0)  # F15
    X['v80'] = velocity_percentile(swipe, 80.0)  # F16
    X['a20'] = acceleration_percentile(swipe, 20.0)  # F17 # F17–F19: 20/50/80% percentiles of signed accelerations (dv/dt)
    X['a50'] = acceleration_percentile(swipe, 50.0)  # F18
    X['a80'] = acceleration_percentile(swipe, 80.0)  # F19
    X['v_last3_median'] = f20_median_velocity_last3(swipe)  # F20: median of last 3 instantaneous speeds
    X['maxDevSigned'] = f21_largest_signed_deviation(swipe)  # F21: largest signed deviation from end-to-end line
    X['dev20'] = f22_24_deviation_percentile(swipe, 20.0)  # F22 # F22–F24: 20/50/80% percentiles of signed deviations
    X['dev50'] = f22_24_deviation_percentile(swipe, 50.0)  # F23
    X['dev80'] = f22_24_deviation_percentile(swipe, 80.0)  # F24   
    X['avgDirection'] = f25_circular_mean_direction(swipe)  # F25: circular mean of segment directions
    X['length'] = length_of_swipe(swipe) #i # F26: length of trajectory (already implemented below as 'length')
    X['ratio_end_to_length'] = f27_ratio_end_to_length(swipe)   # F27: ratio end-to-end / length
    X['speed'] = speed_of_swipe(swipe) #d # F28: average speed = length / duration (already implemented below as 'speed')
    X['acc_first5pct_median'] = f29_median_initial_acc_first5pnt(swipe)  # F29 (WARNING: 5% points) # F29: median initial acceleration over first 5% of segments



    """

    # GanTouch additional features
    X['velocity'] = magnitude_of_average_velocity_of_swipe(swipe) #d
    X['initial_velocity'] = get_magnitude_of_average_initial_velocity(swipe) #i
    X['final_velocity'] = get_magnitude_of_average_final_velocity(swipe) #i
    X['avg_velocity'] = get_average_speed(swipe) #i
    
    # X['area'] = area_of_swipe(swipe) #i
    X['acceleration'] = acceleration_of_swipe(swipe) #i
    X['avg_acceleration'] = get_average_acceleration(swipe) #i
    X['initial_acceleration'] = get_initial_acceleration(swipe) #i
    X['final_acceleration'] = get_final_acceleration(swipe) #i
#   X['acceleration_percentile_25'] = get_acceleration_percentile_25(swipe) #i
#   X['acceleration_percentile_50'] = get_acceleration_percentile_50(swipe) #i
#   X['acceleration_percentile_75'] = get_acceleration_percentile_75(swipe) #i
#   X['velocity_percentile_25'] = get_velocity_percentile_25(swipe) #i
#   X['velocity_percentile_50'] = get_velocity_percentile_50(swipe) #i
#   X['velocity_percentile_75'] = get_velocity_percentile_75(swipe) #i
    
    X['initial_speed'] = get_initial_speed(swipe) #i
    X['final_speed'] = get_final_speed(swipe) #i
    X['speed_percentile_25'] = get_speed_percentile_25(swipe) #i
    X['speed_percentile_50'] = get_speed_percentile_50(swipe) #i
    X['speed_percentile_75'] = get_speed_percentile_75(swipe) #i
    X['avg_vel_x'] = np.mean(vx)
    X['avg_vel_y'] = np.mean(vy)
    X['avg_acc_x'] = np.mean(ax)
    X['avg_acc_y'] = np.mean(ay)
    X['avg_devs'] = np.mean(deviations)
#   X['max_devs'] = np.max(deviations)
    # trying
    X['25%_vel_x'] = vx[len(vx)//4]
    X['50%_vel_x'] = vx[len(vx)//2]
    X['75%_vel_x'] = vx[len(vx)*3//4]
    X['25%_vel_y'] = vy[len(vy)//4]
    X['50%_vel_y'] = vy[len(vy)//2]
    X['75%_vel_y'] = vy[len(vy)*3//4]
    X['25%_acc_x'] = ax[len(ax)//4]
    X['50%_acc_x'] = ax[len(ax)//2]
    X['75%_acc_x'] = ax[len(ax)*3//4]
    X['25%_acc_y'] = ay[len(ay)//4]
    X['50%_acc_y'] = ay[len(ay)//2]
    X['75%_acc_y'] = ay[len(ay)*3//4]
    # X['median_area'] = median_area_of_swipe(swipe)
    """
    # final_X: List[float] = []
    # for i in sorted(list(X.keys())):
    #     final_X.append(X[i])
    # X['swipe_count'] = number_of_swipes(df)
    # If quadrant feature makes sense ?
    # print(sorted(list(X.keys())))
    # print(final_X)
    return X

def extract_touches(duplicate_end_swipe_generator: List[List[FingerEvent]], count: int) -> Tuple[List[Dict[str, float]], int]:
    swipes = []
    swipes: List[Dict[str, float]]
    for duplicate_end_swipe in duplicate_end_swipe_generator:
        swipe = duplicate_end_swipe[:-1]
        if len(swipe) > 5:
            swipes.append(extract_features(swipe))
        else:
            count += 1
    return swipes, count

if __name__ == "__main__" and False:
    count = 0
    result = []
    for i in range(1, 117):
        print (i, count)
        df = pd.read_csv('Tablet/User{}.csv'.format(i))
        f, count = extract_touches(df, count)
        result.append(len(f))
    print (count)
