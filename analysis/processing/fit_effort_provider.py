# The goal is to provide a class that can yield swipes according to the (x1, y1) -> (x2, y2) requirement.

from analysis.lib.motionevent_classes import FingerEvent
from typing import List, Tuple
def euclidean_distance(xy1: Tuple[float, float], xy2: Tuple[float, float]) -> float:
    return abs(complex(*xy1) - complex(*xy2))
import numpy as np
import pickle
import numpy.typing as npt
from scipy.interpolate import splrep, splev

def bot_line_fit(x1: int, y1: int, x2: int, y2: int, duration_us: int, neighbor_time_delta_us: float) -> List[FingerEvent]:
    trace: List[FingerEvent] = []
    steps = int(duration_us / neighbor_time_delta_us)  # convert ms to us
    for i in range(steps + 1):
        t = int(i * neighbor_time_delta_us)
        x = int(x1 + (x2 - x1) * i / steps)
        y = int(y1 + (y2 - y1) * i / steps)
        trace.append(FingerEvent(timestamp_us=t, x=x, y=y))
    return trace

def extract_exact_swipe_batch(label: str, swipe_file_generator: List[Tuple[str, List[List[FingerEvent]]]]) -> List[List[FingerEvent]]:
    # actually somewhat like draw_motion_event_multi_file.chain_gesture_iterators
    swipe_batches = []
    for file_label, gesture_generator in swipe_file_generator:
        if file_label == label:
            swipe_batches.extend(gesture_generator)
    return swipe_batches

def calculated_distance_to_required(swipe_batches: List[List[FingerEvent]], x1: int, y1: int, x2: int, y2: int) -> Tuple[List[float], List[List[FingerEvent]]]:
    fitted_batches: List[List[FingerEvent]] = []
    distances: List[float] = []
    for swipe in swipe_batches:
        target_displacement_vector = (x2 - x1, y2 - y1)
        current_displacement_vector = (swipe[-1].x - swipe[0].x, swipe[-1].y - swipe[0].y)
        current_distance = euclidean_distance(current_displacement_vector, target_displacement_vector)
        distances.append(current_distance)
        fitted_batches.append(swipe)
    return distances, fitted_batches

def sample_from_softmax_inv_distance(distances: List[float], fitted_batches: List[List[FingerEvent]], softmax_temperature: float) -> List[FingerEvent]:
    """
        softmax_temperature between (0, +inf)
    """
    distances_np = np.array(distances)
    distances_np -= np.min(distances_np) # not max because inversed later
    exp_inv_distances = np.exp(- distances_np / softmax_temperature)
    probabilities = exp_inv_distances / np.sum(exp_inv_distances)

    sampled_indices = np.random.choice(len(fitted_batches), size=1, p=probabilities).item()
    return fitted_batches[sampled_indices]

def drag_and_fit(x1: int, y1: int, x2: int, y2: int, original_swipe: List[FingerEvent]) -> List[FingerEvent]:
    # use scale and rotation transformation to transform original swipe to fit final swipe

    target_complex = complex(x2 - x1, y2 - y1)
    original_complex = complex(original_swipe[-1].x - original_swipe[0].x, original_swipe[-1].y - original_swipe[0].y)
    rotary_transformation = target_complex / original_complex

    # Apply the transformation
    transformed_swipe = []
    for event in original_swipe:
        original_now_offset = complex(event.x - original_swipe[0].x, event.y - original_swipe[0].y)
        transformed_offset = original_now_offset * rotary_transformation
        new_x = x1 + transformed_offset.real
        new_y = y1 + transformed_offset.imag
        transformed_swipe.append(FingerEvent(timestamp_us=event.timestamp_us, x=int(new_x), y=int(new_y)))

    return transformed_swipe

class FitEffortProvider:
    def __init__(self, swipe_batches: List[List[FingerEvent]]):
        self.swipe_batches = swipe_batches

    def dump_batches(self, name: str = "swipe_data.pkl") -> None:
        with open(name, "wb") as f:
            pickle.dump(self.swipe_batches, f)

    def fit(self, x1: int, y1: int, x2: int, y2: int) -> List[FingerEvent]:
        """sample and fit a good swipe"""
        distances, obtained_batches = calculated_distance_to_required(self.swipe_batches, x1, y1, x2, y2)
        sampled_swipe = sample_from_softmax_inv_distance(distances, obtained_batches, softmax_temperature=100.0)
        fitted_swipe = drag_and_fit(x1, y1, x2, y2, sampled_swipe)
        return fitted_swipe

    def humanity_disturbance(self, original_swipe: List[FingerEvent]) -> List[FingerEvent]:
        """add some humanity disturbance to the original swipe(having no endpoints), which only takes into account its starting (x, y) and ending (x, y)"""
        x1, y1 = original_swipe[0].x, original_swipe[0].y
        x2, y2 = original_swipe[-1].x, original_swipe[-1].y
        return self.fit(x1, y1, x2, y2)


def b_spline_faker(trace: List[FingerEvent], neighbor_time_delta_us: float) -> List[FingerEvent]:
    # Implement B-spline fitting and noise addition here
    # add b-spline noise to t, x, y
    # collect original arrays
    n = len(trace)
    t_arr: npt.NDArray[np.float64] = np.array([pt.timestamp_us for pt in trace], dtype=float)
    x_arr: npt.NDArray[np.float64] = np.array([pt.x for pt in trace], dtype=float)
    y_arr: npt.NDArray[np.float64] = np.array([pt.y for pt in trace], dtype=float)

    # generate white‐noise vectors
    rand_t: npt.NDArray[np.float64] = np.random.randn(n)
    rand_x: npt.NDArray[np.float64] = np.random.randn(n)
    rand_y: npt.NDArray[np.float64] = np.random.randn(n)

    # fit cubic B‐splines through the noise
    k = 3
    # choose a few interior knot locations
    num_knots = max(4, n // 4)
    knots = np.linspace(0, n - 1, num_knots)[1:-1]
    tck_t = splrep(np.arange(n), rand_t, k=k, t=knots)
    tck_x = splrep(np.arange(n), rand_x, k=k, t=knots)
    tck_y = splrep(np.arange(n), rand_y, k=k, t=knots)

    # evaluate smooth noise
    noise_t = splev(np.arange(n), tck_t)
    noise_x = splev(np.arange(n), tck_x)
    noise_y = splev(np.arange(n), tck_y)

    # scale factors for noise amplitude
    eps_t  = neighbor_time_delta_us * 0.2
    eps_xy = ((abs(trace[-1].x - trace[0].x) + abs(trace[-1].y - trace[0].y)) / 2) * 0.02 # 0.145 # tuned to avg_dev
    
    # apply noise
    new_t = t_arr + eps_t  * noise_t
    new_x = x_arr + eps_xy * noise_x
    new_y = y_arr + eps_xy * noise_y

    # ensure timestamps remain non‐decreasing
    new_t -= np.min(new_t)  # shift to start at zero
    new_t = np.maximum.accumulate(new_t)

    # write back into trace
    for i, pt in enumerate(trace):
        pt.timestamp_us = int(new_t[i])
        pt.x            = int(new_x[i])
        pt.y            = int(new_y[i])

    return trace