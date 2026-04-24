# The goal is to provide a class that can yield swipes according to the (x1, y1) -> (x2, y2) requirement.

from analysis.lib.motionevent_classes import FingerEvent, SingularActionType
from analysis.lib.feature_library import euclidean_distance, startX, startY, endX, endY, extend_PhysicallyCorrectSingleSwipeType_to_SingularActionType, directionless_displacement, get_direction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast
import numpy as np
import pickle
import numpy.typing as npt
from scipy.interpolate import splrep, splev
from pathlib import Path

def tap(x: int, y: int, start_us: int, duration_us: int) -> SingularActionType:
    """
        a primitive tap that tries to align with adb_wrapper.
    """
    return [
        FingerEvent(timestamp_us=start_us, x=x, y=y),
        FingerEvent(timestamp_us=start_us + duration_us, x=x, y=y)
    ]


def bot_line_fit(x1: int, y1: int, x2: int, y2: int, duration_us: int, neighbor_time_delta_us: float, end_upfinger_time_us: int) -> SingularActionType:
    """
        TODO compare with app logs to see what really happens for an original bot swipe. 
    """
    
    trace: List[FingerEvent] = []
    steps = int(duration_us / neighbor_time_delta_us)  # convert ms to us
    for i in range(steps + 1):
        t = int(i * neighbor_time_delta_us)
        x = int(x1 + (x2 - x1) * i / steps)
        y = int(y1 + (y2 - y1) * i / steps)
        trace.append(FingerEvent(timestamp_us=t, x=x, y=y))

    # TODO what really is this time us?
    last_time_us = trace[-1].timestamp_us
    trace.append(FingerEvent(timestamp_us=last_time_us + end_upfinger_time_us, x=x2, y=y2))

    return trace

TBD_END_UPFINGER_TIME_US = 11000 # this value, under the current throwaway implementation, should not affect the final result. Note that it is different from adb wrapper, which use 50000 liftup.
def raw_faker(event_list: SingularActionType) -> SingularActionType:
    return bot_line_fit(x1=startX(event_list), y1=startY(event_list), x2=endX(event_list), y2=endY(event_list), duration_us=500 * 1000, neighbor_time_delta_us=11000, end_upfinger_time_us=TBD_END_UPFINGER_TIME_US)

def extract_exact_swipe_batch(label: str, swipe_file_generator: List[Tuple[str, List[SingularActionType]]]) -> List[SingularActionType]:
    # actually somewhat like draw_motion_event_multi_file.chain_gesture_iterators
    swipe_batches: List[SingularActionType] = []
    for file_label, gesture_generator in swipe_file_generator:
        if file_label == label:
            swipe_batches.extend(gesture_generator)
    return swipe_batches

def calculated_distance_to_required(swipe_batches: List[SingularActionType], x1: int, y1: int, x2: int, y2: int) -> Tuple[List[float], List[SingularActionType]]:
    fitted_batches: List[SingularActionType] = []
    distances: List[float] = []
    for swipe in swipe_batches:
        target_displacement_vector = (x2 - x1, y2 - y1)
        current_displacement_vector = (endX(swipe) - startX(swipe), endY(swipe) - startY(swipe))
        current_distance = euclidean_distance(*current_displacement_vector, *target_displacement_vector)
        distances.append(current_distance)
        fitted_batches.append(swipe)
    return distances, fitted_batches

def sample_from_softmax_inv_distance(distances: List[float], fitted_batches: List[SingularActionType], softmax_temperature: float, random_generator: Optional[np.random.Generator] = None) -> SingularActionType:
    """
        softmax_temperature between (0, +inf)
    """
    distances_np = np.array(distances)
    distances_np -= np.min(distances_np) # not max because inversed later
    exp_inv_distances = np.exp(- distances_np / softmax_temperature)
    probabilities = exp_inv_distances / np.sum(exp_inv_distances)

    if random_generator is None:
        random_generator = np.random.default_rng()

    sampled_indices = random_generator.choice(len(fitted_batches), size=1, p=probabilities).item()
    return fitted_batches[sampled_indices]

def drag_and_fit(x1: int, y1: int, x2: int, y2: int, original_swipe: SingularActionType) -> SingularActionType:
    # use scale and rotation transformation to transform original swipe to fit final swipe

    target_complex = complex(x2 - x1, y2 - y1)
    original_complex = complex(original_swipe[-1].x - original_swipe[0].x, original_swipe[-1].y - original_swipe[0].y)
    rotary_transformation = target_complex / original_complex

    # Apply the transformation
    transformed_swipe: List[FingerEvent] = []
    for event in original_swipe:
        original_now_offset = complex(event.x - original_swipe[0].x, event.y - original_swipe[0].y)
        transformed_offset = original_now_offset * rotary_transformation
        new_x = x1 + transformed_offset.real
        new_y = y1 + transformed_offset.imag
        transformed_swipe.append(FingerEvent(timestamp_us=event.timestamp_us, x=int(new_x), y=int(new_y)))

    return transformed_swipe

def drag_to_start(x1: int, y1: int, original_swipe: SingularActionType) -> SingularActionType:
    # only translate the original swipe to fit the starting point, without scaling and rotation, which is a simpler version of drag_and_fit. 
    transformed_swipe: List[FingerEvent] = []
    delta_x = x1 - startX(original_swipe)
    delta_y = y1 - startY(original_swipe)

    for event in original_swipe:
        new_x = event.x + delta_x
        new_y = event.y + delta_y
        transformed_swipe.append(FingerEvent(timestamp_us=event.timestamp_us, x=new_x, y=new_y))

    return transformed_swipe

class FitEffortProvider:
    def __init__(self, swipe_batches: List[SingularActionType], temperature: float = 100.0, random_state: Optional[int] = None) -> None:
        self.swipe_batches = swipe_batches
        self.temperature = temperature
        if random_state is not None:
            self.numpy_compatible_rng = np.random.default_rng(seed=random_state)
        else:
            self.numpy_compatible_rng = None

    def dump_batches(self, name: str = "swipe_data.pkl") -> None:
        raise NotImplementedError("Previously PhysicallyCorrectSingleSwipeType was generated and pickled in a separate script. After the recent refactor, we should pickle SingularActionType instead, which contain  and can be transformed to PhysicallyCorrectSingleSwipeType on the fly if needed. ")
        with open(name, "wb") as f:
            pickle.dump(self.swipe_batches, f)

    def fit(self, x1: int, y1: int, x2: int, y2: int) -> SingularActionType:
        """sample and fit a good swipe"""
        distances, obtained_batches = calculated_distance_to_required(self.swipe_batches, x1, y1, x2, y2)
        sampled_swipe = sample_from_softmax_inv_distance(distances, obtained_batches, softmax_temperature=self.temperature, random_generator=self.numpy_compatible_rng)
        fitted_swipe = drag_and_fit(x1, y1, x2, y2, sampled_swipe)
        return fitted_swipe
    
    def fit_only_start(self, x1: int, y1: int, x2: int, y2: int) -> SingularActionType:
        """only fit the starting point, which is a simpler version of fit, and can be used when we only care about the starting point, or when the original swipe is already good enough in terms of displacement vector but just has a wrong starting point."""
        distances, obtained_batches = calculated_distance_to_required(self.swipe_batches, x1, y1, x2, y2)
        sampled_swipe = sample_from_softmax_inv_distance(distances, obtained_batches, softmax_temperature=self.temperature, random_generator=self.numpy_compatible_rng)
        fitted_swipe = drag_to_start(x1, y1, sampled_swipe)
        return fitted_swipe

    def humanity_disturbance(self, original_swipe: SingularActionType) -> SingularActionType:
        """add some humanity disturbance to the original swipe(having no endpoints), which only takes into account its starting (x, y) and ending (x, y)"""
        x1, y1 = startX(original_swipe), startY(original_swipe)
        x2, y2 = endX(original_swipe), endY(original_swipe)
        return self.fit(x1, y1, x2, y2)


def b_spline_faker(trace: SingularActionType, neighbor_time_delta_us: float) -> SingularActionType:

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
    noise_t = np.asarray(splev(np.arange(n), tck_t), dtype=np.float64)
    noise_x = np.asarray(splev(np.arange(n), tck_x), dtype=np.float64)
    noise_y = np.asarray(splev(np.arange(n), tck_y), dtype=np.float64)

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

    new_trace: List[FingerEvent] = []
    for t, x, y in zip(new_t, new_x, new_y):
        new_trace.append(FingerEvent(timestamp_us=int(t), x=int(x), y=int(y)))

    new_trace_complete = extend_PhysicallyCorrectSingleSwipeType_to_SingularActionType(new_trace) # ensure the trace is still a valid SingularActionType after modification, and can be transformed to PhysicallyCorrectSingleSwipeType on the fly if needed. This is a safety check and may raise an error if the noise addition causes some issues.

    return new_trace_complete