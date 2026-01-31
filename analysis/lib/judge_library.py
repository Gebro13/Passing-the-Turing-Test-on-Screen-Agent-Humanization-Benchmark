import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from scipy.stats import norm, kstest
from sklearn.metrics import mean_squared_error
from motionevent_classes import FingerEvent
from typing import Dict, Tuple, TypedDict, List

from gesture_log_reader_utils import SessionType
from tap_duration_extract import is_tap
from calculate_roc_auc_from_feature import get_numeric_feature_column_names, get_feature_columns, ThresholdPosterior
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from extract_feature_of_swipes import build_features_dataframe

def swipe_judger(single_finger_trace: List[FingerEvent]) -> bool:
    """
    Judge whether the trace is a tap or a swipe.
    """
    coords = np.array([(e.x, e.y) for e in single_finger_trace])
    start = coords[0]
    end = coords[-1]

    vec = end - start
    # compute total displacement between start and end
    norm = np.linalg.norm(vec)

    TAP_THRESHOLD = 30.0     # pixels

    # if there is almost no movement, it's a tap
    if norm < TAP_THRESHOLD:
        return False
    return True

def naive_agent_judger(single_finger_trace: list[FingerEvent]) -> bool:
    """
    Judge whether the trace is from an artificial agent.
    """
    # the current naive agent behaves:
    # 1. the swipe is always a straight line
    # 2. the tap is always at the same position)
    # judge whether the trace is a swipe or a tap thru beginning (x,y) and end(x,y), then judge accordingly.
    # extract the (x, y) coordinates from the trace
    coords = np.array([(e.x, e.y) for e in single_finger_trace])
    start = coords[0]
    end = coords[-1]

    vec = end - start
    # compute total displacement between start and end
    norm = np.linalg.norm(vec)

    TAP_THRESHOLD = 30.0     # pixels
    STRAIGHT_THRESHOLD = 2.0  # max perpendicular deviation in pixels

    # if there is almost no movement, it's a tap
    if norm < TAP_THRESHOLD:
        delta_t = single_finger_trace[-1].timestamp_us - single_finger_trace[0].timestamp_us
        # if the tap is too fast, it's likely an agent
        return delta_t < 10 * 1000  # 10ms threshold for tap speed

    # otherwise treat as swipe: check straightness
    # perpendicular distance from each point to the line start→end
    deltas = coords - start
    perp_dists = np.abs(vec[1] * deltas[:, 0] - vec[0] * deltas[:, 1]) / norm

    # if the path is nearly a straight line, flag as artificial
    return perp_dists.max() < STRAIGHT_THRESHOLD


def ds_is_human_swipe_against_b_spline(events: List[FingerEvent], velocity_threshold=0.8, jerk_threshold=500, curvature_std_threshold=0.2,
                   max_deviation_ratio=0.15, pause_velocity_ratio=0.1, start_end_velocity_ratio=1.8):
    """
    Determine if a swipe is human-generated or a B-spline approximation.
    A submodule to be used after judging that the swipe is human.
    
    Args:
        events: List of FingerEvent(t, x, y) with attributes .t, .x, .y.
        velocity_threshold: Threshold for sudden stop detection (default: 0.8).
        jerk_threshold: Threshold for jerk magnitude (default: 500 px/ms³).
        curvature_std_threshold: Threshold for curvature variance (default: 0.2).
        max_deviation_ratio: Max deviation from straight line (default: 0.15).
        pause_velocity_ratio: Velocity threshold for pause detection (default: 0.1).
        start_end_velocity_ratio: Start/end velocity asymmetry threshold (default: 1.8).
    
    Returns:
        bool: True if swipe is likely human, False if likely B-splined.
    """
    if len(events) < 10:
        return True  # Too short, assume human (or discard)
    
    # Convert events to numpy arrays
    t = np.array([e.timestamp_us for e in events])
    x = np.array([e.x for e in events])
    y = np.array([e.y for e in events])
    points = np.column_stack((x, y))
    
    # --- 1. Endpoint Behavior Analysis ---
    # Sudden Stop Detection (last 5% of points)
    last_5pct = max(1, len(points) // 20)
    vx = np.diff(x[-last_5pct:]) / np.diff(t[-last_5pct:])
    vy = np.diff(y[-last_5pct:]) / np.diff(t[-last_5pct:])
    velocities = np.sqrt(vx**2 + vy**2)
    if len(velocities) >= 3:
        velocity_drop = velocities[-3:].max() / velocities.mean()
        if velocity_drop < velocity_threshold:
            return True  # Human (sudden stop)
    
    # Lift-off Tremor (FFT on last 10 points)
    last_10 = min(10, len(points))
    if last_10 >= 4:
        displacements = np.sqrt(np.diff(x[-last_10:])**2 + np.diff(y[-last_10:])**2)
        fft = np.fft.fft(displacements - np.mean(displacements))
        freqs = np.fft.fftfreq(len(displacements), d=np.mean(np.diff(t[-last_10:])))
        dominant_freq = np.abs(freqs[np.argmax(np.abs(fft))])
        if dominant_freq > 15:  # High-frequency jitter
            return True
    
    # --- 2. Curvature & Smoothness Metrics ---
    # Single-Arc Consistency (Quadratic Bézier fit)
    def quadratic_bezier_residuals(points):
        n = len(points)
        t_vals = np.linspace(0, 1, n)
        A = np.column_stack([(1-t_vals)**2, 2*(1-t_vals)*t_vals, t_vals**2])
        bx = np.linalg.lstsq(A, points[:, 0], rcond=None)[0]
        by = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
        bezier_x = A @ bx
        bezier_y = A @ by
        residuals = np.sqrt((points[:, 0] - bezier_x)**2 + (points[:, 1] - bezier_y)**2)
        return residuals
    
    residuals = quadratic_bezier_residuals(points)
    if np.mean(residuals) < 0.1:  # Low residuals → follows a single arc
        return True
    
    # Local Curvature Variance
    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    curvature_std = np.std(curvature)
    if curvature_std > curvature_std_threshold:
        return False  # B-splined (high curvature variation)
    
    # --- 3. Velocity Profile Signatures ---
    # Bell-Shaped Velocity
    velocities = np.sqrt(dx**2 + dy**2)
    peak_idx = np.argmax(velocities)
    first_third = velocities[:len(velocities)//3].mean()
    last_third = velocities[-len(velocities)//3:].mean()
    if (velocities[peak_idx] > 1.5 * first_third and 
        velocities[peak_idx] > 1.5 * last_third):
        return True  # Human (bell-shaped)
    
    # Jerk Analysis
    jerk = np.sqrt(np.gradient(ddx, t)**2 + np.gradient(ddy, t)**2)
    if np.mean(jerk) > jerk_threshold:
        return False  # B-splined (high jerk)
    
    # --- 4. Path Geometry Tests ---
    # Deviation from Straight Line
    start_end_line = np.linspace(points[0], points[-1], len(points))
    perp_distances = np.abs(np.cross(points - start_end_line[0], 
                                    start_end_line[-1] - start_end_line[0])) / np.linalg.norm(start_end_line[-1] - start_end_line[0])
    max_deviation = np.max(perp_distances)
    swipe_length = np.linalg.norm(points[-1] - points[0])
    if max_deviation > swipe_length * max_deviation_ratio:
        return False  # B-splined (unnatural curvature)
    
    # Turning Angle Consistency
    displacement_vectors = np.diff(points, axis=0)
    angles = np.arctan2(displacement_vectors[1:, 1], displacement_vectors[1:, 0]) - \
             np.arctan2(displacement_vectors[:-1, 1], displacement_vectors[:-1, 0])
    angle_sign_changes = np.sum(np.diff(np.sign(angles)) != 0)
    if angle_sign_changes > 0.2 * len(angles):
        return False  # B-splined (oscillations)
    
    # --- 5. Temporal Dynamics ---
    # Event Timing Irregularity
    time_deltas = np.diff(t)
    if np.var(time_deltas) < 0.1:  # Machine-generated timestamps
        return False
    
    # Pause Detection
    peak_velocity = np.max(velocities)
    pause_threshold = peak_velocity * pause_velocity_ratio
    pauses = np.where(velocities < pause_threshold)[0]
    if len(pauses) > 3 and np.any(np.diff(pauses) == 1):  # Consecutive low-velocity points
        return True
    
    # --- 6. Composite Features ---
    # Start/End Asymmetry
    start_vel = velocities[:len(velocities)//3].mean()
    end_vel = velocities[-len(velocities)//3:].mean()
    if start_vel > start_end_velocity_ratio * end_vel:
        return True  # Human (asymmetric swipe)
    
    # B-spline Residual Analysis
    t_normalized = (t - t[0]) / (t[-1] - t[0])
    spline = make_interp_spline(t_normalized, points, k=3)
    spline_points = spline(t_normalized)
    spline_residuals = np.linalg.norm(points - spline_points, axis=1)
    _, p_value = kstest(spline_residuals, 'norm')
    if p_value > 0.05:  # Residuals follow normal distribution → B-splined
        return False
    
    # Default: Assume human if no strong B-spline indicators
    return True


def ds_enhanced_judger(event: List[FingerEvent]) -> bool:
    """
    Enhanced judging function for touch events.
    Judge whether the trace is from an artificial agent.
    This function first uses the naive agent judger, then applies a more complex B-spline analysis.
    """
    naive_result = naive_agent_judger(event)
    if not naive_result:
        # probably tricked
        return not ds_is_human_swipe_against_b_spline(event)
    return naive_result




def get_log_odds_with_threshold_posterior(
        value: float, 
        threshold_info: ThresholdPosterior
    ) -> float:
    """
    Given a value and threshold posterior info, return the log odds of being agent.

    :param value: Description
    """
    if value < threshold_info['threshold']:
        return threshold_info['lesser_than_threshold_agent_log_odds']
    else:
        return threshold_info['greater_than_threshold_agent_log_odds']

class UltimateClassifier():

    ResultType = List[Tuple[int, float]]  # List of (action_begin_or_end_time_us, cumulant_log_odds)

    def __init__(
            self, 
            classifiers_to_use_for_clusters: Dict[int, Tuple[Pipeline, XGBClassifier]], 
            tap_us_classifier_for_clusters: Dict[int, ThresholdPosterior],
            interval_us_classifier: Dict[int, ThresholdPosterior]
        ):
        """
        Assume that negative label stand for humans, positive label stand for agents. This is true throughout the project.
        
        :param self: Description
        :param classifiers_to_use_for_clusters: Description
        :type classifiers_to_use_for_clusters: Dict[int, Tuple[Pipeline, XGBClassifier]]
        :param tap_us_classifier_for_clusters: Description
        :type tap_us_classifier_for_clusters: Dict[int, ThresholdPosterior]
        :param interval_us_classifier: Description
        :type interval_us_classifier: Dict[int, ThresholdPosterior]
        """
    
        self.classifiers_to_use_for_clusters = classifiers_to_use_for_clusters
        self.tap_us_classifier = tap_us_classifier_for_clusters
        self.interval_us_classifier = interval_us_classifier

    def predict(self, X: SessionType, task_cluster_id: int, prior_log_odds: float = 0.0) -> ResultType:
        """
        For a session, for each existent action, predict with all its history whether it is from an agent or a human.
        
        :param X: Description
        :param task_cluster_id: Description
        :param prior_log_odds: log(P(agent) / P(human)) before seeing any action in this session.
        :return: A list of tuples, each tuple is (action_begin_or_end_time_us, cumulant_log_odds) where cumulant_log_odds is log(P(agent | X) / P(human | X)).
        """

        cumulant_log_odds: float = prior_log_odds
        

        session = X[1]
        
        if (len(session) == 0):
            return []
        
        last_end_time = session[0][0].timestamp_us # used when idx == 1

        results = []

        for idx, action in enumerate(session):

            if idx > 0:
                this_begin_time = action[0].timestamp_us
                
                # predict and update interval
                interval_log_odds = get_log_odds_with_threshold_posterior(
                    this_begin_time - last_end_time,
                    self.interval_us_classifier[task_cluster_id]
                )
                cumulant_log_odds += interval_log_odds

                results.append((this_begin_time, cumulant_log_odds))

            
            action_type = 'tap' if is_tap(action) else 'swipe'
            # get features
            if action_type == 'tap':
                duration_us = action[-1].timestamp_us - action[0].timestamp_us
                tap_log_odds = get_log_odds_with_threshold_posterior(
                    duration_us,
                    self.tap_us_classifier[task_cluster_id]
                )
                cumulant_log_odds += tap_log_odds
            elif action_type == 'swipe':
            
                swipe_single_df = build_features_dataframe([("temp", [action])])

                feature_names = get_numeric_feature_column_names(swipe_single_df)
                numeric_feature_columns = get_feature_columns(swipe_single_df, feature_names)

                classifier_to_use = self.classifiers_to_use_for_clusters[task_cluster_id][0] # use svm pipeline for log odds

                # predict_log_proba returns log probilities [n_samples, n_classes], class 0 is human, class 1 is agent
                swipe_log_probs_array = classifier_to_use.predict_log_proba(numeric_feature_columns)

                swipe_log_odds = swipe_log_probs_array[0][1] - swipe_log_probs_array[0][0]

                # print(f"swipe log odds: {swipe_log_odds}")
                # print(type(swipe_log_odds))
                cumulant_log_odds += swipe_log_odds
            else:
                raise ValueError(f"Unknown action type: {action_type}")
            
            this_end_time = action[-1].timestamp_us
            results.append((this_end_time, cumulant_log_odds))
            last_end_time = this_end_time

        return results
