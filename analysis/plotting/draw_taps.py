

# %% [markdown]
# Examine whether the swipe/tap distinguisher is reasonable or not.

# %% setup logging
import logging
import os
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "ERROR").upper(), logging.ERROR),
    format="%(levelname)s [%(name)s] %(message)s",
)


# %%


import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict
import pickle
import sys

from sklearn.model_selection import train_test_split
import sklearn.preprocessing

# %%

SELF_FOLDER = Path(__file__).resolve().parent
PROJ_FOLDER = SELF_FOLDER.parent.parent
assert PROJ_FOLDER / "analysis" / "plotting" == SELF_FOLDER, "The script's location has changed, please update the path handling code accordingly."

sys.path.append(str(PROJ_FOLDER)) # very necessary since we may call this file from other directories


# %% read data and set up generators


from analysis.lib.motionevent_classes import SingularActionType, FingerEvent, SessionType, SwipeFeaturedSessionType
from analysis.lib.gesture_log_reader_utils import ranged_modified_generator_without_session_timestamp
from analysis.processing.extract_feature_of_swipes import keep_swipe, build_features_dataframe
from analysis.processing.calculate_roc_auc_from_feature import (
    filter_df,
    compute_auc_per_feature,
    compute_acc_per_feature_using_x_1_x_point_on_roc_auc,
    calculate_svm_and_xgboost,
    calculate_mutual_information_binary,
    get_numeric_feature_column_names,
    get_feature_columns,
    classify_using_lstm,
    LSTMClassificationResult,
)
from data_collection.automations import PROJ_FOLDER_ABSOLUTE, generate_timestamp
from analysis.lib.feature_library import FEATURE_LENGTH, is_tap, swipe_judger_v2

# %%

# get the FORMAT_DATA_TIMESTAMP_XLSX_PATH from environment variable; if not set, raise an error
FORMATED_DATA_TIMESTAMP_XLSX_PATH = os.environ.get("FORMATED_DATA_TIMESTAMP_XLSX_PATH")
if FORMATED_DATA_TIMESTAMP_XLSX_PATH is None:
    raise ValueError("Please set the environment variable FORMATED_DATA_TIMESTAMP_XLSX_PATH to the path of the formatted data Excel file containing swipe sessions and timestamps.")
FORMATED_DATA_TIMESTAMP_XLSX_PATH = Path(FORMATED_DATA_TIMESTAMP_XLSX_PATH)
# %%
formated_data_timestamps = pd.read_excel(FORMATED_DATA_TIMESTAMP_XLSX_PATH, header=0, index_col=None, dtype=str)
TASK_COUNT = 116

columns_list = list(formated_data_timestamps.columns)

from operator import itemgetter


from typing import List

# %%
"""
task_cluster_by_name = {
    "social_media": ['今日头条', '微博', '小红书', '知乎'],
    "shopping": ['jd', '淘宝', '菜鸟裹裹', '美团', 'eleme'],
    "video_streaming": ['iqiyi', '哔哩哔哩', 'qqmusic'],
    "trip_planning": ['ctrip', '高德地图', '航旅纵横', '去哪儿'],
    "office_learning": ['腾讯文档', '腾讯会议', '有道词典', '好大夫'],
}

# check that each task is classified
all_mentioned_tasks = set()
for task_list in task_cluster_by_name.values():
    all_mentioned_tasks.update(task_list)
all_given_tasks = set(formated_data_timestamps['app'].unique())
assert all_mentioned_tasks == all_given_tasks, f"Some tasks are not classified: {all_given_tasks - all_mentioned_tasks}"

task_cluster_idx_by_name: Dict[str, List[int]] = {
    key: list(formated_data_timestamps.index[formated_data_timestamps["app"].isin(task_list)]) for key, task_list in task_cluster_by_name.items()
}

task_clusters: List[List[int]] = [
    list(formated_data_timestamps.index[formated_data_timestamps["app"].isin(lister)]) for key, lister in task_cluster_by_name.items()
]

# assert that task_cluster_idx_by_name and task_clusters are consistent
for key, idx_list in task_cluster_idx_by_name.items():
    assert idx_list == task_clusters[list(task_cluster_by_name.keys()).index(key)], f"Inconsistent cluster for {key}"
"""
# %%

humans: List[str] = list(itemgetter(2, 3, 4, 33, 34, 35, 36)(columns_list))
rot_tap_humanized_agents: List[str] = list(itemgetter(5, 7, 9, 20, 39)(columns_list))
non_humanized_agents: List[str] = list(itemgetter(12, 14, 16, 19, 37)(columns_list))
rot_humanized_agents: List[str] = list(itemgetter(23, 25)(columns_list))
fake_rot_tap_humanized_agents: List[str] = list(itemgetter(28, 30, 41)(columns_list))

# %%

single_gesture_generator: Callable[[List[str], Optional[Callable[[SingularActionType], SingularActionType]]], List[SingularActionType]] = partial(
    ranged_modified_generator_without_session_timestamp,
    formated_data_timestamps=formated_data_timestamps,
    filtering_and_modification_function=None,
    index_range=None
)

# %%


# %%


from analysis.processing.fit_effort_provider import FitEffortProvider
from agent_tools.fake_adb.adb_wrapper import load_PhysicallyCorrectSingleSwipeType_pickle

pkl_path = PROJ_FOLDER / "analysis" / "processing" / "swipe_data.pkl"
pickled: List[SingularActionType] = load_PhysicallyCorrectSingleSwipeType_pickle(pkl_path)
fitter = FitEffortProvider(pickled)

# %%

human_gestures = single_gesture_generator(participants=humans, humanity_disturbance=None)

from analysis.plotting.draw_motion_event2 import (
    throw_away_timestamp,
    plot_gestures
)

human_session_taps = [a_tap for a_tap in human_gestures if is_tap(a_tap)]

# plot_gestures(throw_away_timestamp(human_session_taps), have_legend=False)

is_tap_by_pixel = [is_tap(gesture) for gesture in human_gestures]
is_tap_by_length = [not swipe_judger_v2(gesture) for gesture in human_gestures]

# print a confusion matrix of these two binary classifications
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(is_tap_by_pixel, is_tap_by_length)
print("Confusion Matrix (rows: pixel-based, columns: length-based)")
print(cm)
print("\nClassification Report:")
print(classification_report(is_tap_by_pixel, is_tap_by_length, target_names=["Not Tap", "Tap"]))

# there are no gestures that are classified as taps by length but not by pixel.

# there are some gestures that are classified as taps by pixel but not by length, let's plot them to see what they look like
gestures_pixel_tap_only = [gesture for gesture, pixel_tap, length_tap in zip(human_gestures, is_tap_by_pixel, is_tap_by_length) if pixel_tap and not length_tap]

# there are 190 of them.
plot_gestures(throw_away_timestamp(gestures_pixel_tap_only), have_legend=False)

