# classify: social media, shopping, video streaming, trip planning, office&learning, others



# %% [markdown]
# # Evaluate a swipe humanization method
# Run this file without arguments to evaluate `method_asdf_on_swipe` from `lib1.thingy2`.
# Output is formatted for an LLM agent to quickly understand the humanization status.

# %% setup logging
from argparse import ArgumentParser
import logging
import os
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "ERROR").upper(), logging.ERROR),
    format="%(levelname)s [%(name)s] %(message)s",
)

# %% read data and set up generators
import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict
import pickle
import sys

from sklearn.model_selection import train_test_split

from analysis.lib.motionevent_classes import SingularActionType, FingerEvent, SessionType, SwipeFeaturedSessionType
from analysis.lib.gesture_log_reader_utils import ranged_modified_generator_without_session_timestamp, ranged_modified_generator_with_session_timestamp
from analysis.processing.extract_feature_of_swipes import keep_swipe, build_features_dataframe_for_sessions
from analysis.processing.calculate_roc_auc_from_feature import (
    filter_df,
    compute_auc_per_feature,
    compute_acc_per_feature_using_x_1_x_point_on_roc_auc,
    calculate_svm_and_xgboost,
    calculate_mutual_information_binary,
    get_numeric_feature_column_names,
    get_feature_columns,
    classify_using_arima,
    ARIMAClassificationResult,
)
from data_collection.automations import PROJ_FOLDER_ABSOLUTE, generate_timestamp
from analysis.lib.feature_library import FEATURE_LENGTH

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate the effectiveness of a swipe humanization method using ARIMA classification.")
    parser.add_argument("--formated_data_path", type=str, required=True, help="Path to the formatted data Excel file containing swipe sessions and timestamps.")
    args = parser.parse_args()

    # %%
    
    formatted_data_path: Path = Path(args.formated_data_path)
    formated_data_timestamps = pd.read_excel(formatted_data_path, header=0, index_col=None, dtype=str)
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

    swipe_generator_for_sessions: Callable[[List[str], Optional[Callable[[SingularActionType], SingularActionType]]], List[SessionType]] = partial(
        ranged_modified_generator_with_session_timestamp,
        formated_data_timestamps=formated_data_timestamps,
        filtering_and_modification_function=keep_swipe,
        index_range=None
    )

    # %%
    from analysis.processing.fit_effort_provider import FitEffortProvider, b_spline_faker, raw_faker
    from agent_tools.fake_adb.adb_wrapper import load_PhysicallyCorrectSingleSwipeType_pickle, GLOBAL_EVENT_INTERVAL_US

    pkl_path = PROJ_FOLDER_ABSOLUTE / "analysis" / "processing" / "swipe_data.pkl"
    pickled: List[SingularActionType] = load_PhysicallyCorrectSingleSwipeType_pickle(pkl_path)
    fitter = FitEffortProvider(pickled)

    b_spline_faker_wrapped: Callable[[SingularActionType], SingularActionType] = partial(b_spline_faker, neighbor_time_delta_us=GLOBAL_EVENT_INTERVAL_US)


    # %%


    def filtering_nonzero_sessions(sessions: List[SessionType]) -> List[SessionType]:
        filtered_sessions: List[SessionType] = []
        for session in sessions:
            session_id, swipe_iterator = session
            if len(swipe_iterator) > 0:
                filtered_sessions.append(session)
        return filtered_sessions

    # %%

    human_sessions = filtering_nonzero_sessions(swipe_generator_for_sessions(participants=humans, humanity_disturbance=None))

    # sample 20% of human sessions for testing, and the rest for training the humanization method if needed
    human_sessions_train, human_sessions_test = train_test_split(human_sessions, test_size=0.2, random_state=42)

    human_sessions = human_sessions_test



    human_session_features: List[SwipeFeaturedSessionType] = [build_features_dataframe_for_sessions(session) for session in human_sessions]

    generators_in_comparison: Dict[str, List[SessionType]] = {
        # "non_humanized_agents": filtering_nonzero_sessions(swipe_generator_for_sessions(participants=non_humanized_agents, humanity_disturbance=None)),
        # "rot_tap_humanized_agents": filtering_nonzero_sessions(swipe_generator_for_sessions(participants=rot_tap_humanized_agents, humanity_disturbance=None)),
        "rot_humanized_agents": filtering_nonzero_sessions(swipe_generator_for_sessions(participants=rot_humanized_agents, humanity_disturbance=None)),
        # "fake_rot_tap_humanized_agents": filtering_nonzero_sessions(swipe_generator_for_sessions(participants=fake_rot_tap_humanized_agents, humanity_disturbance=None)),
        # "offline_rot_tap_humanized_agents": filtering_nonzero_sessions(swipe_generator_for_sessions(participants=non_humanized_agents, humanity_disturbance=fitter.humanity_disturbance)),
        # "offline_b_spline_humanized_agents": filtering_nonzero_sessions(swipe_generator_for_sessions(participants=non_humanized_agents, humanity_disturbance=b_spline_faker_wrapped)),
        # "offline_unhumanized_rot_tap_agent": filtering_nonzero_sessions(swipe_generator_for_sessions(participants=rot_tap_humanized_agents, humanity_disturbance=raw_faker)),
    }


    # %%

    results: Dict[str, ARIMAClassificationResult] = {}

    for method_name, session_generator in generators_in_comparison.items():
        print(f"Evaluating method: {method_name}")
        featured_sessions = [build_features_dataframe_for_sessions(session) for session in session_generator]
        
        total_sessions = human_session_features + featured_sessions
        total_labels = [False] * len(human_session_features) + [True] * len(featured_sessions) # False for human, True for agent

        result = classify_using_arima(FEATURE_LENGTH, total_sessions, total_labels)
        results[method_name] = result

    timestamp = generate_timestamp()

    # %% simply dump the pkl
    with open(f"arima_classification_results_{timestamp}.pkl", "wb") as f:
        pickle.dump(results, f)

    with open(f"arima_classification_results_{timestamp}.pkl", "rb") as f:
        loaded_results = pickle.load(f)

    # %% print the results as a dataframe

    class ResultPrintingType(TypedDict):
        method_name: str
        acc: float
        TPR: float
        FPR: float

    aggregated_results_for_printing: List[ResultPrintingType] = []

    for method_name, result in results.items():
        printer_obj = ResultPrintingType(
            method_name=method_name,
            acc=result.metrics.accuracy(),
            TPR=result.metrics.recall_or_TPR(),
            FPR=result.metrics.FPR(),
        )
        aggregated_results_for_printing.append(printer_obj)
    df_results = pd.DataFrame(aggregated_results_for_printing)
    print(df_results)
    df_results.to_csv(f"arima_classification_results_{timestamp}.csv", index=False)