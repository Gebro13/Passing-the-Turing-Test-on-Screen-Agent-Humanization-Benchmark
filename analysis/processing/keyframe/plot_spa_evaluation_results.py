from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from analysis.lib.motionevent_classes import SingularActionType
from analysis.lib.keyevent_classes import MotionKeyEvent, IMEEvent, IMESpecialKeyType

from analysis.lib.gesture_log_reader_utils import file_finder, rectify_timestamp_idx, session_noexcept_handling, limited_file_reader_yield, file_reader_yield, gesture_log_schema
from analysis.lib.sensor_log_reader_utils import parse_as_df, sensor_recording_name_schema
from analysis.lib.key_reader_utils import keys_generator_from_motionevent, IME_generator_from_IME_recording, IME_event_name_schema
import numpy as np
from pathlib import Path    
import re
import subprocess
# import cv2
# set up logging
import logging
import base64
from argparse import ArgumentParser
from agent_tools.fake_adb.adb_wrapper import MotionGenerator
from concurrent.futures import ThreadPoolExecutor


def correctness_parser(correctness: str) -> Optional[bool]:
    if correctness == "S":
        return True
    elif correctness == "F":
        return False
    elif correctness == "N" or correctness == "E":
        return None
    else:
        raise ValueError(f"Unexpected correctness value: {correctness}")




if __name__ == "__main__":

    parser = ArgumentParser(description="Extract frames from videos based on gesture events and sensor events, and save them in a structured way for SPA evaluation.")
    
    # add arg for result csv path; this must be given
    parser.add_argument("--result_csv_path", type=str, required=True, help="Path to the result CSV file containing timestamps and metadata.") 
    parser.add_argument("--formatted_data_path", type=str, required=True, help="Path to the formatted data Excel file containing task and timestamp information.")
    args = parser.parse_args()

    correctness_result_csv_path: Path = Path(args.result_csv_path)
    formatted_data_path: Path = Path(args.formatted_data_path)

    # columns: task_identifier,task_app,task_language,task_description,task_difficulty,golden_steps,key_component_final,adb_app,adb_home_page,is_cross_app,direct_no_action_evaluation,direct_no_action_details
    # use task_identifier as index
    correctness_result_csv = pd.read_csv(correctness_result_csv_path, header=0, index_col="task_identifier", dtype=str)


    formated_data_timestamps = pd.read_excel(formatted_data_path, header=0, index_col=None, dtype=str)
    TASK_COUNT = 116

    columns_list = list(formated_data_timestamps.columns)

    from operator import itemgetter



    humans: List[str] = list(itemgetter(2, 3, 4, 33, 34, 35, 36)(columns_list))
    humanized_agents: List[str] = list(itemgetter(5, 7, 9, 20, 39)(columns_list))
    non_humanized_agents: List[str] = list(itemgetter(12, 14, 16, 19, 37)(columns_list))
    rot_humanized_agents: List[str] = list(itemgetter(23, 25)(columns_list))
    fake_rot_tap_humanized_agents: List[str] = list(itemgetter(28, 30, 41)(columns_list))

    agent_list = {
        # "human": humans,
        "Raw": non_humanized_agents,
        "Only Swipe Humanized": rot_humanized_agents,
        "Rot Tap Humanized": humanized_agents,
        "Fake Rot Tap Humanized": fake_rot_tap_humanized_agents,
    }

    agents_name_list = [name for type, names in agent_list.items() for name in names]
    print(f"All agents: {agents_name_list}")

    print(f"All agents: {agents_name_list}")

    def get_agent_type(agent_name: str) -> str:
        for agent_type, agent_names in agent_list.items():
            if agent_name in agent_names:
                return agent_type
        raise ValueError(f"Agent name '{agent_name}' not found in any agent type list.")


    task_cluster_by_name = {
        "social_media": ['今日头条', '微博', '小红书', '知乎'],
        "shopping": ['jd', '淘宝', '菜鸟裹裹', '美团', 'eleme'],
        "video_streaming": ['iqiyi', '哔哩哔哩', 'qqmusic'],
        "trip_planning": ['ctrip', '高德地图', '航旅纵横', '去哪儿'],
        "office_learning": ['腾讯文档', '腾讯会议', '有道词典', '好大夫'],
    }

    # check that each task is classified
    all_mentioned_tasks: Set[str] = set()
    for task_list in task_cluster_by_name.values():
        all_mentioned_tasks.update(task_list)
    all_given_tasks = set(formated_data_timestamps['app'].unique())
    assert all_mentioned_tasks == all_given_tasks, f"Some tasks are not classified: {all_given_tasks - all_mentioned_tasks}"


    task_clusters: Dict[str, List[int]] = {
        key: list(formated_data_timestamps.index[formated_data_timestamps["app"].isin(lister)]) for key, lister in task_cluster_by_name.items()
    }

    task_statistics: Dict[str, Dict[str, Dict[str, int]]] = {
        agent_type: {
            task_cluster_name: {
                "task_correctness_count": 0,
                "task_total_count": 0
            } for task_cluster_name in task_cluster_by_name.keys()
        } for agent_type in agent_list.keys()
    }

    timestamp_list = rectify_timestamp_idx(
        formatted_data_timestamps_df=formated_data_timestamps,
        participants=agents_name_list
    )

    for participant, task_tuples in timestamp_list.items():
        agent_type = get_agent_type(participant)

        for task_idx, timestamp in task_tuples:

            task_name = formated_data_timestamps.loc[task_idx, "app"]
            
            if timestamp not in correctness_result_csv.index:
                print(f"Timestamp {timestamp} for task {task_name} not found in correctness result CSV. Skipping.")
                continue
            correctness: str = correctness_result_csv.loc[timestamp, "direct_no_action_evaluation"]
            correct = correctness_parser(correctness)
            if correct is None:
                print(f"Timestamp {timestamp} for task {task_name} has correctness value '{correctness}', which indicates no-action or error. Skipping correctness counting for this task.")
                continue

            for cluster_name, cluster_task_indices in task_clusters.items():
                if task_idx in cluster_task_indices:
                    task_statistics[agent_type][cluster_name]["task_total_count"] += 1
                    if correct:
                        task_statistics[agent_type][cluster_name]["task_correctness_count"] += 1
                    break
    


    """



    import matplotlib.pyplot as plt

    cluster_names = list(task_cluster_by_name.keys())
    cluster_plot_data = []



    for i, indices in enumerate(task_clusters):
        name = cluster_names[i]
        for label, col in [("Raw", RAW_correctness),  ("Only Swipe Humanized", only_swipe_correctness), ("Rot Tap Humanized", online_humanized_correctness), ("Fake Rot Tap", fake_rot_tap_correctness)]:
            success_sum, total_sum = 0, 0
            for idx in indices:
                res = _parse_ratio_to_tuple(formated_data_timestamps.iloc[idx][col])
                if res:
                    success_sum += res[0]
                    total_sum += res[1]
            if total_sum > 0:
                cluster_plot_data.append({"Cluster": name, "Type": label, "Success Ratio": success_sum / total_sum, "idx": i})



    """
    cluster_plot_data = []

    for agent_type, stats in task_statistics.items():
        for cluster_name, cluster_stats in stats.items():
            if cluster_stats["task_total_count"] > 0:
                success_ratio = cluster_stats["task_correctness_count"] / cluster_stats["task_total_count"]
                cluster_plot_data.append({"Cluster": cluster_name, "Type": agent_type, "Success Ratio": success_ratio})

    import seaborn as sns
    import matplotlib.pyplot as plt
    df_cluster_stats = pd.DataFrame(cluster_plot_data)
    plt.figure(figsize=(5, 4))
    # choose a classy palette
    sns.barplot(data=df_cluster_stats, x="Cluster", y="Success Ratio", hue="Type", palette="Greens")
    plt.ylim(0, 1)
    plt.title("Success Ratio by Task Cluster: Online vs Raw")
    plt.ylabel("Success Ratio")
    # rotate xlabel by 45 degrees and align right
    plt.xticks(rotation=13, ha='right')

    # make xlabel disappear
    plt.xlabel("")

    # make legend box smaller
    plt.legend(title="Participant Type", bbox_to_anchor=(0.60, 1), loc='upper left', fontsize=7)

    # draw some horizontal lines extending from the ytick labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig("success_ratio_by_cluster.pdf", format="pdf", bbox_inches='tight')
    plt.show()





