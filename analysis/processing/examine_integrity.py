# %% read data and set up generators
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple


PROJ_PATH = Path(__file__).resolve().parent.parent.parent
assert PROJ_PATH / "analysis" / "processing" == Path(__file__).resolve().parent, "The script's location has changed, please update the path handling code accordingly."

import sys
sys.path.append(str(PROJ_PATH))


from analysis.lib.gesture_log_reader_utils import ranged_modified_generator_with_session_timestamp


# %%

if __name__ == "__main__":

    parser = ArgumentParser(description="Examine the integrity of the collected gesture event data.")
    parser.add_argument("--formated_data_path", type=str, required=True, help="Path to the formatted data Excel file containing swipe sessions and timestamps.")

    args = parser.parse_args()

    formatted_data_path: Path = Path(args.formated_data_path)

    formated_data_timestamps = pd.read_excel(formatted_data_path, header=0, index_col=None, dtype=str)
    TASK_COUNT = 116

    columns_list = list(formated_data_timestamps.columns)

    from operator import itemgetter

    humans: List[str] = list(itemgetter(2, 3, 4, 33, 34, 35, 36)(columns_list))
    humanized_agents: List[str] = list(itemgetter(5, 7, 9, 20, 39)(columns_list))
    non_humanized_agents: List[str] = list(itemgetter(12, 14, 16, 19, 37)(columns_list))
    rot_humanized_agents: List[str] = list(itemgetter(23, 25)(columns_list))
    fake_rot_tap_humanized_agents: List[str] = list(itemgetter(28, 30, 41)(columns_list))



    # %% 

    agent_raw_generator = ranged_modified_generator_with_session_timestamp(
        formated_data_timestamps=formated_data_timestamps,
        participants=humanized_agents + non_humanized_agents + rot_humanized_agents + fake_rot_tap_humanized_agents,
        filtering_and_modification_function=None,
    )

    print("agents done")

    human_raw_generator = ranged_modified_generator_with_session_timestamp(
        formated_data_timestamps=formated_data_timestamps,
        participants=humans,
        filtering_and_modification_function=None,
    )

    print(f"Collected {len(human_raw_generator)} sessions for humans.")