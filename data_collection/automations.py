import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple, Callable, TypedDict, Union
import PIL.Image
from pathlib import Path

import threading

# ...existing code...
import json
import random
import pandas as pd



COLLECTION_FOLDER_ABSOLUTE: Path = Path(__file__).resolve().parent
PROJ_FOLDER_ABSOLUTE: Path = COLLECTION_FOLDER_ABSOLUTE.parent
FAKE_ADB_PATH_ABSOLUTE: Path = PROJ_FOLDER_ABSOLUTE / "agent_tools" / "fake_adb"

"""
# assert the running folder is PROJ_FOLDER, otherwise throw error
if Path.cwd() != PROJ_FOLDER_ABSOLUTE:
    raise NotImplementedError("Please run this script from the project root folder.")
    print(f"Changing working directory from {Path.cwd()} to project folder {PROJ_FOLDER_ABSOLUTE}")
    os.chdir(PROJ_FOLDER_ABSOLUTE)
"""

if len(sys.path) == 0 or sys.path[0] != str(PROJ_FOLDER_ABSOLUTE):
    sys.path.insert(0, str(PROJ_FOLDER_ABSOLUTE))

import data_collection.automations_general_android as general
import data_collection.automations_specific_phone as specific


with open(COLLECTION_FOLDER_ABSOLUTE / "automations_agents.json", "r") as f:
    automations_agents_config = json.load(f)

# if COLLECTION_FOLDER_ABSOLUTE / "temp" does not exist, create it
if not os.path.exists(COLLECTION_FOLDER_ABSOLUTE / "temp"):
    os.makedirs(COLLECTION_FOLDER_ABSOLUTE / "temp")

recorder_env = automations_agents_config["recorder"]["conda_env"]

# image_save_path_folder = COLLECTION_FOLDER_ABSOLUTE / "screenshot"
# image_save_path = image_save_path_folder / "screenshot.png"



def poisson_interval(mean_seconds: float = 1.1) -> float:
    # Time between events in a Poisson process follows an exponential distribution
    return random.expovariate(1.0 / mean_seconds)
stop_event = threading.Event()
resume_event = threading.Event()


def run_useless_action_loop_method_2(mean_interval_seconds: float = 1.1):
    """
    Perform useless micro-swipes at random intervals.


    target_expected_interval_us=int(1e6 * 1.1),
    """
    print("Starting useless action loop method 2...")
    while not stop_event.is_set():
        sleep_time = poisson_interval(mean_interval_seconds)
        time.sleep(sleep_time)
        resume_event.wait()  # Wait until resumed

        # get the absolute path of the file
        fake_adb_py_path = FAKE_ADB_PATH_ABSOLUTE / "adb_wrapper.py"

        os.system(f"python {fake_adb_py_path} shell input fake custom_fake_action_3 2> /dev/null")



TIMESTAMP_RECORDER = COLLECTION_FOLDER_ABSOLUTE / "temp" / "timestamp_recorder.txt"


def prepare_data_collection_session(tmux_session_name: str):
    """result: a new tmux session is created. No move on phone."""
    os.system(f"tmux new-session -d -s {tmux_session_name}")
    os.system(f"tmux send-keys -t {tmux_session_name} \"conda deactivate\" Enter")
    os.system(f"tmux send-keys -t {tmux_session_name} \"conda activate {recorder_env}\" Enter")

def start_another_data_collection(tmux_session_name: str):
    """
        condition: tmux session is already existent and prompting; phone screen already opened an app in foreground  
        result: still phone show app in foreground; the app may not be competent for agent screenshot. A new data collection session started in tmux  
    """
    os.system(f"tmux send-keys -t {tmux_session_name} \"python3 {COLLECTION_FOLDER_ABSOLUTE}/main.py --automatic_switch_app\" Enter")

def generate_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_timestamp() -> str:
    with open(TIMESTAMP_RECORDER, "r") as f:
        timestamp = f.read().strip()
    return timestamp

def halt_data_collection_session(tmux_session_name: str) -> None:
    """
        condition: phone screen already opened an app in foreground
        result: data collection is halted and tmux prompting; phone can be in any state
        returns: a time stamp of the current data collection session
    """
    for _ in range(4):
        os.system(f"tmux send-keys -t {tmux_session_name} Enter")
        time.sleep(0.3)
    time.sleep(5.0) # wait for data collection to halt completely. Probably have to remove this waiting from legacy scripts.

default_data_collection_session_name = "mydadaacollection"

from abc import ABC, abstractmethod

class AutomationConfig(TypedDict):
    dir: str
    conda_env: str

def session_name_constructor(agent_name: str) -> str:
    return f"{agent_name}_data_collection_session"

class AgentLauncher(ABC):

    @abstractmethod
    def __init__(self, name: str):
        """
        Initialize the AgentLauncher with a given name, and prepare the tmux session.
        
        :param name: a self-defined name of the agent
        """
        self.name: str = name
        self.session_name: str = session_name_constructor(name)
        self.info_dict: AutomationConfig = automations_agents_config[name]
        self.recording_file_path_absolute = COLLECTION_FOLDER_ABSOLUTE / "temp" / f"{name}_whether_idle.txt"
        self._prepare_session()

    @abstractmethod
    def _prepare_session(self):
        """
        Instantiate a new tmux session self.session_name, and prepare the environment for the agent.
        
        :result: A tmux session is created and prepared. Can launch async_invoke_agent immediately after this if no data collection is needed.
        """
        pass

    @abstractmethod
    def async_invoke_agent(self, experiment_name: str):
        """
        Minimal code to invoke the agent to start a new experiment in a prepared session(sure to be prepared and idle on the instantiation of this agent object.) and when finishes, send idle signal with &&.  
        **Is non-blocking.**  
        You can also use this function to launch the agent for personal tasks; works as long as the session is idle, without any other preparations. 
        
        :param experiment_name: The query for the agent in human language.
        """
        pass

    def _write_session_as_busy(self):
        """
        Write "busy" to the indicator file to mark the session as busy.
        """
        with open(self.recording_file_path_absolute, "w") as f:
            f.write("busy\n")
        time.sleep(0.1)  # ensure the file is written

    def _check_if_session_is_idle(self) -> bool:
        """
        Should align busy/idle marking with the agent code.

        :return: Whether the indicator file shows the session is idle.
        :rtype: bool
        """
        with open(self.recording_file_path_absolute, "r") as f:
            content = f.read().strip()
            if content == "idle":
                return True
            elif content == "busy":
                return False
            else:
                raise ValueError(f"Unexpected content in {self.recording_file_path_absolute}: {content}")

    @abstractmethod
    def _block_until_session_is_idle(self, timeout_seconds: int):
        """
        Either wait until the session is idle, or force halt it after timeout.
        
        :param timeout_seconds: The timeout in seconds.
        """
        pass
    
    def _redirect_tmux_output_to_file(self, save_path: Path):
        """
        Redirect the tmux session output to the given file path.
        
        :param save_path: The path to stream the output into.
        """
        os.system(f"tmux pipe-pane -t {self.session_name}:0.0 -o \"cat >> {save_path}\"")

    def _end_tmux_output_redirection(self):
        """End the redirection of tmux session output, which also happen to flush the buffer."""
        os.system(f"tmux pipe-pane -t {self.session_name}:0.0")

    def do_an_experiment_core(self, experiment_name: str, timestamp: str, timeout_seconds: int = 1800):
        """
            Launch the agent until it halts.  
            Records tmux output; halting condition is either the agent finishes and marks idle, or timeout and we force halt it.
            Requires the app screen to be prepared i.e. a suitable state for agent to take over.

            returns: the path to the file where the tmux output is saved. The file is named with a random hex to avoid conflicts, and it's the caller's responsibility to move/rename it after this function returns.
        """

        os.system(f"echo 'launched {self.name} experiment on date ' $(date) >> experiment_log.txt")
        os.system(f"echo \'experiment name: {experiment_name}\' >> experiment_log.txt")

        random_hex_name = hex(random.randint(0, 2**32 - 1))[2:]
        previous_path = PROJ_FOLDER_ABSOLUTE / "logs" / f"agent_output_{random_hex_name}.txt"
        self._redirect_tmux_output_to_file(previous_path)
        
        resume_event.set()   # resume fake action generation

        self._write_session_as_busy()
        self.async_invoke_agent(experiment_name)
        self._block_until_session_is_idle(timeout_seconds=timeout_seconds)
        self._end_tmux_output_redirection()

        resume_event.clear()  # halt fake action generation

        target_path = PROJ_FOLDER_ABSOLUTE / "logs" /  f"agent_output_{timestamp}.txt"
        os.rename(previous_path, target_path)


    def do_an_experiment(self, callable_app_launch: Callable[[], None], experiment_name: str, timeout_seconds: int = 1800) -> str:
        """
        Do an experiment with the given app launch function and experiment name.
        Phone in any state before that.

        :param callable_app_launch: A function that launches the app to be tested; it require phone showing app screen. It clicks open the app and configure it until ready to serve.
        :param experiment_name: query for agent.
        :param timeout_seconds: The timeout in seconds.
        
        :return: A timestamp string indicating when the experiment was completed. e.g. 20231012_153045
        :rtype: str
        """        
        specific.prepare_apps_screen()
        callable_app_launch()
        start_another_data_collection(default_data_collection_session_name)
        time.sleep(20.0)
        timestamp = get_timestamp()
        
        self.do_an_experiment_core(experiment_name, timestamp, timeout_seconds)

        halt_data_collection_session(default_data_collection_session_name)

        return timestamp

    def destroy(self):
        """Destroy the tmux session."""
        os.system(f"tmux kill-session -t {self.session_name}")


class MobileAgentELauncher(AgentLauncher):
    def __init__(self, name: str = "mobile_agent_e", variation: Literal["gpt-4o-2024-11-20", "claude-3-5-sonnet-20241022"] = "gpt-4o-2024-11-20"):
        super().__init__(name)
        # session is prepared.
        self.variation = variation
        if self.variation != "gpt-4o-2024-11-20" and self.variation != "claude-3-5-sonnet-20241022":
            raise ValueError(f"Unsupported variation: {self.variation}")
        self._inject_mobile_agent_e_model_name()
    
    def _prepare_session(self):
        tmux_session_name = self.session_name
        os.system(f"tmux new-session -d -s {tmux_session_name}")
        # currently the conda stack is [ base | (environment of this script __main__) | base >>. So we need to deactivate twice.
        os.system(f"tmux send-keys -t {tmux_session_name} \"conda deactivate && conda deactivate && conda activate {self.info_dict['conda_env']}\" Enter")
        os.system(f"tmux send-keys -t {tmux_session_name} \"cd {self.info_dict['dir']}\" Enter")
        os.system(f"tmux send-keys -t {tmux_session_name} \"source scripts/backend_variables.sh\" Enter")
        os.system(f"tmux send-keys -t {tmux_session_name} \'export PATH={FAKE_ADB_PATH_ABSOLUTE}:$PATH\' Enter") # pastbug: this is where the path gets contaminated. should use single quotes instead of double quotes.

    def _inject_mobile_agent_e_model_name(self):
        os.system(f"tmux send-keys -t {self.session_name} \"export OPENAI_MODEL_NAME={self.variation}\" Enter")
        time.sleep(1.0)
        os.system(f"tmux send-keys -t {self.session_name} Enter")

    def async_invoke_agent(self, experiment_name: str):
        os.system(f"tmux send-keys -t {self.session_name} \"python run.py --run_name testing --setting individual --instruction \'{experiment_name}\' ; echo idle > {self.recording_file_path_absolute}\" Enter")

    def _block_until_session_is_idle(self, timeout_seconds: int = 1800):
        check_interval_seconds: int = 5

        for _ in range(timeout_seconds // check_interval_seconds):  # wait up to timeout_seconds
            time.sleep(check_interval_seconds)
            if self._check_if_session_is_idle():
                print("The mobile-agent-e session is idle now.")
                return
        print("The mobile-agent-e session is still not idle after timeout. Force halt it.")
        os.system(f"tmux send-keys -t {self.session_name} C-c")
        time.sleep(2.0)
        os.system(f"tmux send-keys -t {self.session_name} C-c")

MobileAgentELauncher_gpt_4o_2024_11_20 = lambda: MobileAgentELauncher(variation="gpt-4o-2024-11-20")
MobileAgentELauncher_claude_3_5_sonnet_20241022 = lambda: MobileAgentELauncher(variation="claude-3-5-sonnet-20241022")


class UiTarsLauncher(AgentLauncher):
    """
        Warning: Before refactoring, the async agent was first launched, then after 20 secs, fake action launched.
    """
    def __init__(self, name: str = "ui_tars"):
        super().__init__(name)

    # rewrite the messy functions here. Then we can delete the messy functions.
    def _prepare_session(self):
        tmux_session_name = self.session_name
        os.system(f"tmux new-session -d -s {tmux_session_name}")
        os.system(f"tmux send-keys -t {tmux_session_name} \"conda deactivate && conda deactivate && conda activate {self.info_dict['conda_env']}\" Enter")
        os.system(f"tmux send-keys -t {tmux_session_name} \"cd {self.info_dict['dir']}\" Enter")
        os.system(f"tmux send-keys -t {tmux_session_name} \'export PATH={FAKE_ADB_PATH_ABSOLUTE}:$PATH\' Enter")

    def async_invoke_agent(self, experiment_name: str):
        os.system(f"tmux send-keys -t {self.session_name} \"npx @ui-tars/cli start -t adb -q \'{experiment_name}\' ; echo idle > {self.recording_file_path_absolute}\" Enter")

    def _block_until_session_is_idle(self, timeout_seconds: int):
        check_interval_seconds: int = 5

        for _ in range(timeout_seconds // check_interval_seconds):  # wait up to timeout_seconds
            time.sleep(check_interval_seconds)
            if self._check_if_session_is_idle():
                print("The ui-tars session is idle now.")
                return
        print("The ui-tars session is still not idle after timeout. Force halt it.")
        os.system(f"tmux send-keys -t {self.session_name} C-c")
        time.sleep(2.0)
        os.system(f"tmux send-keys -t {self.session_name} C-c")


class CpmGuiAgentLauncher(AgentLauncher):

    def __init__(self, name: str = "cpm_gui_agent"):
        super().__init__(name)
    
    def _prepare_session(self):
        tmux_session_name = self.session_name
        os.system(f"tmux new-session -d -s {tmux_session_name}")
        os.system(f"tmux send-keys -t {tmux_session_name} \"conda deactivate && conda deactivate && conda activate {self.info_dict['conda_env']}\" Enter")
        os.system(f"tmux send-keys -t {tmux_session_name} \"cd {self.info_dict['dir']}\" Enter")
        os.system(f"tmux send-keys -t {tmux_session_name} \'export PATH={FAKE_ADB_PATH_ABSOLUTE}:$PATH\' Enter")

    def async_invoke_agent(self, experiment_name: str):
        os.system(f"tmux send-keys -t {self.session_name} \"python run_agent.py --task \'{experiment_name}\' ; echo idle > {self.recording_file_path_absolute}\" Enter")

    def _block_until_session_is_idle(self, timeout_seconds: int = 1800):
        check_interval_seconds: int = 5

        for _ in range(timeout_seconds // check_interval_seconds):  # wait up to timeout_seconds
            time.sleep(check_interval_seconds)
            if self._check_if_session_is_idle():
                print("The cpm-gui-agent session is idle now.")
                return
        print("The cpm-gui-agent session is still not idle after timeout. Force halt it.")
        os.system(f"tmux send-keys -t {self.session_name} C-c")
        time.sleep(2.0)
        os.system(f"tmux send-keys -t {self.session_name} C-c")
        

class OpenAutoGLMLauncher(AgentLauncher):

    def __init__(self, name: str = "open_autoglm"):
        super().__init__(name)

    def _prepare_session(self):
        tmux_session_name = self.session_name
        os.system(f"tmux new-session -d -s {tmux_session_name}")
        os.system(f"tmux send-keys -t {tmux_session_name} \"conda deactivate && conda deactivate && conda activate {self.info_dict['conda_env']}\" Enter")
        os.system(f"tmux send-keys -t {tmux_session_name} \"cd {self.info_dict['dir']}\" Enter")
        os.system(f"tmux send-keys -t {tmux_session_name} \'export PATH={FAKE_ADB_PATH_ABSOLUTE}:$PATH\' Enter")
        os.system(f"tmux send-keys -t {tmux_session_name} \"source environment_variables.sh\" Enter")

    def async_invoke_agent(self, experiment_name: str):
        os.system(f"tmux send-keys -t {self.session_name} \'python main.py --base-url $BASE_URL --model $MODEL_NAME --apikey $API_KEY {experiment_name} ; echo idle > {self.recording_file_path_absolute}\' Enter")

    def _open_autoglm_activate_in_case_of_stuck(self):
        """
        When Open_AutoGLM encounters human verification or black screen etc., it may say "Press Enter to continue", so we send Enter keys a few times to pretend to help it continue.
        
        :param self: Description
        """
        for _ in range(3):
            os.system(f"tmux send-keys -t {self.session_name} Enter")
            time.sleep(0.5)

    def _block_until_session_is_idle(self, timeout_seconds: int = 1800):
        check_interval_seconds: int = 5

        for _ in range(timeout_seconds // check_interval_seconds):  # wait up to timeout_seconds
            time.sleep(check_interval_seconds)
            self._open_autoglm_activate_in_case_of_stuck()
            if self._check_if_session_is_idle():
                print("The OpenAutoGLM session is idle now.")
                return
        print("The OpenAutoGLM session is still not idle after timeout. Force halt it.")
        os.system(f"tmux send-keys -t {self.session_name} C-c")
        time.sleep(2.0)
        os.system(f"tmux send-keys -t {self.session_name} C-c")

    


with open(COLLECTION_FOLDER_ABSOLUTE / "app_name_translations.json", "r") as f:
    translations: Dict[str, str] = json.load(f)

def load_not_done_task_pair(attendee_name: str, task_file_path: Path) -> Optional[Tuple[int, str, str]]:
    """
        if return none, no pending tasks for this attendee.  
        prints: the next task for the user, including app name and task description originally in the task csv.  
        returns: first_empty_row_idx, callable_app_name, task_description
    """
    df = pd.read_csv(task_file_path, header=0, dtype=str)
    
    app_name_col = "app_name"
    user_col = df[attendee_name]
    app_description_col = "app_description"
    # find rows where the user's column is empty (NaN or empty/whitespace)
    empty_row_idx = user_col[user_col.isnull() | (user_col.astype(str).str.strip() == "")].index
    print(empty_row_idx)
    if not empty_row_idx.empty:
        first_empty_row_idx = empty_row_idx[0]
        first_empty_row = df.iloc[first_empty_row_idx]
        task_file_app_name = first_empty_row[app_name_col]
        task_file_app_description = first_empty_row[app_description_col]

        callable_app_name = translations[task_file_app_name]
        print(f"Next task for user {attendee_name}:")
        print(f"App Name: {task_file_app_name}")
        print(f"App Description: {task_file_app_description}")
        return (first_empty_row_idx, callable_app_name, task_file_app_description)
    else:
        print(f"No pending tasks found for user {attendee_name}.")
        return None

def write_timestamp_to_idx(task_file_path: Path, attendee_name: str, empty_row_idx: int, timestamp: str):
    df = pd.read_csv(task_file_path, header=0, dtype=str)
    df.at[df.index[empty_row_idx], attendee_name] = timestamp
    df.to_csv(task_file_path, index=False)
    print(f"Recorded completion time for user {attendee_name} in task file.")

TASK_CSV_PATH = PROJ_FOLDER_ABSOLUTE / "tasks.csv"

try:
    os.system("adb devices")
    if __name__ == "__main__":
        time.sleep(1.0) # wait for adb to stabilize

    if __name__ == "__main__": # + "start_useless_action_loop":
        # create a subprocess to run the useless action loop
        if False:
            useless_action_thread = threading.Thread(target=run_useless_action_loop_method_1, args=("useless_action_rules.jsonl", 1.0, 0.5), daemon=True)
        else:
            # use useless swipes
            if False:
                useless_action_thread = threading.Thread(target=useless_micro_swipe_loop, args=(1.0, 200), daemon=True)
            else:
                useless_action_thread = threading.Thread(target=run_useless_action_loop_method_2, args=(1.1,), daemon=True)

        resume_event.clear()  # start with fake action generation halted
        useless_action_thread.start()
        input("Press Enter to continue to experiments...")
        print("Continuing to experiments...")

    if __name__ == "__main__":
        for agent_factory, column_namer in [
            # collect raw for example. Change accordingly for other experiments.
            (UiTarsLauncher, "ui_tars_raw"),
            (MobileAgentELauncher_gpt_4o_2024_11_20, "mobile_agent_e_gpt_4o_1228_raw"),
            (MobileAgentELauncher_claude_3_5_sonnet_20241022, "mobile_agent_e_claude_sonnet_3_7_raw"),
            (CpmGuiAgentLauncher, "cpm_gui_agent_humanity_rotate"),
            (OpenAutoGLMLauncher, "open_autoglm_agent_raw"),
        ]:
            prepared_agent_object = agent_factory()
            prepare_data_collection_session(default_data_collection_session_name)
            experiment_func = prepared_agent_object.do_an_experiment
            while True:
                next_task = load_not_done_task_pair(column_namer, TASK_CSV_PATH)
                if next_task is None:
                    break
                empty_row_idx, app_name, task_description = next_task
                callable_app = specific.general_get_app_func(app_name)
                timestamp = experiment_func(callable_app, task_description)
                print(f"Received timestamp: {timestamp}")
                write_timestamp_to_idx(TASK_CSV_PATH, column_namer, empty_row_idx, timestamp)
                time.sleep(2.0)
            time.sleep(30.0) # so that the data pulling can be done
            os.system("tmux kill-session -t " + default_data_collection_session_name)
            prepared_agent_object.destroy()
            print(f"All experiments for {column_namer} done.")


except Exception as e:
    print("Exception occurred:", e)
    
    # halt the useless process if it exists
            # No direct way to kill a thread; in real code, use a threading.Event to signal it to stop
            # Here we just let it be daemon so it will exit when main program exits
    for agent_name in automations_agents_config.keys():
        session_name = session_name_constructor(agent_name)
        os.system(f"tmux kill-session -t {session_name}")

    # send enter to data collection session to halt it
    halt_data_collection_session(default_data_collection_session_name)
    time.sleep(10.0)
    os.system("tmux kill-session -t " + default_data_collection_session_name)
    print("All sessions killed due to exception.")
