import os
import time
from typing import Dict, List, Optional, Tuple, Callable
import datetime
import PIL.Image

import threading

# ...existing code...
import json
import random
import pandas as pd

def fast_screenshot(adb_path: str = "adb") -> PIL.Image.Image:
    if True:
        command = adb_path + " shell rm /sdcard/screenshot.png"
        os.system(command)
        command = adb_path + " shell screencap -p /sdcard/screenshot.png"
        os.system(command)
        
        # if ./screenshot is not existent, then create the directory
        if not os.path.exists("./screenshot"):
            os.makedirs("./screenshot")

        command = adb_path + " pull /sdcard/screenshot.png ./screenshot/screenshot.png"
        os.system(command)
    else:
        pass
    image = PIL.Image.open("./screenshot/screenshot.png")
    # resize image to 1080x1920
    image = image.resize((1080, 1920))
    return image

def get_rgb(image: PIL.Image.Image, x: int, y: int) -> Tuple[int, int, int]:
    pixel = image.getpixel((x, y))
    if isinstance(pixel, int):
        return (pixel, pixel, pixel)
    elif len(pixel) == 4:
        return pixel[:3]
    elif len(pixel) == 3:
        return pixel
    else:
        raise ValueError("Unsupported pixel format")

def get_screenshot_and_get_rgb(x: int, y: int) -> Tuple[int, int, int]:
    image = fast_screenshot()
    return get_rgb(image, x, y)

def l1_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> int:
    return sum(abs(c1[i] - c2[i]) for i in range(3))

def pixel_on_screenshot_is_color(x: int, y: int, target_color: Tuple[int, int, int], l1_distance_threshold: int) -> bool:
    current_color = get_screenshot_and_get_rgb(x, y)
    distance = l1_distance(current_color, target_color)
    return distance <= l1_distance_threshold

def tap(x: int, y: int):
    os.system(f"adb shell input tap {x} {y}")

password: List[Tuple[int, int]] = [

]

def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
    os.system(f"adb shell input swipe {x1} {y1} {x2} {y2} {duration_ms}")

def power():
    os.system("adb shell input keyevent 26")

def start_phone():
    # power()
    swipe(540, 1800, 540, 700, duration_ms=200)
    for x, y in password:
        tap(x, y)
        time.sleep(0.1)

def app_switch():
    os.system("adb shell input keyevent KEYCODE_APP_SWITCH")

def home():
    os.system("adb shell input keyevent KEYCODE_HOME")

def clean_every_app(final_wait_seconds: float = 4.0):
    """
    start: anywhere anystate  
    end: home screen with no app in background  
    """
    for _ in range(10):
        home()
        time.sleep(0.1)
    app_switch()
    time.sleep(0.3)
    tap(540, 1800)
    time.sleep(final_wait_seconds)

def prepare_apps_screen():
    """
    start: anywhere anystate  
    end: app screen with no app in background  
    """
    clean_every_app()
    # for i in range(3): # no need now because app screen is set to top 
    #     swipe(900, 1100, 100, 1100, duration_ms=300)
    # time.sleep(0.1)

def switch_app_from_sensorevent_to_appscreen():
    """
    
    start: sensor event app in foreground
    end: app screen in foreground
    """
    time.sleep(1.0)
    app_switch()
    time.sleep(1.0)
    tap(850, 1300)
    time.sleep(1.0)

def back():
    os.system("adb shell input keyevent KEYCODE_BACK")


def poisson_interval(mean_seconds: float = 1.1) -> float:
    # Time between events in a Poisson process follows an exponential distribution
    return random.expovariate(1.0 / mean_seconds)
stop_event = threading.Event()
resume_event = threading.Event()


def run_useless_action_loop_method_2(mean_interval: float = 1.1):
    """
    Perform useless micro-swipes at random intervals.


    target_expected_interval_us=int(1e6 * 1.1),
    """
    print("Starting useless action loop method 2...")
    while not stop_event.is_set():
        sleep_time = poisson_interval(mean_interval)
        time.sleep(sleep_time)
        resume_event.wait()  # Wait until resumed
        os.system("python fake_adb/adb_wrapper.py shell input fake custom_fake_action_3 2> /dev/null")


x_coords = [160, 420, 680, 940]
y_coords = [150,
            390,
            630,
            870,
            1110, 
            1350]

table_of_app_funcs = [
["",        "zhihu",   "voov",     "qqmusic"],
["tdocs",   "taobao",  "qunar",    "jd"     ],
["",        "",        "iqiyi",    "ctrip"  ],
["eleme",   "meituan", "bilibili", "rednote"],
["gaode",   "haodf",   "toutiao",  "youdao"],
["cainiao", "weibo",   "dianping", "umetrip"]
]

def query_position(app_func_name: str) -> Tuple[int, int]:
    for (i, row) in enumerate(table_of_app_funcs):
        for (j, func_name) in enumerate(row):
            if func_name == app_func_name:
                return (x_coords[j], y_coords[i])

def jd():
    tap(916, 373)
    time.sleep(7.0)
    tap(665, 243)

def ctrip():
    tap(898, 646)
    time.sleep(7.0)

def ctrip_disturbance_resolution():
    if not pixel_on_screenshot_is_color(540, 45, (37, 130, 245), 10):
        swipe(540, 266, 540, 1412, duration_ms=2000)

def eleme():
    tap(132, 843) # open eleme
    time.sleep(7.0)
    tap(210, 927) # select sjtu as the place
    time.sleep(1.0)

def eleme_cleanup():
    prepare_apps_screen()
    eleme()
    tap(754, 1872) # click for the cart
    time.sleep(0.5)
    while True:
        if pixel_on_screenshot_is_color(720, 650, (214,229,231), 10):
            # 720, 650 # place for a gray-blue shadow
            # 470, 1223 # place for a very special blue icon
            break
        bin_coordinates = (991, 402)
        if pixel_on_screenshot_is_color(*bin_coordinates, (254, 246, 223), 10):
            print("The bin icon is not in the expected place.")
            bin_coordinates = (991, 524) # try a lower place
        tap(*bin_coordinates) # click delete for the first shop
        time.sleep(0.2)
        tap(709, 1090) # confirm delete
        time.sleep(1.0)

def qqmusic():
    tap(910, 156)
    time.sleep(15.0)

def iqiyi():
    tap(682, 626)
    time.sleep(7.0)

def tdocs():
    tap(160, 390)
    time.sleep(10.0)
    tap(860, 1530)
    time.sleep(1.0)
    tap(860, 1530)
    time.sleep(1.0)

"""
def general_run_app(app_name: str):
    # first check whether app_name is defined in global space; if not, query its position
    if app_name in globals():
        globals()[app_name]()
    else:
        pos = query_position(app_name)
        print(f"General run app: tapping at position {pos} for app {app_name}")
        tap(*pos)
        time.sleep(15.0)
"""

def general_get_app_func(app_name: str) -> Callable[[], None]:
    if app_name in globals():
        return globals()[app_name]
    else:
        def func():
            pos = query_position(app_name)
            tap(*pos)
            time.sleep(15.0)
        return func



PROJ_FOLDER = os.path.dirname(os.path.abspath(__file__))
FAKE_ADB_PATH = os.path.join(PROJ_FOLDER, "fake_adb")

with open("automations_agents.json", "r") as f:
    automations_agents_config = json.load(f)

USER_NAME = os.getenv("USER")
MACHINE_NAME = os.uname().nodename
recorder_path = automations_agents_config["recorder"]["conda_env"]




def prepare_data_collection_session(tmux_session_name: str):
    """result: a new tmux session is created. No move on phone."""
    os.system(f"tmux new-session -d -s {tmux_session_name}")
    os.system(f"tmux send-keys -t {tmux_session_name} \"conda deactivate\" Enter")
    os.system(f"tmux send-keys -t {tmux_session_name} \"conda activate {recorder_path}\" Enter")

def start_another_data_collection(tmux_session_name: str):
    """
        condition: tmux session is already existent and prompting; phone screen already opened an app in foreground  
        result: still phone show app in foreground; a new data collection session started in tmux  
    """
    os.system(f"tmux send-keys -t {tmux_session_name} \"python3 main.py --automatic_switch_app\" Enter")

def halt_data_collection_session(tmux_session_name: str) -> str:
    """
        condition: tmux session is already existent and prompting; phone screen already opened an app in foreground
        result: data collection is halted; phone can be in any state
        returns: a time stamp of the current data collection session
    """
    res = os.popen(f"tmux capture-pane -t {tmux_session_name} -p").read()
    # find the last _ in res
    last_underscore_idx = res.rfind("_")
    timestamp = res[last_underscore_idx - 8:last_underscore_idx + 7]
    for _ in range(4):
        os.system(f"tmux send-keys -t {tmux_session_name} Enter")
        time.sleep(0.3)
    return timestamp


mobile_agent_e_dir = automations_agents_config["mobile_agent_e"]["dir"]
mobile_agent_e_conda_env = automations_agents_config["mobile_agent_e"]["conda_env"]

def prepare_mobile_agent_e_session(tmux_session_name: str):
    os.system(f"tmux new-session -d -s {tmux_session_name}")
    # currently the conda stack is [ base | (environment of this script __main__) | base >>. So we need to deactivate twice.
    os.system(f"tmux send-keys -t {tmux_session_name} \"conda deactivate && conda deactivate && conda activate {mobile_agent_e_conda_env}\" Enter")
    os.system(f"tmux send-keys -t {tmux_session_name} \"cd {mobile_agent_e_dir}\" Enter")
    os.system(f"tmux send-keys -t {tmux_session_name} \"source scripts/backend_variables.sh\" Enter")
    os.system(f"tmux send-keys -t {tmux_session_name} \'export PATH={FAKE_ADB_PATH}:$PATH\' Enter") # pastbug: this is where the path gets contaminated. should use single quotes instead of double quotes.

def inject_mobile_agent_e_model_name(tmux_session_name: str, model_name: str):
    os.system(f"tmux send-keys -t {tmux_session_name} \"export OPENAI_MODEL_NAME={model_name}\" Enter")
    time.sleep(1.0)
    os.system(f"tmux send-keys -t {tmux_session_name} Enter")

def launch_mobile_agent_e(tmux_session_name: str, experiment_name: str):
    os.system(f"tmux send-keys -t {tmux_session_name} \"python run.py --run_name testing --setting individual --instruction \'{experiment_name}\'\" Enter")

ui_tars_dir = automations_agents_config["ui_tars"]["dir"]
ui_tars_conda_env = automations_agents_config["ui_tars"]["conda_env"]

def prepare_ui_tars_session(tmux_session_name: str):
    os.system(f"tmux new-session -d -s {tmux_session_name}")
    os.system(f"tmux send-keys -t {tmux_session_name} \"conda deactivate && conda deactivate && conda activate {ui_tars_conda_env}\" Enter")
    os.system(f"tmux send-keys -t {tmux_session_name} \"cd {ui_tars_dir}\" Enter")
    os.system(f"tmux send-keys -t {tmux_session_name} \'export PATH={FAKE_ADB_PATH}:$PATH\' Enter")

def launch_ui_tars_experiment(tmux_session_name: str, experiment_name: str):
    os.system(f"tmux send-keys -t {tmux_session_name} \"npx @ui-tars/cli start -t adb -q \'{experiment_name}\'\" Enter")

def check_if_ui_tars_session_is_idle(tmux_session_name: str) -> bool:
    # capture-pane -S -50 -E -10  # Capture from 50 lines up to 10 lines up
    # The -p flag outputs the captured content to stdout, allowing you to pipe it directly to files or other commands.
    res = os.popen(f"tmux capture-pane -t {tmux_session_name} -p -S -10").read()
    # print(res[-len(ui_tars_dir)-10:])
    if ui_tars_dir[-34:] in res[-len(ui_tars_dir)-10:]:
        return True
    return False

def check_if_mobile_agent_e_session_is_idle(tmux_session_name: str) -> bool:
    res = os.popen(f"tmux capture-pane -t {tmux_session_name} -p -S -5").read()
    # remove \n from res
    res = res.replace("\n", "")
    if (f"{USER_NAME}@{MACHINE_NAME}:" + mobile_agent_e_dir + "$") in res[-len(mobile_agent_e_dir)-20:]:
        return True
    return False

default_data_collection_session_name = "mydadaacollection"
default_ui_tars_session_name = "ui_Tars_session"
default_mobile_agent_e_session_name = "mobile_agent_e_session"

def do_an_ui_tars_experiment(callable_app_launch: Callable[[], None], experiment_name: str) -> str:
    """
        callable_app_launch: a function that launches the app to be tested; it require phone showing app screen. It clicks open the app and configure it until ready to serve.
    """
    resume_event.clear()
    prepare_apps_screen()
    callable_app_launch()
    start_another_data_collection(default_data_collection_session_name)
    time.sleep(8.0)
    launch_ui_tars_experiment(default_ui_tars_session_name, experiment_name)
    time.sleep(20.0)
    resume_event.set()
    while True:
        time.sleep(5.0)
        if check_if_ui_tars_session_is_idle(default_ui_tars_session_name):
            print("The ui-tars session is idle now.")
            break
    timestamp = halt_data_collection_session(default_data_collection_session_name)
    time.sleep(5.0)
    return timestamp

def do_an_mobile_agent_e_experiment(callable_app_launch: Callable[[], None], experiment_name: str, timeout_seconds: int = 1800):
    resume_event.clear()
    prepare_apps_screen()
    callable_app_launch()
    start_another_data_collection(default_data_collection_session_name)
    os.system(f"echo 'launched mobile-agent-e experiment on date ' $(date) >> experiment_log.txt")
    os.system(f"echo \'experiment name: {experiment_name}\' >> experiment_log.txt")
    time.sleep(8.0)
    launch_mobile_agent_e(default_mobile_agent_e_session_name, experiment_name)
    time.sleep(20.0)
    resume_event.set()
    for _ in range(timeout_seconds // 5):
        time.sleep(5.0)
        if check_if_mobile_agent_e_session_is_idle(default_mobile_agent_e_session_name):
            print("The mobile-agent-e session is idle now.")
            break
    if not check_if_mobile_agent_e_session_is_idle(default_mobile_agent_e_session_name):
        print("The mobile-agent-e session is still not idle after timeout. Force halt it.")
        os.system(f"tmux send-keys -t {default_mobile_agent_e_session_name} C-c")
        time.sleep(2.0)
        os.system(f"tmux send-keys -t {default_mobile_agent_e_session_name} C-c")
    timestamp = halt_data_collection_session(default_data_collection_session_name)
    time.sleep(5.0)
    return timestamp


default_cpm_gui_session_name = "cpm_gui_agent_session"
# launch a cpm_gui_agent session; it is locatdcx

default_cpm_gui_agent_dir = automations_agents_config["cpm_gui_agent"]["dir"]
default_cpm_gui_agent_conda_env = automations_agents_config["cpm_gui_agent"]["conda_env"]
def prepare_cpm_gui_agent_session(tmux_session_name: str):
    os.system(f"tmux new-session -d -s {tmux_session_name}")
    os.system(f"tmux send-keys -t {tmux_session_name} \"conda deactivate && conda deactivate && conda activate {default_cpm_gui_agent_conda_env}\" Enter")
    os.system(f"tmux send-keys -t {tmux_session_name} \"cd {default_cpm_gui_agent_dir}\" Enter")
    os.system(f"tmux send-keys -t {tmux_session_name} \'export PATH={FAKE_ADB_PATH}:$PATH\' Enter")

def launch_cpm_gui_agent(tmux_session_name: str, experiment_name: str):
    os.system(f"tmux send-keys -t {tmux_session_name} \"python run_agent.py --task \'{experiment_name}\'\" Enter")

def check_if_cpm_gui_agent_session_is_idle(tmux_session_name: str) -> bool:
    res = os.popen(f"tmux capture-pane -t {tmux_session_name} -p -S -2").read()
    # remove \n from res
    res = res.replace("\n", "")
    if (f"{USER_NAME}@{MACHINE_NAME}:" + default_cpm_gui_agent_dir + "$") in res[-len(default_cpm_gui_agent_dir)-20:]:
        return True
    return False

def do_a_cpm_gui_agent_experiment(callable_app_launch: Callable[[], None], experiment_name: str, timeout_seconds: int = 1800):
    prepare_apps_screen()
    callable_app_launch()
    start_another_data_collection(default_data_collection_session_name)
    os.system(f"echo 'launched cpm-gui-agent experiment on date ' $(date) >> experiment_log.txt")
    os.system(f"echo \'experiment name: {experiment_name}\' >> experiment_log.txt")
    time.sleep(8.0)
    launch_cpm_gui_agent(default_cpm_gui_session_name, experiment_name)
    time.sleep(20.0)
    for _ in range(timeout_seconds // 5):
        time.sleep(5.0)
        if check_if_cpm_gui_agent_session_is_idle(default_cpm_gui_session_name):
            print("The cpm-gui-agent session is idle now.")
            break
    if not check_if_cpm_gui_agent_session_is_idle(default_cpm_gui_session_name):
        print("The cpm-gui-agent session is still not idle after timeout. Force halt it.")
        os.system(f"tmux send-keys -t {default_cpm_gui_session_name} C-c")
        time.sleep(2.0)
        os.system(f"tmux send-keys -t {default_cpm_gui_session_name} C-c")
    timestamp = halt_data_collection_session(default_data_collection_session_name)
    time.sleep(5.0)
    return timestamp


with open("app_name_translations.json", "r") as f:
    translations: Dict[str, str] = json.load(f)

def load_not_done_task_pair(attendee_name: str, task_file_path: str) -> Optional[Tuple[int, str, str]]:
    """
        if return none, no pending tasks for this attendee
        returns: first_empty_row_idx, app_name, task_description
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
        translated_app_name = translations[first_empty_row[app_name_col]]
        print(f"Next task for user {attendee_name}:")
        print(f"App Name: {first_empty_row[app_name_col]}")
        print(f"App Description: {first_empty_row[app_description_col]}")
        return (first_empty_row_idx, translated_app_name, first_empty_row[app_description_col])
    else:
        print(f"No pending tasks found for user {attendee_name}.")
        return None

def write_timestamp_to_idx(task_file_path: str, attendee_name: str, empty_row_idx: int, timestamp: str):
    df = pd.read_csv(task_file_path, header=0, dtype=str)
    df.at[df.index[empty_row_idx], attendee_name] = timestamp
    df.to_csv(task_file_path, index=False)
    print(f"Recorded completion time for user {attendee_name} in task file.")

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
        useless_action_thread.start()
        input("Press Enter to continue to experiments...")
        print("Continuing to experiments...")

    if __name__ == "__main__" + "ui_tars":
        
        prepare_data_collection_session(default_data_collection_session_name)
        prepare_ui_tars_session(default_ui_tars_session_name)
        column_namer = "ui_tars_raw"
        experiment_func = do_an_ui_tars_experiment
        while True:
            next_task = load_not_done_task_pair(column_namer, "tasks.csv")
            if next_task is None:
                break
            empty_row_idx, app_name, task_description = next_task
            callable_app = general_get_app_func(app_name)
            timestamp = experiment_func(callable_app, task_description)
            print(f"Received timestamp: {timestamp}")
            write_timestamp_to_idx("tasks.csv", column_namer, empty_row_idx, timestamp)
            time.sleep(2.0)
        time.sleep(30.0) # so that the data pulling can be done
        os.system("tmux kill-session -t " + default_data_collection_session_name)
        os.system("tmux kill-session -t " + default_ui_tars_session_name)
        print("All experiments done.")

    if __name__ == "__main__" + "mobile_agent_e":
        prepare_data_collection_session(default_data_collection_session_name)
        prepare_mobile_agent_e_session(default_mobile_agent_e_session_name)
        
        inject_mobile_agent_e_model_name(
            default_mobile_agent_e_session_name,
              "gpt-4o-2024-11-20")

        column_namer = "mobile_agent_e_gpt_4o_1228_raw"
        experiment_func = do_an_mobile_agent_e_experiment
        while True:
            next_task = load_not_done_task_pair(column_namer, "tasks.csv")
            if next_task is None:
                break
            empty_row_idx, app_name, task_description = next_task
            callable_app = general_get_app_func(app_name)
            timestamp = experiment_func(callable_app, task_description)
            print(f"Received timestamp: {timestamp}")
            write_timestamp_to_idx("tasks.csv", column_namer, empty_row_idx, timestamp)
            time.sleep(2.0)
        time.sleep(30.0) # so that the data pulling can be done
        os.system("tmux kill-session -t " + default_data_collection_session_name)
        os.system("tmux kill-session -t " + default_mobile_agent_e_session_name)
        print("All experiments done.")

    if __name__ == "__main__": # + "mobile_agent_e":
        prepare_data_collection_session(default_data_collection_session_name)
        prepare_mobile_agent_e_session(default_mobile_agent_e_session_name)
        inject_mobile_agent_e_model_name(
            default_mobile_agent_e_session_name,
              "claude-3-5-sonnet-20241022")
        column_namer = "mobile_agent_e_claude_sonnet_3_7_raw"
        experiment_func = do_an_mobile_agent_e_experiment
        while True:
            next_task = load_not_done_task_pair(column_namer, "tasks.csv")
            if next_task is None:
                break
            empty_row_idx, app_name, task_description = next_task
            callable_app = general_get_app_func(app_name)
            timestamp = experiment_func(callable_app, task_description)
            print(f"Received timestamp: {timestamp}")
            write_timestamp_to_idx("tasks.csv", column_namer, empty_row_idx, timestamp)
            time.sleep(2.0)
        time.sleep(30.0) # so that the data pulling can be done
        os.system("tmux kill-session -t " + default_data_collection_session_name)
        os.system("tmux kill-session -t " + default_mobile_agent_e_session_name)
        print("All experiments done.")

    if __name__ == "__main__" + "cpm_gui_agent":
        prepare_data_collection_session(default_data_collection_session_name)
        prepare_cpm_gui_agent_session(default_cpm_gui_session_name)
        column_namer = "cpm_gui_agent_humanity_rotate"
        experiment_func = do_a_cpm_gui_agent_experiment
        while True:
            next_task = load_not_done_task_pair(column_namer, "tasks.csv")
            if next_task is None:
                break
            empty_row_idx, app_name, task_description = next_task
            callable_app = general_get_app_func(app_name)
            timestamp = experiment_func(callable_app, task_description)
            print(f"Received timestamp: {timestamp}")
            write_timestamp_to_idx("tasks.csv", column_namer, empty_row_idx, timestamp)
            time.sleep(2.0)
        time.sleep(30.0) # so that the data pulling can be done
        os.system("tmux kill-session -t " + default_data_collection_session_name)
        os.system("tmux kill-session -t " + default_cpm_gui_session_name)
        print("All experiments done.")




except Exception as e:
    print("Exception occurred:", e)
    
    # halt the useless process if it exists
            # No direct way to kill a thread; in real code, use a threading.Event to signal it to stop
            # Here we just let it be daemon so it will exit when main program exits
    os.system("tmux kill-session -t " + default_ui_tars_session_name)
    os.system("tmux kill-session -t " + default_mobile_agent_e_session_name)

    # send enter to data collection session to halt it
    halt_data_collection_session(default_data_collection_session_name)
    time.sleep(10.0)
    os.system("tmux kill-session -t " + default_data_collection_session_name)
    print("All sessions killed due to exception.")
