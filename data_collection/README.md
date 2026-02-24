# Data Collection Guide

This directory contains tools for collecting high-fidelity interaction data from Android devices, including touch gestures, sensor events, and screen recordings. The data is used for the Agent Humanization Benchmark (AHB) to analyze and improve mobile GUI agent behavior.

## Overview

The data collection system captures:
- **Motion Events**: Raw touch input coordinates, pressure, timing from `/dev/input/eventX`
- **Sensor Events**: Accelerometer, gyroscope, magnetometer data from a custom Android app
- **Screen Recordings**: Video of the device screen during interaction
- **IME Events**: Keyboard input events for text entry analysis
- **Fake Actions**: Non-functional gestures for humanization research

## Prerequisites

### Hardware & Software Requirements

1. **Android Device**:
   - USB debugging enabled (Settings → Developer Options → USB debugging)
   - Android 7.0 or higher recommended
   - Screen unlocked (no pin/pattern lock during collection)

2. **Desktop System**:
   - Linux or macOS (Windows not fully supported)
   - Python 3.8 or higher
   - ADB installed and in PATH

3. **Python Dependencies**:
   ```bash
   pip install -r ../requirements.txt
   ```
   This includes: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `seaborn`, `xgboost`, `Pillow`, `psutil`

4. **ADBKeyboard** (required for IME event capturing):
   - Install from: https://github.com/senzhk/ADBKeyBoard
   - On device: `adb install ADBKeyboard.apk`
   - Set as default keyboard: `adb shell ime set com.android.adbkeyboard/.AdbIME`

5. **Fake ADB Wrapper** (for agent data collection, even if without humanization):
   - Located at `../agent_tools/fake_adb/`
   - See `../agent_tools/fake_adb/README.md` for setup

### Critical: Metadata Setup

Some functionality requires metadata files that are NOT in this online repository. Please refer to [](../README.md#critical-setup-steps) for instructions on how to obtain and set up these files.

**Required metadata files**:
- `app_name_translations.json` - App name to launcher function mapping
- `swipe_data.pkl` - Pre-processed human swipe data for agent humanization.
- `tasks.csv` - Task definitions for data collection

## Step-by-Step Setup

### 1. Install Motion Logger App

The motion logger app records sensor data (accelerometer, gyroscope, magnetometer).

**Option A: Compile from source** (recommended for customization):
```bash
# Clone the java branch of this repository in another directory
# Open MyMotionLogger project in Android Studio
# Build and install to device
```

**Option B: Use pre-compiled APK**:
```bash
adb install MyMotionLogger/app/build/outputs/apk/debug/app-debug.apk
```

**Configure the app**:
- Open the app on device
- Grant all permissions (especially background activity permissions)
- In Developer Options on device:
  - Increase "Background process limit" to allow background recording
  - Add the app to "Apps that can run in background"

### 2. Configure Screen Recording

The default Android `screenrecord` has a 180-second limit. **This modification is necessary** for recordings longer than 3 minutes:

#### Method A: Use Modified Binary (Recommended)

1. Pull the screenrecord binary:
   ```bash
   adb pull /system/bin/screenrecord screenrecord
   ```

2. Convert to hex and edit:
   ```bash
   xxd screenrecord > screenrecord.hex
   ```

3. Edit `screenrecord.hex`:
   - Find the last occurrence of `b400 0000` (hex for 180 in little-endian)
   - Change to `0807 0000` (hex for 1800 = 30 minutes)
   - **Note**: The exact bytes to modify may vary by device. Search for `b400` first.

4. Convert back and push:
   ```bash
   xxd -r screenrecord.hex > screenrecord_modified
   chmod +x screenrecord_modified
   adb push screenrecord_modified /data/local/tmp/
   ```

#### Method B: Use Existing Modified Binary

If you already have a modified binary:
```bash
adb push screenrecord_modified /data/local/tmp/
```
However, it is not guaranteed to work on all devices due to differences in the `screenrecord` implementation.

**Important**:

For automation:
- Place the binary at `/data/local/tmp/screenrecord_modified` (required by `controller.py`)

For standalone usage:
- Do NOT use the `--time-limit` argument with the modified binary
- To stop recording, use `adb shell pkill -l SIGINT screenrecord` (not the modified filename)

### 3. Configure Agent Automation (Recommended)

For automated agent data collection, edit `automations_agents.json`:

```json
{
    "recorder": {
        "conda_env": "your_recorder_env_name"
    },
    "mobile_agent_e": {
        "dir": "/absolute/path/to/mobile_agent_e",
        "conda_env": "mobile_agent_e_env"
    },
    "ui_tars": {
        "dir": "/absolute/path/to/ui_tars",
        "conda_env": "your_recorder_env_name" // since we use the javascript version for ui-tars, we can run it using the conda environment for recording to support only the adb wrapper.
    },
    "cpm_gui_agent": {
        "dir": "/absolute/path/to/cpm_gui_agent",
        "conda_env": "cpm_gui_agent_env"
    },
    "open_autoglm": {
        "dir": "/absolute/path/to/Open-AutoGLM",
        "conda_env": "open_autoglm_env"
    }
}
```

The first "recorder" entry is for the later `main.py` that records data under the orchestration of `automations.py`.

See `agents/README.md` for agent-specific setup instructions.

## Usage

### Collecting Human Data (With Some Automation)

Prerequistes: [Fulfil all requirements of collecting human data without automation](#collecting-human-data-without-automation), [Configure phone-specific automation](#automations_specific_phonepy)

This mode records a human user performing tasks when you want the device to prepare itself automatically.

```bash
python data_collection/main.py \
    --user your_username \
    --task_provide_file tasks.csv \
    --automatically_prepare_provided_task_app \
    --automatic_switch_app \
    --automatic_exit_app_and_reset
```

**Options**:
- `--user`: Your identifier (used for organizing data)
- `--task_provide_file`: CSV file with tasks (see below)
- `--automatically_prepare_provided_task_app`: Automatically opens the required app before recording
- `--automatic_switch_app`: Switches from sensor logger to app screen automatically
- `--automatic_exit_app_and_reset`: Cleans up app state after recording

**Workflow**:

0. You need to ensure that `data_collection/automations_specific_phone.py` has resetters for the main app screen and launchers for the apps in your tasks.csv and `app_name_translations.json`
1. The script reads the next pending task for your user from the CSV
2. Opens the required app (if `--automatically_prepare_provided_task_app` is set)
3. Starts recording gestures, sensors, screen, and IME events
4. Waits for you to press Enter when done
5. Please Patiently wait for the recording to stop and save all data to `./logs/`
6. The timestamp for the completed task is automatically marked in the CSV in the end

### Collecting Human Data (Without Automation)

Prerequistes: [Hardware & Software Requirements (Excluding adb wrapper setup)](#hardware--software-requirements), [Critical Metadata Setup](#critical-metadata-setup), [Sensor logger setup](#1-install-motion-logger-app), [Configure Screen Recording](#2-configure-screen-recording)

```bash
python data_collection/main.py
```

This is useful when you're already on the app screen and just want to start recording. However, You need to manually dismiss the sensor recorder popup and record down the task timestamp manually for later analysis.

### Collecting Agent Data

Prerequisites: [Hardware & Software Requirements](#hardware--software-requirements), [Critical Metadata Setup](#critical-metadata-setup), [Configure Screen Recording](#2-configure-screen-recording), [Configure Agent Automation](#3-configure-agent-automation-recommended)

To collect data from automated agents (UI-TARS, MobileAgent-E, etc.):

```bash
python data_collection/automations.py
```

This runs automated experiments with agents. You need to:
0. Download the agents. If it is not listed in our support list, please refer to [Custom Agent Setup](#custom-agents).
1. Configure `automations_agents.json` with agent repo root paths and conda environment names; note that these conda environments should support running the fake adb wrapper, so add the required dependencies `requirements.txt` to those environments.
2. Set up fake_adb in your PATH (for humanized agents)
3. Have a `tasks.csv` file with tasks for the agent to perform

**See `automations.py` for agent-specific experiment runners**:
- `__main__ui_tars`: UI-TARS experiments
- `__main__mobile_agent_e`: MobileAgent-E experiments (multiple model variants)
- `__main__cpm_gui_agent`: CPM-GUI-Agent experiments
- `__main__`: OpenAutoGLM experiments
- They will be refactored to match the more modular structure of OpenAutoGLM in the near future.

## File Structure

After running data collection, files are saved to `./logs/`:

```
logs/
├── gesture_recording_20260219_103045.log      # Raw touch events from getevent
├── sensor_recording_20260219_103045.txt       # Sensor data from motion logger
├── screen_recording_20260219_103045.mp4       # Screen recording video
├── IME_event_20260219_103045.txt              # Keyboard input events
├── agent_output_20260219_103045.txt           # Agent console output (if applicable)
└── timestamp_recorder.txt                      # Current recording timestamp
```

### Gesture Log Format

Each line in `gesture_recording_*.log` follows the `getevent -lt` format:

```
[   12345.678901] EV_ABS       ABS_MT_POSITION_X    00000213
[   12345.678902] EV_ABS       ABS_MT_POSITION_Y    000003c8
[   12345.678903] EV_SYN       SYN_REPORT           00000000
```

You don't need to parse these manually; the `analysis/lib/gesture_log_reader_utils.py` library have relevant functions to process them directly.

### Sensor Log Format

Each line in `sensor_recording_*.txt` contains timestamped sensor readings.
You don't need to parse these manually; the `analysis/lib/sensor_log_reader_utils.py` library have relevant functions to process them directly.

### IME Log Format

Each line in `IME_event_*.txt` contains phone timestamp and character:

```
123.456 `H`
123.457 `e`
123.458 `l`
```

## Tasks CSV Format

The `tasks.csv` file defines tasks for users/agents to perform. Structure:

```csv
app_name,Unnamed: 1,app_description,,user1,user2,ui_tars_raw,...
jd,,搜索"无线耳机",打开一个非自营卖家的商品页面，查看其一条评价。,20260219_103045,,,
taobao,something_unimportant,搜索"华为P50"。,,20260219_110000,,
...
```

- **app_name**: Must match an entry in `app_name_translations.json`
- **app_description**: Human-readable task description
- **User/Agent columns**: Timestamp when task was completed (empty = pending; if want to skip, manually fill any non-empty string)

## Configuration Files

### automations_agents.json

Configures agent paths and conda environments for automation, as mentioned in [before](#3-configure-agent-automation-recommended):

```json
{
    "agent_name": {
        "dir": "/absolute/path/to/agent",
        "conda_env": "environment_name"
    }
}
```

### automations_general_android.py

Contains general Android automation utilities:
- `tap(x, y)`: Tap at coordinates
- `swipe(x1, y1, x2, y2, duration_ms)`: Swipe gesture
- `fast_screenshot()`: Take screenshot
- `pixel_on_screenshot_is_color()`: Check pixel color

### automations_specific_phone.py

Contains phone-specific app launchers:
- App position grid (6 rows × 4 columns)
- App-specific launcher functions (e.g., `jd()`, `ctrip()`)
- `general_get_app_func(app_name)`: Returns launcher for any app
These may need to be modified based on your phone's ui layout and the apps you want to test with, and you may also need to consider quirks of the app being tested, such as tricky but fixed popups, configuration or authentication upon entry.

### app_name_translations.json

Maps task CSV app names to callable function names in `automations_specific_phone.py`.

## Common Issues

### ADB Device Not Found

```bash
# Check device is connected
adb devices

# If unauthorized, accept the RSA key prompt on device
# Then reconnect
adb kill-server
adb start-server
adb devices
```

### Screen Recording Truncated at 180s

**Symptoms**: Recording stops at 180s or fails to start

**Solutions**:
1. Verify modified binary is at `/data/local/tmp/screenrecord_modified`
2. Check file permissions: `adb shell ls -l /data/local/tmp/screenrecord_modified`
3. Try recording manually: `adb shell /data/local/tmp/screenrecord_modified /sdcard/test.mp4`

### Sensor Logger Not Recording

**Symptoms**: `sensor_recording_*.txt` is empty, missing or truncated at about 3 minutes

**Solutions**:
1. Verify app is installed: `adb shell pm list packages | grep motionlogger`
2. Check app has permissions: Open app manually and grant all permissions
3. Verify background activity: Check Developer Options → "Apps that can run in background"
4. Manually start app: `adb shell am start -S com.example.motionlogger/.MainActivity`

### Gesture Recording Stopped Early

**Symptoms**: Recording stops when device screen turns off

**Solutions**:
- Increase screen timeout in device settings
- Keep device plugged in during recording
- Don't let device sleep during collection (data alignment is tricky after sleep)

### Agent Automation Fails

**Symptoms**: Agent doesn't start or hangs

**Solutions**:
1. Verify agent paths in `automations_agents.json` are absolute
2. Check conda environments exist: `conda env list`
3. Ensure fake_adb is in PATH: `which adb` should point to fake_adb wrapper
4. Check tmux is installed: `tmux -V`
5. Review agent-specific README in `agents/` subdirectory
6. Attach to tmux session for manual inspection: `tmux attach-session -t agent_name`

## Advanced Usage

### Custom App Launchers

Add new app launchers to `automations_specific_phone.py`:

```python
def myapp():
    general.tap(540, 1000)  # Tap app icon
    time.sleep(15.0)         # Wait for app to load
    general.tap(200, 300)   # Additional setup if needed

# Add to table_of_app_funcs for automatic discovery
table_of_app_funcs[0][0] = "myapp"
```

Then add translation to `app_name_translations.json`:

```json
{
    "My App": "myapp"
}
```
[]()
### Custom Agents

In order to run experiments with new agents not listed in `automations_agents.json` and automatically collect data with the fake ADB wrapper with or without humanization, you can follow these steps:

0. Pull the agent yourself, find its demo and set it up according to its instructions.
1. `export PATH=/path/to/project/agent_tools/fake_adb:$PATH` in its running environment to test the fake ADB wrapper for humanized behavior; If the agent does not use the `adb` cli tool to operate the phone, you may need to modify its code to integrate the fake ADB wrapper for humanization, such as by replacing action functions to calls to `adb`. Run `main.py` while the agent is running to collect the data and examine whether the fake ADB wrapper is working as intended (e.g., by checking the generated fake actions in the gesture recording logs and whether they are human-like). 
1. Configure the agent in `automations_agents.json` with its directory and conda environment
2. Create a new agent class in `automations.py` that inherits from `AgentLauncher` and implements the required methods, similar to the existing agent experiment classes (e.g., `OpenAutoGLMLauncher`)
3. Add it to the agents being iterated in the `__main__` block of `automations.py` for automated running with a easy-to-identify name.
4. Add a column in `tasks.csv` for this agent named with the same easy-to-identify name.
5. You are ready to run automated experiments with this agent using `automations.py` and collect the data for analysis!


## Next Steps

After collecting data:
1. See `../analysis/README.md` for data processing and analysis
2. Use `analysis_playground.ipynb` to explore recorded data
3. Extract features and train detectors using the analysis libraries
