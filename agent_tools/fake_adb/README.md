# Fake ADB Wrapper

The Fake ADB Wrapper is an ADB (Android Debug Bridge) wrapper that intercepts and modifies touch input commands to make agent interactions in the digital platform's perception loggable and appear more human-like. It is designed for use with mobile GUI agents that need to exhibit human-like touch dynamics to avoid detection by digital platforms.

## Purpose

This tool enables agents to:
- Generate human-like tap and swipe gestures with realistic timing
- Inject fake actions (non-functional gestures) to mask mechanical execution patterns
- Record and replay touch events at hardware event level (via `/dev/input/eventX`)
- Support both bot-like (precise, mechanical) and human-like (noisy, variable) behavior modes

## Prerequisites

### Dependencies

1. **android-touch-record-replay** binary:
   - Follow instructions at https://github.com/Cartucho/android-touch-record-replay
   - Binary must be available at `/data/local/tmp/mysendevent-arm64` on your Android device

2. **Python Requirements** (from `requirements.txt`):
   ```bash
   pip install numpy pandas matplotlib scipy scikit-learn seaborn xgboost
   ```

3. **ADB** installed and configured on your system with USB debugging enabled on Android device

4. **Unix-like System** (Linux, macOS):
   - This tool uses `fcntl` for process locking, which is not available on Windows

5. **Recorded Human Swipes** (Required for human-like behavior):
   - Download the pre-recorded human swipe data from the Hugging Face dataset `metadata/` and place it in `analysis/processing/swipe_data.pkl`

## Installation

1. Clone this repository and navigate to the project root:
   ```bash
   cd /path/to/Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark
   ```

2. Try add the fake_adb directory to your PATH:
   ```bash
   export PATH=/path/to/agent_tools/fake_adb:$PATH
   ```
   This is automatically done when `data_collection/automations.py` is collecting agent data.

3. The wrapper is ready to use. See Configuration below.

## Configuration

Edit `adb_wrapper_config.json`:

```json
{
    "real_adb": "/home/<user_name>/Android/Sdk/platform-tools/adb", // example path to your adb binary; run `which adb` to find it
    "global_touch_device": "/dev/input/event4",
    "global_event_interval_us": 11000,
    "global_fake_human": false
}
```

### Configuration Fields

| Field | Description | Example |
|-------|-------------|---------|
| `real_adb` | Path to your actual ADB binary | `/home/user/Android/Sdk/platform-tools/adb` |
| `global_touch_device` | Touch device input path on Android | `/dev/input/event4` |
| `global_event_interval_us` | Time interval between dots in swipe (microseconds) | `11000` (11ms) |
| `global_fake_human` | Enable human-like behavior (vs bot-like) | `true` or `false` |

**Note**: 
- The `global_touch_device` value is device-specific. Find it by:
```bash
adb shell getevent -l | grep ABS_MT_POSITION
```
The output will show the device path (e.g., `/dev/input/event4`).
- The `global_event_interval_us` value is also device-specific. Find it out by running the sensor logger `MyMotionLogger` app, then run `adb shell input swipe 100 500 900 500` to make a swipe on the blank zone in the sensor logger app, and check the sensorevent log for the timestamps of Motion Events.

- The time taken for a tap is actually also device-specific. Find it out by tapping on the blank zone in the sensor logger app with `adb shell input tap 500 800` and checking the interval between the down event and up event in the sensor logger. Adjust the default 1000us in `do_tap()` if needed.

## Usage

### Direct Command Usage

The wrapper intercepts and modifies standard `adb shell input` commands:

```bash
export PATH=/path/to/agent_tools/fake_adb:$PATH  # Override adb with fake_adb wrapper

# Tap at coordinates (x, y)
adb shell input tap 500 800

# Swipe from (x1, y1) to (x2, y2) over 500ms
adb shell input swipe 100 500 900 500 500

# Send text
adb shell input text "Hello World"
adb shell input text "你好世界"  # Unicode text; not possible with standard adb, automatically converted to advanced input with IME support under the hood

# Press keys
adb shell input keyevent KEYCODE_ENTER
adb shell input keyevent KEYCODE_BACK
```

These commands will be intercepted and modified to produce human-like behavior when `global_fake_human=true`.

### Python API

For programmatic use in agent code (don't add the fake_adb directory to PATH in this case, just import the wrapper directly):

```python
import agent_tools.fake_adb.adb_wrapper as fake_adb

# Tap with human-like behavior
fake_adb.do_tap("adb", x=500, y=800, fake_human=True)

# Swipe with human-like behavior
fake_adb.do_swipe("adb", x1=100, y1=500, x2=900, y2=500, fake_human=True)

# Swipe with specific duration (bot mode)
fake_adb.do_swipe("adb", x1=100, y1=500, x2=900, y2=500, time_ms=500)

# Type text
fake_adb.do_type("adb", "Hello World")

# Press keys
fake_adb.do_enter("adb")
fake_adb.do_key_sequence("adb", 102)  # HOME key
```

### Fake Actions (Humanization)

Inject non-functional gestures to disrupt mechanical patterns:

```bash
# Fake action 1: Short press, move right, release
adb shell input fake custom_fake_action_1

# Fake action 2: Draw a square
adb shell input fake custom_fake_action_2

# Fake action 3: Draw a circle (requires prior tap position)
adb shell input fake custom_fake_action_3
```

These can be called in parallel with agent actions to mask timing patterns.

## Advanced Features

### Tap Position Recording

Fake action 3 uses the last tap position as circle center. The wrapper automatically records tap positions:

```bash
adb shell input tap 540 960  # Position saved to tap_position_record.txt
adb shell input fake custom_fake_action_3  # Circle drawn around (540, 960)
```

### IME Event Capturing

Keyboard input events are logged to `ime_bin_event_capturer.txt` for later analysis. The log path can be customized via `IME_EVENT_PATH.txt`:

```bash
echo "/absolute/path/to/log.txt" > IME_EVENT_PATH.txt
```

### MotionGenerator API

For fine-grained control over gesture generation:

```python
from agent_tools.fake_adb.adb_wrapper import MotionGenerator

# Generate human-like swipe trace
trace = MotionGenerator.generate_swipe_trace(
    x1=100, y1=500, x2=900, y2=500,
    duration_us=500000,
    neighbor_time_delta_us=11000,
    fake_human=True
)

# Convert to event sequence
event_sequence = MotionGenerator.swipe_to_event_trace(
    trace=trace,
    evdev="/dev/input/event4"
)

# Flush to device
MotionGenerator.flush_event_sequence(
    adb_path="adb",
    tosend_device="/dev/input/event4",
    event_list=event_sequence
)
```

## Troubleshooting

### "fcntl is not available on this platform"
This tool requires a Unix-like system. Windows is not supported. Use WSL (Windows Subsystem for Linux) or a Linux/macOS machine.

### "Device not found" or "Permission denied"
- Ensure USB debugging is enabled on Android device
- Run `adb devices` to verify device is detected
- Check that `real_adb` path in config is correct
- On Android, verify `/data/local/tmp/mysendevent-arm64` exists and is executable

### Swipes not working / no effect
- Verify `global_touch_device` in config matches your device's touch input path
- Test with: `adb shell getevent -l /dev/input/event4` (replace with your device)
- You should see events when touching the screen

### IME events not being logged
- Ensure `IME_EVENT_PATH.txt` contains a valid absolute path
- Check write permissions for the log directory
- When `data_collection/automations.py` is running, it should automatically log IME events to a non-bin file when the agent types; no interference is needed.


## Performance Considerations

- The wrapper uses a file lock (`adb_wrapper.lock`) to prevent concurrent invocations
- Each gesture is flushed as a batch to minimize round-trip latency
- Human-like mode adds ~80ms ± 10ms latency to taps (configurable)
- In bot mode, duration is typically < 1ms for taps

## Integration with Agents

To use fake_adb with your agent, without demand of automatic data collection:

1. Set the fake_adb directory in your PATH before launching the agent:
   ```bash
   export PATH=/path/to/agent_tools/fake_adb:$PATH
   ```

2. The agent's ADB commands will automatically be intercepted and modified, if it also calls `adb` through the command line under the hood. This is the case for most existing agents like UI-TARS, MobileAgent-E, etc. that use `adb shell input` commands for interaction. 

- If your agent uses other means rather than `adb` cli to interact with the phone, then non-trivial integration may be needed. You can directly import the `adb_wrapper` module in your agent code and call the provided APIs to perform human-like interactions.

For automated data collection with supported agents, simply run `data_collection/automations.py` and the fake ADB wrapper will be used automatically for those agents; for unsupported agents, please see [Using Custom Agents](../../data_collection/README.md#custom-agents) for instructions on how to set up.

## License

Part of the Agent Humanization Benchmark (AHB) for the "Turing Test on Screen" research project.
