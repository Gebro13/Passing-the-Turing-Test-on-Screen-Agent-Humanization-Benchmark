# Data Analysis Guide

This directory contains tools for processing and analyzing interaction data collected from Android devices. The analysis framework is designed to extract behavioral features, compare human vs. agent behavior, and evaluate humanization strategies for the Agent Humanization Benchmark (AHB).

## Overview

The analysis pipeline:
1. **Read raw logs**: Parse gesture and sensor files.
2. **Extract features**: Compute various features from interactions
3. **Compare populations**: Analyze differences between human and agent behaviors
4. **Train detectors**: Build classifiers to distinguish human vs. agent interactions
5. **Evaluate humanization**: Quantify effectiveness of humanization strategies
6. **Visualize results**: Generate plots for exploration and reporting

## Prerequisites

### Python Dependencies

```bash
pip install -r ../requirements.txt
```

Required packages: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `seaborn`, `xgboost`

### Data Requirements

Analysis requires data from `../logs/` directory. Download and unzip files from the Hugging Face dataset page. 
The following files are expected for feature analysis:
- `gesture_recording_*.log`: Raw touch events
- `Formated_Data_Renamed.xlsx`: Datasheet containing collected session timestamps. If new data is collected, update this file with session timestamps from `tasks.csv`. 

And optionally for statistics and plotting:
- `sensor_recording_*.txt`: Sensor data

**Note:** the files can be put anywhere in any sub-sub-... directory under `../logs/` as long as the glob pattern matches. For example, `../logs/**/gesture_recording_*.log` will also work. The file finder in the analysis scripts will search recursively under `../logs/` for matching files.

Also, ensure you have the pre-processed swipe data for human-like swipe generation:
- Pickled swipe data: `analysis/processing/swipe_data.pkl` (for human-like swipe generation)

## Directory Structure

```
analysis/
├── lib/                          # Core libraries
│   ├── motionevent_classes.py    # Event data structures
│   ├── gesture_log_reader_utils.py  # Parse gesture logs
│   ├── sensor_log_reader_utils.py   # Parse sensor logs
│   └── feature_library.py        # Feature extraction functions
├── processing/                   # Data processing scripts
│   ├── fit_effort_provider.py    # Human-like swipe generator
│   ├── extract_feature_of_swipes.py  # Swipe feature extraction
│   ├── tap_duration_extract.py   # Tap duration analysis
│   ├── interval_extract.py       # Temporal interval extraction
│   ├── interval_judge.py         # Interval classification
│   ├── judge_library.py          # Classification utilities
│   ├── calculate_roc_auc_from_feature.py  # calculation of various statistics, not restricted to ROC/AUC
│   └── generate_humanized_swipe_and_test_feature.py  # Humanization testing
├── plotting/                     # Visualization tools
│   ├── draw_motion_event.py      # Plot touch events
│   ├── draw_motion_event2.py     # Alternative motion event plot
│   ├── draw_motion_event_multi_file.py  # Multi-file comparison
│   ├── draw_sensor_event.py      # Plot sensor data
│   └── compare_sensor_with_gesture.py   # Combined visualization
└── swipe_data.pkl                # Pre-processed human swipe data (metadata)
```

## Quick Start

### Using Jupyter Notebook (Recommended)

The main analysis interface is `../analysis_playground.ipynb`. This notebook contains sections for:

1. **Data Loading**: Load and parse gesture/sensor logs
2. **Feature Extraction**: Extract kinematic features from swipes and taps
3. **Visualization**: Plot motion events, sensor data, and comparisons
5. **Humanization Testing**: Apply offline humanization strategies and evaluate effectiveness
6. **Statistics Analysis**: Compute numerical scores for humanization effectiveness (e.g., accuracy reduction)
4. **Classification**: Train threshold and SVM/XGBoost detectors to distinguish human vs. agent


**To use**: Open the notebook `analysis_playground.ipynb` in VS Code and run through the sections sequentially.

### Command-Line Analysis（May be deprecated）

For batch processing, use the scripts in `processing/`:

```bash
# Extract features from swipe data
python processing/extract_feature_of_swipes.py \
    --input-glob "../logs/gesture_recording_*.log" \
    --output features.csv

# Calculate ROC/AUC from extracted features
python processing/calculate_roc_auc_from_feature.py \
    --features features.csv \
    --output roc_results.csv

# Generate humanized swipes and test features
python processing/generate_humanized_swipe_and_test_feature.py \
    --input ../logs/gesture_recording_20260219_103045.log \
    --output humanized_test_results.csv
```

## Core Libraries Explained

### lib/motionevent_classes.py

Data structures for events:

```python
from analysis.lib.motionevent_classes import GotEvent, FingerEvent

# GotEvent: Raw event from getevent
event = GotEvent(
    timestamp_us=123456789,
    device="/dev/input/event4",
    type=3,      # EV_ABS
    code=53,     # ABS_MT_POSITION_X
    value=540    # x-coordinate
)

# FingerEvent: Abstract finger position
finger = FingerEvent(
    timestamp_us=123456789,
    x=540,
    y=960
)
```

### lib/gesture_log_reader_utils.py

Parse gesture logs:

```python
from analysis.lib.gesture_log_reader_utils import parse_gesture_log

# Parse a single log file
events = parse_gesture_log("../logs/gesture_recording_20260219_103045.log") # nonexistent llm hallucinated function

# Get swipes (finger movements)
swipes = events.get_swipes() # nonexistent llm hallucinated function


# Get taps (single touch points)
taps = events.get_taps() # nonexistent llm hallucinated function

```

### lib/sensor_log_reader_utils.py

Parse sensor logs:

```python
from analysis.lib.sensor_log_reader_utils import parse_sensor_log

# Parse sensor data
sensor_data = parse_sensor_log("../logs/sensor_recording_20260219_103045.txt")

# Access different sensors
accel_data = sensor_data.accelerometer  # (timestamp, x, y, z)
gyro_data = sensor_data.gyroscope       # (timestamp, x, y, z)
mag_data = sensor_data.magnetometer     # (timestamp, x, y, z)
```

### lib/feature_library.py

Extract features from interactions:

```python
from analysis.lib.feature_library import extract_swipe_features

# Extract features from a swipe
features = extract_swipe_features(swipe_events)

# Features include:
# - Path length
# - Average speed
# - Maximum speed
# - Acceleration variance
# - Curvature
# - Jerk (rate of acceleration change)
# - And many more (see feature_library.py for full list)
```

## Processing Scripts Explained

### fit_effort_provider.py

Generates human-like swipes using historical human data:

```python
from analysis.processing.fit_effort_provider import FitEffortProvider

# Load pre-trained provider
provider = FitEffortProvider.load("../metadata/analysis/processing/swipe_data.pkl")

# Generate human-like swipe from (x1,y1) to (x2,y2)
human_swipe = provider.fit(x1=100, y1=500, x2=900, y2=500)

# Convert to event sequence for fake_adb
from agent_tools.fake_adb.adb_wrapper import MotionGenerator
event_sequence = MotionGenerator.swipe_to_event_trace(human_swipe)
```

### extract_feature_of_swipes.py

Batch extract features from multiple log files:

```bash
python processing/extract_feature_of_swipes.py \
    --input-glob "../logs/gesture_recording_*.log" \
    --output features.csv \
    --label-column "agent_type" \
    --label-value "ui_tars"
```

Output CSV contains feature vectors for each swipe.

### tap_duration_extract.py

Analyze tap durations:

```bash
python processing/tap_duration_extract.py \
    --input "../logs/gesture_recording_*.log" \
    --output tap_durations.csv
```

### interval_extract.py

Extract temporal intervals between actions:

```bash
python processing/interval_extract.py \
    --input "../logs/gesture_recording_*.log" \
    --output intervals.csv
```

### calculate_roc_auc_from_feature.py

Compute ROC curves and AUC scores for feature-based classification:

```bash
python processing/calculate_roc_auc_from_feature.py \
    --features features.csv \
    --label-column "is_human" \
    --output roc_results.csv
```

## Plotting Tools

### draw_motion_event.py

Visualize touch events:

```bash
python plotting/draw_motion_event.py \
    --input ../logs/gesture_recording_20260219_103045.log \
    --output motion_plot.png
```

### draw_sensor_event.py

Plot sensor data over time:

```bash
python plotting/draw_sensor_event.py \
    --input ../logs/sensor_recording_20260219_103045.txt \
    --output sensor_plot.png
```

### compare_sensor_with_gesture.py

Overlay sensor and gesture data:

```bash
python plotting/compare_sensor_with_gesture.py \
    --gesture-log ../logs/gesture_recording_20260219_103045.log \
    --sensor-log ../logs/sensor_recording_20260219_103045.txt \
    --output combined_plot.png
```

### draw_motion_event_multi_file.py

Compare multiple recordings:

```bash
python plotting/draw_motion_event_multi_file.py \
    --input-glob "../logs/gesture_recording_*.log" \
    --labels "human1,agent1,human2,agent2" \
    --output comparison_plot.png
```

## Manual Classification Pipeline

Please directly use the Jupyter notebook `analysis_playground.ipynb` for better experience. If so, you can skip this section.

### 1. Extract Features

```python
from analysis.lib.gesture_log_reader_utils import parse_gesture_log
from analysis.lib.feature_library import extract_swipe_features
import pandas as pd

# Parse multiple files
features_list = []
for log_file in log_files:
    events = parse_gesture_log(log_file)
    for swipe in events.get_swipes():
        features = extract_swipe_features(swipe)
        features["label"] = "human" if "human" in log_file else "agent"
        features_list.append(features)

df = pd.DataFrame(features_list)
df.to_csv("features.csv", index=False)
```

### 2. Train Detector

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load features
df = pd.read_csv("features.csv")

# Split data
X = df.drop("label", axis=1)
y = (df["label"] == "human").astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate
y_pred_proba = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc:.4f}")
```

### 3. Feature Importance

```python
import matplotlib.pyplot as plt

# Get feature importances
importances = clf.feature_importances_
feature_names = X.columns

# Plot
plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances)
plt.yticks(range(len(importances)), feature_names)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
```

## Humanization Strategy Evaluation

### Path Transformation

Apply mathematical transformations to agent paths:

```python
from analysis.processing.fit_effort_provider import bot_line_fit

# Generate bot-like (linear) swipe
bot_swipe = bot_line_fit(
    x1=100, y1=500, x2=900, y2=500,
    duration_us=500000,
    neighbor_time_delta_us=11000
)

# Extract features
bot_features = extract_swipe_features(bot_swipe)

# Compare with human features
print(f"Bot path length: {bot_features['path_length']}")
```

### Temporal Adjustment

Adjust tap durations and action intervals to match human distributions:

```python
import numpy as np

# Sample from human tap duration distribution
human_tap_durations = np.array([...])  # Load from human data
sampled_duration = np.random.choice(human_tap_durations)

# Use sampled duration for agent taps
MotionGenerator.tap(adb_path, x, y, duration_us=sampled_duration)
```

### Fake Action Injection

Inject non-functional gestures to mask temporal patterns:

```python
# In agent loop, generate fake actions at random intervals
import threading
import time
import random

def fake_action_loop():
    while True:
        time.sleep(random.expovariate(1.0 / 1.1))  # Mean interval 1.1s
        MotionGenerator.custom_fake_action_3(adb_path)

thread = threading.Thread(target=fake_action_loop, daemon=True)
thread.start()
```

## Common Issues

### Missing swipe_data.pkl

**Symptoms**: `FileNotFoundError: swipe_data.pkl`

**Solution**: Copy from metadata.

### Feature Extraction Fails

**Symptoms**: NaN values or errors during extraction

**Solutions**:
1. Verify log file format is correct
2. Check for corrupted log files
3. Ensure sufficient events (minimum 2 points for swipe)

### Plot Rendering Issues

**Symptoms**: Blank plots or errors

**Solutions**:
1. Check matplotlib backend: `matplotlib.use('Agg')` for headless
2. Verify input files exist and are readable
3. Increase DPI for large datasets: `plt.savefig("plot.png", dpi=300)`

## Next Steps

1. **Explore data**: Run through `analysis_playground.ipynb` sections
2. **Extract features**: Use `extract_feature_of_swipes.py` on your data
3. **Train detectors**: Build classifiers to distinguish human vs. agent
4. **Test humanization**: Apply strategies and measure AUC reduction
5. **Visualize results**: Generate plots for reporting

See the main repository README for links to all documentation.
