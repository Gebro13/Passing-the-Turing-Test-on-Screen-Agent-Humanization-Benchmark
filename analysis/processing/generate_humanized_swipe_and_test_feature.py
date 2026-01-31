# (540, 960) -> (540, 192)

from motionevent_classes import FingerEvent
from feature_library import extract_features
from draw_motion_event2 import throw_away_timestamp, plot_gestures
import numpy as np
from typing import List, Tuple
from fit_effort_provider import extract_exact_swipe_batch, FitEffortProvider
from gesture_log_reader_utils import filtered_gesture_generator_from_files_no_timestamp
from extract_feature_of_swipes import cleanse_into_swipe
from pathlib import Path

## plot the statistics into histogram
import matplotlib.pyplot as plt
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate human-like swipes and plot feature histogram.")
    parser.add_argument("--counts", type=int, default=200, help="Number of swipes to generate.")
    # add plot switch
    parser.add_argument("--plot_feature", action="store_true", default=False, help="Whether to plot the feature histogram.")
    parser.add_argument("--plot_swipes", action="store_true", default=False, help="Whether to plot the swipe traces.")
    parser.add_argument("--physical_output", action="store_true", default=False, help="Whether to operate the phone with these swipes.")
    parser.add_argument("--virtual_output_as_feature_type", type=str, default=None, help="What tag to output virtual features into a csv for copy pasting.")
    parser.add_argument("--source_label", type=str, default=None, help="What label to use for the source swipes.")
    parser.add_argument("--catalog_csv", type=str, default="logs_inf.csv", help="Path to the catalog CSV file.")
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()

    counts: int = args.counts

    example_swipe_batch = extract_exact_swipe_batch(
        args.source_label,
        filtered_gesture_generator_from_files_no_timestamp(
            pd.read_csv(args.catalog_csv),
            Path(args.log_dir),
            cleanse_into_swipe
        )
    )

    print("length of example_swipe_batch:", len(example_swipe_batch))

    humanize_sample_provider = FitEffortProvider(
        example_swipe_batch
    )
    humanize_sample_provider.dump_batches()

    target_swipe = ((540, 960), (540, 192))
    swipes: List[List[FingerEvent]] = [
        humanize_sample_provider.fit(target_swipe[0][0], target_swipe[0][1], target_swipe[1][0], target_swipe[1][1])
        # controller.MotionGenerator.generate_swipe_trace(540, 960, 540, 192, fake_human=True)
        for _ii in range(counts)
    ]


    swipes_features = [
        extract_features(swipe) for swipe in swipes
    ]

    if args.plot_feature:
        # Flatten the list of features
        feature_name: str = "avg_devs"
        feature_array = np.array([
            swipe_features[feature_name] for swipe_features in swipes_features
        ])


        # Plot the histogram
        plt.figure(figsize=(12, 6))
        plt.hist(feature_array, bins=30, alpha=0.7, color='blue')
        plt.title("Histogram of Swipe Features")
        plt.xlabel("Feature Value")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75)
        plt.show()
        plt.savefig(f"temp_histogram_{feature_name}.png")

    if args.plot_swipes:
        plot_gestures(throw_away_timestamp(swipes))
        plt.savefig("temp_swipes.png")

    if args.physical_output:
        raise NotImplementedError("Physical output is not implemented. Better write directly to a csv(? what about the timing?)")
        # controller.MotionGenerator.flush_event_sequence("adb", "")

    if isinstance(args.virtual_output_as_feature_type, str):
        df_feat = pd.DataFrame(swipes_features) # a df for each file, containing all features but no label
        df_feat.insert(0, "type", args.virtual_output_as_feature_type)
        cols = ["type"] + [c for c in df_feat.columns if c != "type"]
        out_path = "temp_ooooood_fit_attack_features.csv"
        df_feat[cols].to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
