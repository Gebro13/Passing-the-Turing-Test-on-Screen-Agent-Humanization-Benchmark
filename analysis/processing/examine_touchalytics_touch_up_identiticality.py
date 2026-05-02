"""cat data.csv | head -n 20 ; there's no column name in the .csv file.
0,36,3,1334893336544,0,1,272,269,0.21,0.04444445,0.0
0,36,3,1334893336790,2,1,262,271,0.32,0.04444445,0.0
0,36,3,1334893336795,2,1,123,327,0.28,0.04444445,0.0
0,36,3,1334893336800,1,1,123,327,0.28,0.04444445,0.0 # look at this
0,36,3,1334893336885,0,1,216,298,0.34,0.04444445,0.0 
0,36,3,1334893336907,2,1,216,298,0.38,0.04444445,0.0
0,36,3,1334893336926,2,1,147,312,0.59999996,0.0888889,0.0
0,36,3,1334893336949,2,1,115,328,0.63,0.04444445,0.0
0,36,3,1334893336952,2,1,98,336,0.17,0.13333336,0.0
0,36,3,1334893336971,1,1,98,336,0.17,0.13333336,0.0
0,36,3,1334893337798,0,1,108,339,0.48,0.04444445,0.0
0,36,3,1334893337814,2,1,118,337,0.48,0.04444445,0.0
0,36,3,1334893337820,2,1,135,333,0.48,0.04444445,0.0
0,36,3,1334893337830,2,1,158,330,0.35999998,0.04444445,0.0
0,36,3,1334893337907,2,1,267,344,0.14999999,0.04444445,0.0
0,36,3,1334893337909,1,1,267,344,0.14999999,0.04444445,0.0 # look at this
0,36,3,1334893338531,0,1,190,343,0.35999998,0.04444445,0.0
0,36,3,1334893338547,2,1,178,345,0.37,0.04444445,0.0
0,36,3,1334893338553,2,1,160,347,0.39999998,0.04444445,0.0
0,36,3,1334893338566,2,1,141,354,0.39999998,0.04444445,0.0
"""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

columns = [
    'phone_id', 'user_id', 'doc_id', 'time_ms', 'action',
    'phone_orientation', 'x', 'y', 'pressure', 'area_covered', 'finger_orientation'
]

if __name__ == "__main__":
    parser = ArgumentParser(description="Examine the identicality of touch events before and after action=1 (touch up) for TouchAlytics data.")
    parser.add_argument("--data_csv_path", type=str, required=True, help="Path to the CSV file containing touch event data.")
    args = parser.parse_args()
    data_csv_path: Path = Path(args.data_csv_path)

    df = pd.read_csv(data_csv_path, header=None, names=columns)

    compare_cols = [c for c in columns if c not in ('time_ms', 'action')]

    touch_up = df[df['action'] == 1]
    total_count = len(touch_up)
    warnings = 0
    for idx in touch_up.index:
        if idx == 0:
            print(f"WARNING row {idx}: action=1 is the first row, no previous row to compare")
            warnings += 1
            continue
        prev = df.loc[idx - 1, compare_cols]
        curr = df.loc[idx, compare_cols]
        if not prev.equals(curr):
            diff_cols = prev.index[prev != curr].tolist()
            print(f"WARNING row {idx}: action=1 differs from previous row in columns: {diff_cols}")
            print(f"  prev(time={df.loc[idx - 1, 'time_ms']}): {prev[diff_cols].to_dict()}")
            print(f"  curr(time={df.loc[idx, 'time_ms']}): {curr[diff_cols].to_dict()}")
            warnings += 1

    print(f"\nTotal action=1 rows: {len(touch_up)}, warnings: {warnings}, percentage: {warnings / total_count:.2%}  ")