import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, List
import numpy as np
import pandas as pd

#!/usr/bin/env python3
"""
Plot overlaid histograms of interval_us per type from a CSV file.

The CSV must contain:
- type: string labels
- interval_us: integer or numeric values

Example:
    python interval_judge.py --input extracted_intervals.csv --output interval_hist.png --bins auto --max-interval-us 200000
"""


import matplotlib.pyplot as plt


def parse_bins(value: str) -> Union[int, str]:
        """
        Parse bins argument. Accepts an integer or any valid numpy/matplotlib binning strategy:
        'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'
        """
        try:
                return int(value)
        except ValueError:
                return value  # pass through strategy strings


def parse_xlim(values: Optional[List[str]]) -> Optional[Tuple[float, float]]:
        if not values:
                return None
        if len(values) != 2:
                raise argparse.ArgumentTypeError("--xlim requires two numbers: min max")
        try:
                x0, x1 = float(values[0]), float(values[1])
        except ValueError as e:
                raise argparse.ArgumentTypeError(f"--xlim values must be numeric: {e}")
        if x1 <= x0:
                raise argparse.ArgumentTypeError("--xlim max must be greater than min")
        return (x0, x1)


def load_data(path: Path) -> pd.DataFrame:
        if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")
        df = pd.read_csv(path)
        required = {"type", "interval_us"}
        missing = required.difference(df.columns)
        if missing:
                raise ValueError(f"Missing required columns in CSV: {', '.join(sorted(missing))}")
        # Coerce types
        df["type"] = df["type"].astype(str)
        df["interval_us"] = pd.to_numeric(df["interval_us"], errors="coerce")
        df = df.dropna(subset=["interval_us"])
        if df.empty:
                raise ValueError("No valid rows after cleaning interval_us.")
        return df


def plot_histograms(
        df: pd.DataFrame,
        bins: Union[int, str],
        density: bool,
        figsize: Tuple[float, float],
        alpha: float,
        logx: bool,
        logy: bool,
        xlim: Optional[Tuple[float, float]],
        legend_loc: str,
        edgecolor: str,
):
        types = sorted(df["type"].unique())
        if len(types) == 0:
                raise ValueError("No types found to plot.")
        # Colors
        cmap = plt.get_cmap("tab20" if len(types) > 10 else "tab10")
        colors = [cmap(i % cmap.N) for i in range(len(types))]

        fig, ax = plt.subplots(figsize=figsize)

        for i, t in enumerate(types):
            vals = df.loc[df["type"] == t, "interval_us"].to_numpy()
            if vals.size == 0:
                continue
            ax.hist(
                vals,
                bins=bins,
                alpha=alpha,
                label=str(t),
                histtype="stepfilled",
                color=colors[i],
                edgecolor=edgecolor,
                linewidth=0.6,
                density=density,
            )

        ax.set_title("Interval distributions by type")
        ax.set_xlabel("interval_us")
        ax.set_ylabel("density" if density else "count")
        if xlim:
            ax.set_xlim(xlim)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.legend(title="type", loc=legend_loc, frameon=True)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        fig.tight_layout()
        return fig, ax


def main(argv=None) -> int:
        parser = argparse.ArgumentParser(description="Plot overlaid histograms of interval_us per type.")
        parser.add_argument("--input", "-i", type=Path, default=Path("extracted_intervals.csv"),
                                                help="Path to input CSV (default: extracted_intervals.csv)")
        parser.add_argument("--output", "-o", type=Path, default=Path("interval_hist.png"),
                                                help="Path to output image file (default: interval_hist.png)")
        parser.add_argument("--bins", type=parse_bins, default="auto",
                                                help="Histogram bins: integer or strategy (auto, fd, scott, rice, sturges, sqrt). Default: auto")
        parser.add_argument("--density", action="store_true",
                                                help="Plot probability density instead of counts.")
        parser.add_argument("--figsize", nargs=2, type=float, metavar=("W", "H"), default=(10.0, 6.0),
                                                help="Figure size in inches (width height). Default: 10 6")
        parser.add_argument("--alpha", type=float, default=0.5,
                                                help="Bar transparency for overlays. Default: 0.5")
        parser.add_argument("--logx", action="store_true", help="Log scale on X axis.")
        parser.add_argument("--logy", action="store_true", help="Log scale on Y axis.")
        parser.add_argument("--xlim", nargs=2, metavar=("MIN", "MAX"),
                                                help="Limit X axis range (two numbers).")
        parser.add_argument("--legend-loc", default="best",
                                                help="Legend location (e.g., best, upper right). Default: best")
        parser.add_argument("--dpi", type=int, default=150, help="Saved image DPI. Default: 150")
        parser.add_argument("--edgecolor", type=str, default="black", help="Bar edge color. Default: black")
        parser.add_argument("--max-interval-us", type=float, default=None, metavar="MAX_US",
                                                help="Discard rows with interval_us greater than this value. Default: no upper limit")

        args = parser.parse_args(argv)
        try:
                xlim = parse_xlim(args.xlim)
                df = load_data(args.input)

                if args.max_interval_us is not None:
                        df = df[df["interval_us"] <= args.max_interval_us]
                        if df.empty:
                                raise ValueError("No rows left after applying --max-interval-us filter.")

                fig, _ = plot_histograms(
                        df=df,
                        bins=args.bins,
                        density=args.density,
                        figsize=tuple(args.figsize),
                        alpha=args.alpha,
                        logx=args.logx,
                        logy=args.logy,
                        xlim=xlim,
                        legend_loc=args.legend_loc,
                        edgecolor=args.edgecolor,
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                print(f"Saved histogram to: {args.output}")
                return 0
        except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1


if __name__ == "__main__":
        raise SystemExit(main())