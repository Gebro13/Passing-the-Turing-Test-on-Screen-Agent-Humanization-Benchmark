# from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Optional, Protocol, Tuple, TypedDict, Union, Set
import numpy.typing as npt
import argparse

import torch
import tqdm
from analysis.lib.motionevent_classes import SingularActionType, SessionType, SwipeFeaturedSessionType
from analysis.processing.extract_feature_of_swipes import build_features_dataframe
from sklearn.feature_selection import mutual_info_classif

# compute svm and xgboost on the filtered data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import sklearn.pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import random

# Paths and label configuration; the totality of input and output
DATA_DIR = Path(__file__).resolve().parent

def filter_df(df: pd.DataFrame, pos_label: str, neg_label: str) -> pd.DataFrame:
    """Filter df to keep only rows with type in {pos_label, neg_label}."""
    if "type" not in df.columns:
        raise ValueError("Input DataFrame must have a 'type' column.")
    filtered = df[df["type"].isin({pos_label, neg_label})].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered

def load_filtered_df(csv_path: Path, pos_label: str, neg_label: str) -> pd.DataFrame:
    """Load csv_path and keep only the two target classes.

    Returns a dataframe with original columns, filtered to rows where type is POS_LABEL or NEG_LABEL.
    """

    df = pd.read_csv(csv_path)
    filtered_df = filter_df(df, pos_label, neg_label)
    return filtered_df


def get_numeric_feature_column_names(df: pd.DataFrame) -> list[str]:
    """
    Return numeric feature columns (exclude the target column).
    
    :param df: this dataframe has a 'type' column followed by feature columns.
    :return: Description
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c != "type"]


def compute_auc_per_feature(filtered_df: pd.DataFrame, pos_label: str, neg_label: str) -> pd.DataFrame:
    """
        Compute ROC AUC for each numeric feature using the raw feature as the scoring function.
        automatically drops nan values.
        Returns a dataframe with columns: feature, auc, auc_oriented, n, n_pos, n_neg
    """
    features: List[str] = get_numeric_feature_column_names(filtered_df)
    results: List[Dict[str, Union[str, float, int]]] = []
    for feat in features:
        sub: pd.DataFrame = filtered_df[["type", feat]].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            continue
        # Ensure numpy arrays with explicit dtypes to avoid pandas ExtensionArray typing issues
        y: npt.NDArray[np.bool_] = (sub["type"] == pos_label).to_numpy(dtype=np.int8)
        # Need both classes present for a valid ROC
        if y.size == 0 or y.min() == y.max():
            continue
        scores: npt.NDArray[np.float64] = pd.to_numeric(sub[feat], errors="coerce").to_numpy(dtype=float)
        try:
            auc_val = roc_auc_score(y, scores)
        except ValueError:
            continue
        # Orientation-invariant AUC for sorting convenience
        auc_oriented = auc_val if auc_val >= 0.5 else 1 - auc_val
        results.append({
            "feature": feat,
            "auc": float(auc_val),
            "auc_oriented": float(auc_oriented),
            "n": int(len(scores)),
            "n_pos": int(y.sum()),
            "n_neg": int(len(y) - y.sum()),
        })
    res = pd.DataFrame(results)
    if not res.empty:
        res = res.sort_values(["auc_oriented", "auc"], ascending=[False, False]).reset_index(drop=True)
    return res


@dataclass
class BinaryClassificationMetrics:
    TP: int
    FP: int
    TN: int
    FN: int

    def accuracy(self) -> float:
        total = self.TP + self.FP + self.TN + self.FN
        if total == 0:
            raise ValueError("Total number of samples is zero in BinaryClassificationMetrics.")
        return (self.TP + self.TN) / total
    
    def precision(self) -> float:
        denom = self.TP + self.FP
        if denom == 0:
            raise ValueError("Precision is undefined when TP + FP is zero.")
        return self.TP / denom
    
    def recall_or_TPR(self) -> float:
        denom = self.TP + self.FN
        if denom == 0:
            raise ValueError("Recall is undefined when TP + FN is zero.")
        return self.TP / denom
        
    def FPR(self) -> float:
        denom = self.FP + self.TN
        if denom == 0:
            raise ValueError("FPR is undefined when FP + TN is zero.")
        return self.FP / denom

    def flexible_tpr(self) -> float:
        hard_tpr = self.recall_or_TPR()
        if (hard_tpr < 0.5):
            return 1 - hard_tpr
        else:
            return hard_tpr
        
    def flexible_fpr(self) -> float:
        hard_fpr = self.FPR()
        if (self.recall_or_TPR() < 0.5):
            return 1 - hard_fpr
        else:
            return hard_fpr

    # now add a simple summary for print(), print all metrics in several good tables.
    def summary(self) -> str:
        acc = self.accuracy()
        prec = self.precision()
        rec = self.recall_or_TPR()
        fpr = self.FPR()
        return (f"Accuracy: {acc:.4f}\n"
                f"Precision: {prec:.4f}\n"
                f"Recall/TPR: {rec:.4f}\n"
                f"FPR: {fpr:.4f}\n")


@dataclass
class ThresholdPosterior:
    """
    Dataclass representing posterior probabilities for a binary classification threshold.

    :param metrics: Binary classification metrics (TP, FP, TN, FN)
    :param threshold: The threshold value used for classification
    :ivar lesser_than_threshold_agent_log_odds: log(P(less than threshold | agent) / P(less than threshold | human))
    :ivar greater_than_threshold_agent_log_odds: log(P(greater than threshold | agent) / P(greater than threshold | human))
    """
    metrics: BinaryClassificationMetrics
    threshold: float
    lesser_than_threshold_agent_log_odds: float
    greater_than_threshold_agent_log_odds: float

    def __init__(self, metrics: BinaryClassificationMetrics, threshold: float):
        self.metrics = metrics
        self.threshold = threshold
        tpr = self.metrics.recall_or_TPR()
        fpr = self.metrics.FPR()
        self.lesser_than_threshold_agent_log_odds = float(np.log(tpr) - np.log(fpr))
        self.greater_than_threshold_agent_log_odds = float(np.log((1 - tpr) ) - np.log(1 - fpr))


def compute_acc_per_feature_using_max_precision(filtered_df: pd.DataFrame, pos_label: str, neg_label: str) -> Tuple[pd.DataFrame, Dict[str, ThresholdPosterior]]:
    """
    return the maximal accuracy of predicting pos_label.
    
    :param filtered_df: Description
    :type filtered_df: pd.DataFrame
    :param pos_label: Description
    :type pos_label: str
    :param neg_label: Description
    :type neg_label: str
    :return: Description
    :rtype: DataFrame
    """
    features: List[str] = get_numeric_feature_column_names(filtered_df)
    results: List[Dict[str, Union[str, float, int]]] = []

    # take out filtered_df with type being either pos_label or neg_label
    filtered_df = filter_df(df=filtered_df, pos_label=pos_label, neg_label=neg_label)

    if len(features) == 0:
        raise ValueError("No numeric features found in the DataFrame.")

    threshold_posteriors: Dict[str, ThresholdPosterior] = {}

    for feat in features:
        sub: pd.DataFrame = filtered_df[["type", feat]].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            raise ValueError("No valid data for feature {}".format(feat))


        y = (sub["type"] == pos_label).to_numpy(dtype=int)
        # Need both classes present
        if len(np.unique(y)) < 2:
            raise ValueError("Both classes must be present in the data.")
        
        raw_scores = pd.to_numeric(sub[feat], errors="coerce").to_numpy(dtype=float)

        precisions, recalls, thresholds = precision_recall_curve(y, raw_scores)
        if len(thresholds) == 0:
            raise ValueError("No thresholds found in precision-recall curve computation.")
        
        """
        # Calculate accuracy at each threshold
        accuracies = []
        for thresh in thresholds:
            y_pred = (raw_scores >= thresh).astype(int)
            acc = accuracy_score(y, y_pred)
            accuracies.append(acc)
        """

        # select the largest precision where recall is between 0.1 and 0.9
        valid_indices = np.where((recalls[:-1] >= 0.1) & (recalls[:-1] <= 0.9))[0]
        if len(valid_indices) == 0:
            raise ValueError("No valid indices found in the specified recall range.")
        max_acc = np.max(precisions[valid_indices])
        thresh = thresholds[valid_indices][np.argmax(precisions[valid_indices])]
        # pastbug: forgot to apply valid_indices to thresholds, end up with a mismatched thresh

        # renormalize: assume that n_pos turn out to be the equal number of n_neg. Thus, 
        n_pos = int(y.sum())
        n_neg = int(len(y) - y.sum())

        max_acc = (max_acc / n_pos) / (max_acc / n_pos +  (1 - max_acc) / n_neg)

        # note that the positive may be irrelevant to agent being 1 or 0; we just care about threshold_posterior which nihilates the discrepancy.
        # we assume that agent values are more likely to be above the threshold.
        metrics = BinaryClassificationMetrics(
            TP=np.sum((raw_scores >= thresh) & (y == 1)),
            FP=np.sum((raw_scores >= thresh) & (y == 0)),
            TN=np.sum((raw_scores < thresh) & (y == 0)),
            FN=np.sum((raw_scores < thresh) & (y == 1)),
        )

        threshold_posterior = ThresholdPosterior(metrics=metrics, threshold=float(thresh))
        threshold_posteriors[feat] = threshold_posterior

        results.append({
            "feature": feat,
            "acc": float(max_acc),
            "n": int(len(y)),
            "n_pos": n_pos,
            "n_neg": n_neg,
        })

    return pd.DataFrame(results), threshold_posteriors

def compute_acc_per_feature_using_thresholding(filtered_df: pd.DataFrame, pos_label: str, neg_label: str, threshold: float) -> Tuple[pd.DataFrame, Dict[str, ThresholdPosterior]]:
    """
       compute accuracy for each numeric feature using the raw feature as the scoring function and a fixed threshold.
    
    :param filtered_df: columns being "type" and numeric features
    :type filtered_df: pd.DataFrame
    :param pos_label: Description
    :type pos_label: str
    :param neg_label: Description
    :type neg_label: str
    :param threshold: Description
    :type threshold: float
    :return: Description
    :rtype: DataFrame
    """
    features: List[str] = get_numeric_feature_column_names(filtered_df)
    results: List[Dict[str, Union[str, float, int]]] = []

    # take out filtered_df with type being either pos_label or neg_label
    filtered_df = filter_df(df=filtered_df, pos_label=pos_label, neg_label=neg_label)

    threshold_posteriors: Dict[str, ThresholdPosterior] = {}
    for feat in features:
        sub: pd.DataFrame = filtered_df[["type", feat]].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            continue


        y = (sub["type"] == pos_label).to_numpy(dtype=int)
        # Need both classes present
        if len(np.unique(y)) < 2:
            continue
        
        raw_scores = pd.to_numeric(sub[feat], errors="coerce").to_numpy(dtype=float)

        
        # note that the positive may be irrelevant to agent being 1 or 0; we just care about threshold_posterior which nihilates the discrepancy
        y_pred = (raw_scores >= threshold).astype(int)
        acc = accuracy_score(y, y_pred)

        # Create BinaryClassificationMetrics object
        metrics = BinaryClassificationMetrics(
            TP=np.sum((y_pred == 1) & (y == 1)),
            FP=np.sum((y_pred == 1) & (y == 0)),
            TN=np.sum((y_pred == 0) & (y == 0)),
            FN=np.sum((y_pred == 0) & (y == 1)),
        )
        
        threshold_posterior = ThresholdPosterior(metrics=metrics, threshold=float(threshold))
        threshold_posteriors[feat] = threshold_posterior   

        results.append({
            "feature": feat,
            "acc": float(acc),
            "n": int(len(y)),
            "n_pos": int(y.sum()),
            "n_neg": int(len(y) - y.sum()),
        })

    res = pd.DataFrame(results)
    # if not res.empty:
    #     res = res.sort_values("acc", ascending=False).reset_index(drop=True)
    return res, threshold_posteriors


def compute_acc_per_feature_using_break_even_point(filtered_df: pd.DataFrame, pos_label: str, neg_label: str, plot_plt: bool = False) -> Tuple[pd.DataFrame, Dict[str, ThresholdPosterior]]:
    """
       compute accuracy for each numeric feature at the break-even point where precision=recall using the raw feature as the scoring function.
    
    :param filtered_df: columns being "type" and numeric features
    :type filtered_df: pd.DataFrame
    :param pos_label: Description
    :type pos_label: str
    :param neg_label: Description
    :type neg_label: str
    :return: Description
    """
    features: List[str] = get_numeric_feature_column_names(filtered_df)
    results: List[Dict[str, Union[str, float, int]]] = []

    # take out filtered_df with type being either pos_label or neg_label
    filtered_df = filter_df(df=filtered_df, pos_label=pos_label, neg_label=neg_label) # PASTBUG: forgot this line

    threshold_posteriors: Dict[str, ThresholdPosterior] = {}
    for feat in features:
        sub: pd.DataFrame = filtered_df[["type", feat]].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            continue


        y = (sub["type"] == pos_label).to_numpy(dtype=int)
        # Need both classes present
        if len(np.unique(y)) < 2:
            continue
        

        
        # plot sub by type as histogram to see the distribution
        if plot_plt:
            plt.figure()
            for label in sub["type"].unique():
                plt.hist(sub.loc[sub["type"] == label, feat], bins=30, alpha=0.5, label=str(label))
            plt.legend()
            plt.title(f"Histogram of feature {feat} by type")
            plt.show()

        raw_scores = pd.to_numeric(sub[feat], errors="coerce").to_numpy(dtype=float)
        best_acc = 0.0

        # Check both orientations: feature correlated with pos_label, or anti-correlated
        # We calculate the break-even point for both and take the best accuracy
        
        n_pos = int(y.sum())
        n_neg = int(len(y) - y.sum())
        precisions, recalls, thresholds = None, None, None
        for scores in [raw_scores, 1 - raw_scores]:
            try:
                precisions, recalls, thresholds = precision_recall_curve(y, scores)
                if len(thresholds) == 0:
                    continue
                
                # resurrect precisions and recalls so that each label has the same amount of points
                precisions = (precisions / n_pos) / (precisions / n_pos +  (1 - precisions) / n_neg)
                # recall doesn't need to be changed

                if plot_plt:
                    # plot the precision-recall curve and find the break-even point
                    # create new figure
                    plt.figure()
                    # plt.plot(recalls, precisions)
                    # plot scatter points instead
                    plt.scatter(recalls, precisions, s=0.1, alpha=1.0)
                    # input(f"Now comparing {pos_label} vs {neg_label} for feature {feat}. Press Enter to continue...")
                    # use that as title
                    plt.title(f"Precision-Recall Curve for feature {feat} with pos_label {pos_label} and neg_label {neg_label}")


                # Find index where |precision - recall| is minimized.
                # Note: precisions and recalls have length n_thresholds + 1.
                # We slice [:-1] to align with thresholds.

                # weed to only select the indices where recall is between 0.05 and 1.01
                indices = np.where((recalls[:-1] >= 0.05) & (recalls[:-1] <= 1.01))[0]
                if len(indices) == 0:
                    raise ValueError("No valid indices in the specified recall range.")

                precisions = precisions[indices]
                recalls = recalls[indices]
                thresholds = thresholds[indices]

                diffs = np.abs(precisions[:-1] - recalls[:-1])
                idx = np.argmin(diffs)
                
                
                thresh = thresholds[idx]

                print(thresh)

                y_pred = (scores >= thresh).astype(int)
                acc = accuracy_score(y, y_pred)
                
                if acc > best_acc:
                    best_acc = acc

                    # Create BinaryClassificationMetrics object
                    metrics = BinaryClassificationMetrics(
                        TP=np.sum((y_pred == 1) & (y == 1)),
                        FP=np.sum((y_pred == 1) & (y == 0)),
                        TN=np.sum((y_pred == 0) & (y == 0)),
                        FN=np.sum((y_pred == 0) & (y == 1)),
                    )
                    threshold_posterior = ThresholdPosterior(metrics=metrics, threshold=float(thresh))
                    threshold_posteriors[feat] = threshold_posterior
            except Exception:
                continue
        
        if best_acc >= 0.36 and best_acc <= 0.37:
            print(f"Feature {feat} has best accuracy {best_acc} at break-even point.")
            if plot_plt:
                # plot the precision-recall curve and find the break-even point
                # create new figure
                plt.figure()
                # plt.plot(recalls, precisions)
                # plot scatter points instead
                plt.scatter(recalls, precisions, s=0.1, alpha=1.0)
                # input(f"Now comparing {pos_label} vs {neg_label} for feature {feat}. Press Enter to continue...")
                # use that as title
                plt.title(f"Precision-Recall Curve for feature {feat} with pos_label {pos_label} and neg_label {neg_label}")


                plt.figure()
                for label in sub["type"].unique():
                    plt.hist(sub.loc[sub["type"] == label, feat], bins=30, alpha=0.5, label=str(label))
                plt.legend()
                plt.title(f"Histogram of feature {feat} by type")
                plt.show()

        results.append({
            "feature": feat,
            "acc": float(best_acc),
            "n": int(len(y)),
            "n_pos": n_pos,
            "n_neg": n_neg,
        })

    res = pd.DataFrame(results)
    # if not res.empty:
    #     res = res.sort_values("acc", ascending=False).reset_index(drop=True)
    return res, threshold_posteriors

def compute_acc_per_feature_using_x_1_x_point_on_roc_auc(filtered_df: pd.DataFrame, pos_label: str, neg_label: str) -> Tuple[pd.DataFrame, Dict[str, ThresholdPosterior]]:
    """
       compute accuracy for each numeric feature at the point on ROC curve where FPR is 1 - TPR using the raw feature as the scoring function.
    """
    features: List[str] = get_numeric_feature_column_names(filtered_df)
    results: List[Dict[str, Union[str, float, int]]] = []

    # take out filtered_df with type being either pos_label or neg_label
    filtered_df = filter_df(df=filtered_df, pos_label=pos_label, neg_label=neg_label)

    threshold_posteriors: Dict[str, ThresholdPosterior] = {}

    for feat in features:
        sub: pd.DataFrame = filtered_df[["type", feat]].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            continue


        y = (sub["type"] == pos_label).to_numpy(dtype=int)
        # Need both classes present
        if len(np.unique(y)) < 2:
            continue
        
        raw_scores = pd.to_numeric(sub[feat], errors="coerce").to_numpy(dtype=float)

        fpr, tpr, thresholds = roc_curve(y, raw_scores)
        if len(thresholds) == 0:
            continue
        
        # Find index where |fpr - (1 - tpr)| is minimized.
        diffs = np.abs(fpr - (1 - tpr))
        idx = np.argmin(diffs)
        
        
        thresh = thresholds[idx]

        # note that the positive may be irrelevant to agent being 1 or 0; we just care about threshold_posterior which nihilates the discrepancy
        y_pred = (raw_scores >= thresh).astype(int)
        acc = accuracy_score(y, y_pred)
        
        n_pos = int(y.sum())
        n_neg = int(len(y) - y.sum())
        acc = (acc / n_pos) / (acc / n_pos +  (1 - acc) / n_neg) # force an equal number of pos and neg
        acc = max(acc, 1 - acc)  # orientation invariant

        # Create BinaryClassificationMetrics object
        metrics = BinaryClassificationMetrics(
            TP=np.sum((y_pred == 1) & (y == 1)),
            FP=np.sum((y_pred == 1) & (y == 0)),
            TN=np.sum((y_pred == 0) & (y == 0)),
            FN=np.sum((y_pred == 0) & (y == 1)),
        )
        threshold_posterior = ThresholdPosterior(metrics=metrics, threshold=float(thresh))
        threshold_posteriors[feat] = threshold_posterior   

        results.append({
            "feature": feat,
            "acc": float(acc),
            "n": int(len(y)),
            "n_pos": int(y.sum()),
            "n_neg": int(len(y) - y.sum()),
        })

    res = pd.DataFrame(results)
    if not res.empty:
        res = res.sort_values("acc", ascending=False)
    return res, threshold_posteriors


def compute_average_precision_per_feature(filtered_df: pd.DataFrame, pos_label: str, neg_label: str) -> pd.DataFrame:
    """
        Compute average precision for each numeric feature using the raw feature as the scoring function.
        automatically drops nan values.
        Returns a dataframe with columns: feature, ap, n, n_pos, n_neg
    """

    features: List[str] = get_numeric_feature_column_names(filtered_df)
    filtered_df = filter_df(df=filtered_df, pos_label=pos_label, neg_label=neg_label)
    results: List[Dict[str, Union[str, float, int]]] = []
    for feat in features:
        sub: pd.DataFrame = filtered_df[["type", feat]].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            continue
        # Ensure numpy arrays with explicit dtypes to avoid pandas ExtensionArray typing issues
        y: npt.NDArray[np.bool_] = (sub["type"] == pos_label).to_numpy(dtype=np.int8)
        # Need both classes present for a valid AP
        if y.size == 0 or y.min() == y.max():
            continue
        scores: npt.NDArray[np.float64] = pd.to_numeric(sub[feat], errors="coerce").to_numpy(dtype=float)
        try:
            ap_val = average_precision_score(y, scores)
        except ValueError:
            continue

        results.append({
            "feature": feat,
            "acc": float(ap_val),
            "n": int(len(scores)),
            "n_pos": int(y.sum()),
            "n_neg": int(len(y) - y.sum()),
        })
    res = pd.DataFrame(results)
    if not res.empty:
        res = res.sort_values("acc", ascending=False).reset_index(drop=True)
    return res

def compute_acc_per_feature_using_svm(filtered_df: pd.DataFrame, pos_label: str, neg_label: str) -> pd.DataFrame:
    """
        Compute svm accuracy for each numeric feature using the raw feature as the scoring function.
        For each svm accuracy, randomly divide 80% as train set and 20% as test set. Then, test on the test set and obtain accuracy.
        You can refer to the latter calculate_svm_and_xgboost.
        automatically drops nan values.
        Returns a dataframe with columns: feature, acc, n, n_pos, n_neg
    """
    np.random.seed(42)
    features: List[str] = get_numeric_feature_column_names(filtered_df)
    results: List[Dict[str, Union[str, float, int]]] = []

    filtered_df = filter_df(df=filtered_df, pos_label=pos_label, neg_label=neg_label)

    for feat in features:
        # Prepare data for this specific feature
        sub: pd.DataFrame = filtered_df[["type", feat]].replace([np.inf, -np.inf], np.nan).dropna()
        
        if sub.empty:
            continue
            
        # Create X (feature) and y (labels)
        # X needs to be 2D for sklearn
        X = pd.to_numeric(sub[feat], errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
        y = (sub["type"] == pos_label).to_numpy(dtype=int)
        
        # Check if we have enough data and both classes
        if len(y) < 5 or len(np.unique(y)) < 2:
            continue
            
        # Check if we have enough samples per class for splitting (at least 2 per class ideally)
        if (pd.Series(y).value_counts().min() < 2):
            continue

        try:
            # Split data: 80% train, 20% test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train SVM
            # Using a pipeline with StandardScaler is good practice even for single features
            svm_clf = make_pipeline(StandardScaler(), SVC(C=1,kernel="linear", probability=False, random_state=42)) # pastbug: used rbf, which is pointless for single feature
            svm_clf.fit(X_train, y_train)
            
            # Predict and calculate accuracy
            y_pred = svm_clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            results.append({
                "feature": feat,
                "acc": float(acc),
                "n": int(len(y)),
                "n_pos": int(y.sum()),
                "n_neg": int(len(y) - y.sum()),
            })
        except Exception:
            # Catch errors during fitting/splitting (e.g. if stratify fails due to small class size)
            continue

    res = pd.DataFrame(results)
    if not res.empty:
        res = res.sort_values("acc", ascending=False).reset_index(drop=True)
    return res

def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

class SVMAndXGBoostResult(TypedDict):
    svm_accuracy: Optional[float]
    xgb_accuracy: Optional[float]
    svm_test_metrics: BinaryClassificationMetrics
    xgb_test_metrics: BinaryClassificationMetrics

def get_feature_columns(filtered: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Docstring for get_feature_columns
    
    :param filtered: this dataframe has a 'type' column followed by feature columns.
    :param feature_cols: names that exclude `type` column
    :return: the numeric feature columns only
    :rtype: DataFrame
    """
    return (
        filtered.loc[:, feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([float("inf"), float("-inf")], pd.NA)
        .fillna(0.0)
    )

def calculate_mutual_information_binary(features_csv: pd.DataFrame, pos_label: str, neg_label: str) -> pd.DataFrame:
    """
    This function computes mutual information between each feature and the binary label (pos_label vs neg_label).
    
    :param features_csv: this dataframe has a 'type' column followed by feature columns.
    :param pos_label: The label use as y=1
    :param neg_label: The label use as y=0
    :return: A dataframe with columns: feature, mutual_information, n, n_pos, n_neg
    """

    filtered = filter_df(df=features_csv, pos_label=pos_label, neg_label=neg_label)

    feature_cols = get_numeric_feature_column_names(filtered)
    X = get_feature_columns(filtered, feature_cols)
    y = (filtered["type"].astype(str) == pos_label).astype(int)

    if X.empty or y.nunique() < 2:
        return pd.DataFrame(columns=["feature", "mutual_information", "n", "n_pos", "n_neg"])

    mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)

    results: List[Dict[str, Union[str, float, int]]] = []
    for feat, mi in zip(feature_cols, mi_scores):
        results.append({
            "feature": feat,
            "mutual_information": float(mi),
            "n": int(len(y)),
            "n_pos": int(y.sum()),
            "n_neg": int(len(y) - y.sum()),
        })

    res = pd.DataFrame(results)
    if not res.empty:
        res = res.sort_values("mutual_information", ascending=False).reset_index(drop=True)
    return res

def calculate_mutual_information_multiclass(features_csv: pd.DataFrame, classses_to_consider: List[str], output_relative_importance: bool) -> pd.DataFrame:

    """
    This function computes mutual information between each feature and the multi-class label.
    
    :param features_csv: this dataframe has a 'type' column followed by feature columns.
    :param classses_to_consider: List of class labels to consider
    :param output_relative_importance: Whether to output relative importance
    :return: A dataframe with columns: feature, mutual_information, n, n_classes
    """

    filtered = features_csv[features_csv["type"].isin(classses_to_consider)].copy()
    filtered.reset_index(drop=True, inplace=True)

    feature_cols = get_numeric_feature_column_names(filtered)
    X = get_feature_columns(filtered, feature_cols)
    y = filtered["type"].astype(str)

    if X.empty or y.nunique() < 2:
        return pd.DataFrame(columns=["feature", "mutual_information", "n", "n_classes"])

    mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    if output_relative_importance:
        # actually this is somewhat problematic as it does not consider the correlation between features
        total_mi = np.sum(mi_scores)
        if total_mi > 0:
            mi_scores = mi_scores / total_mi

    results: List[Dict[str, Union[str, float, int]]] = []
    for feat, mi in zip(feature_cols, mi_scores):
        results.append({
            "feature": feat,
            "mutual_information": float(mi),
            "n": int(len(y)),
            "n_classes": int(y.nunique()),
        })

    res = pd.DataFrame(results)
    if not res.empty:
        res = res.sort_values("mutual_information", ascending=False).reset_index(drop=True)
    return res

COMPLICATED_MODEL_TEST_SET_SIZE = 0.3

def calculate_svm_and_xgboost(features_csv: pd.DataFrame, pos_label: str, neg_label: str) -> Tuple[SVMAndXGBoostResult, Optional[sklearn.pipeline.Pipeline], Optional[XGBClassifier]]:
    """
    This function computes SVM and XGBoost classification accuracies on the provided features dataframe for the given positive and negative labels, excluding unmentioned labels.
    
    :param features_csv: this dataframe has a 'type' column followed by feature columns.
    :param pos_label: The label use as y=1
    :param neg_label: The label use as y=0
    :return: A tuple containing:
        - A dictionary with SVM and XGBoost accuracies.
        - The trained SVM pipeline (or None if training was not possible).
        - The trained XGBoost classifier (or None if training was not possible).
            - The classifier is trained on 70% of the data and tested on 30%.
            - The prediction is 1 for pos lavel, and 0 for neg label.
    """
    filtered = filter_df(df=features_csv, pos_label=pos_label, neg_label=neg_label)

    
    # prepare features and labels
    feature_cols = get_numeric_feature_column_names(filtered)
    X = get_feature_columns(filtered, feature_cols)
    y = (filtered["type"].astype(str) == pos_label).astype(int)
    # normalize X here.
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    

    if not X.empty and y.nunique() == 2:
        if (y.value_counts().min() < 2):
            return {"svm_accuracy": None, "xgb_accuracy": None}, None, None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=COMPLICATED_MODEL_TEST_SET_SIZE, random_state=42, stratify=y
        )
        # ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
        
        
        # SVM with scaling
        svm_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42))
        svm_clf.fit(X_train, y_train)
        y_pred_svm = svm_clf.predict(X_test)
        svm_acc: float = float(accuracy_score(y_test, y_pred_svm))
        svm_test_metrics = BinaryClassificationMetrics(
            TP=np.sum((y_pred_svm == 1) & (y_test == 1)),
            FP=np.sum((y_pred_svm == 1) & (y_test == 0)),
            TN=np.sum((y_pred_svm == 0) & (y_test == 0)),
            FN=np.sum((y_pred_svm == 0) & (y_test == 1)),
        )

        # XGBoost
        xgb_clf = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=4,
            random_state=42,
            eval_metric="logloss",
        )
        xgb_clf.fit(X_train, y_train)
        y_pred_xgb = xgb_clf.predict(X_test)
        xgb_acc: float = float(accuracy_score(y_test, y_pred_xgb))
        xgb_test_metrics = BinaryClassificationMetrics(
            TP=np.sum((y_pred_xgb == 1) & (y_test == 1)),
            FP=np.sum((y_pred_xgb == 1) & (y_test == 0)),
            TN=np.sum((y_pred_xgb == 0) & (y_test == 0)),
            FN=np.sum((y_pred_xgb == 0) & (y_test == 1)),
        )
        
        result: SVMAndXGBoostResult = {"svm_accuracy": svm_acc, "xgb_accuracy": xgb_acc, "svm_test_metrics": svm_test_metrics, "xgb_test_metrics": xgb_test_metrics}
        return result, svm_clf, xgb_clf
    else:
        return {"svm_accuracy": None, "xgb_accuracy": None, "svm_test_metrics": None, "xgb_test_metrics": None}, None, None


class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 10, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch, max_seq_len, input_size)
        :param lengths: (batch,) actual sequence lengths
        :return: (batch,) logits
        """
        packed: torch.nn.utils.rnn.PackedSequence = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers, batch, hidden_size) → use last layer
        out = self.fc(h_n[-1])  # (batch, 1)
        return out.squeeze(-1)

@dataclass
class LSTMClassificationResult:
    metrics: BinaryClassificationMetrics
    # use pytorch model
    model: Optional[torch.nn.Module]

def classify_using_lstm(
    input_vec_len: int,
    predictors: List[SwipeFeaturedSessionType],
    positive_labeled: List[bool]
) -> LSTMClassificationResult:
    """
    This function trains an LSTM-based binary classifier on the provided swipe session data and returns the classification metrics and the trained model.
    
    :param predictors: A list of swipe session data, where each session is represented as a sequence of feature vectors (SwipeFeaturedSessionType).
    :param positive_labeled: A list of boolean labels corresponding to each session, where True indicates a positive label and False indicates a negative label.
    :return: An LSTMClassificationResult containing the binary classification metrics and the trained LSTM model.
    """
    assert len(predictors) == len(positive_labeled), "Predictors and labels must have the same length."
    for session in predictors:
        assert session.features.shape[1] == input_vec_len, f"Each feature vector must have length {input_vec_len} instead of {session.features.shape[1]}."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert each session's DataFrame to numpy
    sequences = [session.features.to_numpy(dtype=np.float64) for session in predictors]
    labels = np.array(positive_labeled, dtype=np.float32)
    lengths = np.array([seq.shape[0] for seq in sequences], dtype=np.int64)

    # Global normalization across all timesteps
    all_features = np.concatenate(sequences, axis=0)  # (total_timesteps, input_vec_len)
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0)
    std[std == 0] = 1.0 # warning: if a feature has zero variance, we set std to 1 to avoid division by zero. This effectively leaves that feature unchanged after normalization.
    sequences = [(seq - mean) / std for seq in sequences]

    # Pad to max length
    max_len = max(seq.shape[0] for seq in sequences)
    padded = np.zeros((len(sequences), max_len, input_vec_len), dtype=np.float32)
    for i, seq in enumerate(sequences):
        padded[i, :seq.shape[0], :] = seq

    # Train/test split
    indices = np.arange(len(predictors))
    train_idx, test_idx = train_test_split(
        indices, test_size=COMPLICATED_MODEL_TEST_SET_SIZE, random_state=42, stratify=labels
    )

    X_train = torch.tensor(padded[train_idx], dtype=torch.float32, device=device)
    X_test = torch.tensor(padded[test_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(labels[train_idx], dtype=torch.float32, device=device)
    y_test = torch.tensor(labels[test_idx], dtype=torch.float32, device=device)
    # raise an error if y_test is all 0 or all 1, as that would make accuracy meaningless
    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        raise ValueError("Test set has only one class present, cannot evaluate accuracy.")
    else:
        print(f"Test set has {y_test.sum().item()} positive samples and {len(y_test) - y_test.sum().item()} negative samples.")

    len_train = torch.tensor(lengths[train_idx], dtype=torch.long)
    len_test = torch.tensor(lengths[test_idx], dtype=torch.long)

    # Model, optimizer, loss
    model = LSTMClassifier(input_size=input_vec_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training
    n_epochs = 100
    batch_size = 32
    n_train = len(train_idx)
    model.train()
    for epoch in tqdm.trange(n_epochs):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            batch_idx = perm[start:start + batch_size]
            logits = model(X_train[batch_idx], len_train[batch_idx])
            loss = criterion(logits, y_train[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate the lstm
    model.eval()
    with torch.no_grad():
        logits = model(X_test, len_test)
        preds = (logits >= 0.0).long()
        y_true = y_test.long()
        tp = int(((preds == 1) & (y_true == 1)).sum().item())
        fp = int(((preds == 1) & (y_true == 0)).sum().item())
        tn = int(((preds == 0) & (y_true == 0)).sum().item())
        fn = int(((preds == 0) & (y_true == 1)).sum().item())

    metrics = BinaryClassificationMetrics(TP=tp, FP=fp, TN=tn, FN=fn)
    return LSTMClassificationResult(metrics=metrics, model=model)

@dataclass
class ARIMAClassificationResult:
    metrics: BinaryClassificationMetrics
    svm_pipeline: Optional[sklearn.pipeline.Pipeline]

ARIMA_ORDER = (2, 0, 1)  # AR(2), no differencing, MA(1)

def _extract_arima_features_for_session(
    session_matrix: npt.NDArray[np.float64],
    order: Tuple[int, int, int] = ARIMA_ORDER,
) -> npt.NDArray[np.float64]:
    """
    Fit ARIMA(p,d,q) independently to each feature column of a session matrix
    and return a fixed-length feature vector summarizing the temporal dynamics.
    
    For each of the D feature dimensions, extracts:
      - p AR coefficients
      - q MA coefficients
      - sigma2 (residual variance)
      - series mean
    Total output length = D * (p + q + 2).
    
    :param session_matrix: (T, D) array, T timesteps, D features. Already normalized.
    :param order: (p, d, q) ARIMA order
    :return: 1-D array of length D * (p + q + 2)
    """
    from statsmodels.tsa.arima.model import ARIMA
    import warnings

    p, d, q = order
    n_features = session_matrix.shape[1]
    vec_per_dim = p + q + 2  # AR coeffs + MA coeffs + sigma2 + mean
    out = np.zeros(n_features * vec_per_dim, dtype=np.float64)

    for dim_i in range(n_features):
        series = session_matrix[:, dim_i].astype(np.float64)
        offset = dim_i * vec_per_dim
        out[offset + p + q + 1] = float(np.mean(series))  # always store mean

        if len(series) < p + d + q + 2:
            # too short to fit; leave AR/MA/sigma as zeros, mean is set
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(series, order=order, enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(method_kwargs={"maxiter": 50})
            out[offset: offset + p] = res.arparams[:p]
            out[offset + p: offset + p + q] = res.maparams[:q]
            out[offset + p + q] = float(res.sigma2) if np.isfinite(res.sigma2) else 0.0
        except Exception:
            pass  # zeros remain as fallback

    # replace any non-finite values
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def classify_using_arima(
    input_vec_len: int,
    predictors: List[SwipeFeaturedSessionType],
    positive_labeled: List[bool],
    order: Tuple[int, int, int] = ARIMA_ORDER,
) -> ARIMAClassificationResult:
    """
    Fit ARIMA per feature dimension per session to extract fixed-length parameter vectors,
    then classify with an RBF-SVM. Analogous to LSTM→linear, but ARIMA→SVM.
    
    :param input_vec_len: number of feature columns per timestep
    :param predictors: list of sessions, each with a .features DataFrame (rows=timesteps, cols=features)
    :param positive_labeled: boolean labels per session
    :param order: (p, d, q) ARIMA order to use
    :return: ARIMAClassificationResult with metrics and the trained SVM pipeline
    """
    assert len(predictors) == len(positive_labeled), "Predictors and labels must have the same length."

    # Normalize globally across all timesteps
    sequences = [session.features.to_numpy(dtype=np.float64) for session in predictors]
    all_features = np.concatenate(sequences, axis=0)
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0)
    std[std == 0] = 1.0
    sequences = [(seq - mean) / std for seq in sequences]

    # Extract ARIMA parameter vectors
    print(f"Extracting ARIMA{order} features for {len(sequences)} sessions...")
    arima_features = np.array([
        _extract_arima_features_for_session(seq, order=order)
        for seq in tqdm.tqdm(sequences, desc="ARIMA fitting")
    ])  # (n_sessions, n_features * (p + q + 2))

    labels = np.array(positive_labeled, dtype=np.int32)

    # Train/test split
    if np.unique(labels).size < 2 or np.bincount(labels).min() < 2:
        raise ValueError("Need at least 2 samples per class for stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        arima_features, labels,
        test_size=COMPLICATED_MODEL_TEST_SET_SIZE, random_state=42, stratify=labels,
    )

    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        raise ValueError("Test set has only one class present, cannot evaluate accuracy.")
    print(f"Test set has {y_test.sum()} positive samples and {len(y_test) - y_test.sum()} negative samples.")

    # SVM classifier on the ARIMA-derived features
    svm_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42))
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)

    tp = int(np.sum((y_pred == 1) & (y_test == 1)))
    fp = int(np.sum((y_pred == 1) & (y_test == 0)))
    tn = int(np.sum((y_pred == 0) & (y_test == 0)))
    fn = int(np.sum((y_pred == 0) & (y_test == 1)))

    metrics = BinaryClassificationMetrics(TP=tp, FP=fp, TN=tn, FN=fn)
    return ARIMAClassificationResult(metrics=metrics, svm_pipeline=svm_clf)


def make_feature_and_learner_table(task_cluster_id: int, method_name: str, 
                pos_iterator: Tuple[str, List[SingularActionType]], 
                neg_iterator: Tuple[str, List[SingularActionType]]) -> pd.DataFrame:
    result_csv = pd.DataFrame()

    features_csv = build_features_dataframe([pos_iterator, neg_iterator])
    filtered = filter_df(df=features_csv, pos_label=pos_iterator[0], neg_label=neg_iterator[0])
    res = compute_auc_per_feature(filtered_df=filtered, pos_label=pos_iterator[0], neg_label=neg_iterator[0])

    if not res.empty:
        res.set_index(keys="feature", inplace=True)
        auc_oriented = res[["auc_oriented"]] # use the named index as columns
        result_csv[[(task_cluster_id, method_name)]] = auc_oriented
        clf_res, svm_pipeline, xgb = calculate_svm_and_xgboost(features_csv=features_csv, pos_label=pos_iterator[0], neg_label=neg_iterator[0])
        print(clf_res)
        result_csv.loc["svm_accuracy", [(task_cluster_id, method_name)]] = clf_res["svm_accuracy"]
        result_csv.loc["xgb_accuracy", [(task_cluster_id, method_name)]] = clf_res["xgb_accuracy"]
        result_csv.loc["svm_tpr", [(task_cluster_id, method_name)]] = clf_res["svm_test_metrics"].recall_or_TPR()
        result_csv.loc["svm_fpr", [(task_cluster_id, method_name)]] = clf_res["svm_test_metrics"].FPR()
        result_csv.loc["xgb_tpr", [(task_cluster_id, method_name)]] = clf_res["xgb_test_metrics"].recall_or_TPR()
        result_csv.loc["xgb_fpr", [(task_cluster_id, method_name)]] = clf_res["xgb_test_metrics"].FPR()
    else:
        raise ValueError("Resulting AUC dataframe is empty.")

    return result_csv


class AccObtainFunction(Protocol):
    def __call__(self, filtered_df: pd.DataFrame, pos_label: str, neg_label: str) -> Tuple[pd.DataFrame, Dict[str, ThresholdPosterior]]: ...

def make_feature_and_learner_table_but_acc(task_cluster_id: int, method_name: str, 
                pos_iterator: Tuple[str, List[SingularActionType]], 
                neg_iterator: Tuple[str, List[SingularActionType]],
                acc_obtain_function: AccObtainFunction
                ) -> Tuple[pd.DataFrame, sklearn.pipeline.Pipeline, XGBClassifier]:
    result_csv = pd.DataFrame()

    features_csv = build_features_dataframe([pos_iterator, neg_iterator])
    filtered = filter_df(df=features_csv, pos_label=pos_iterator[0], neg_label=neg_iterator[0])
    res, threshold_posteriors = acc_obtain_function(filtered_df=filtered, pos_label=pos_iterator[0], neg_label=neg_iterator[0])

    if not res.empty:
        res.set_index(keys="feature", inplace=True)
        acc_col = res[["acc"]] # use the named index as columns
        result_csv[[(task_cluster_id, method_name)]] = acc_col
        clf_res, svm_pipeline, xgb = calculate_svm_and_xgboost(features_csv=features_csv, pos_label=pos_iterator[0], neg_label=neg_iterator[0])
        if svm_pipeline is None or xgb is None:
            raise ValueError("SVM or XGBoost pipeline could not be created due to insufficient data.")
        print(clf_res)
        result_csv.loc["svm_accuracy", [(task_cluster_id, method_name)]] = clf_res["svm_accuracy"]
        result_csv.loc["xgb_accuracy", [(task_cluster_id, method_name)]] = clf_res["xgb_accuracy"]
        result_csv.loc["svm_tpr", [(task_cluster_id, method_name)]] = clf_res["svm_test_metrics"].recall_or_TPR()
        result_csv.loc["svm_fpr", [(task_cluster_id, method_name)]] = clf_res["svm_test_metrics"].FPR()
        result_csv.loc["xgb_tpr", [(task_cluster_id, method_name)]] = clf_res["xgb_test_metrics"].recall_or_TPR()
        result_csv.loc["xgb_fpr", [(task_cluster_id, method_name)]] = clf_res["xgb_test_metrics"].FPR()

        return result_csv, svm_pipeline, xgb
    else:
        raise ValueError("Resulting ACC dataframe is empty.")


def plot_acc_increase_as_more_feature_used(task_cluster_id: int, method_name: str, pos_iterator: Tuple[str, List[SingularActionType]], neg_iterator: Tuple[str, List[SingularActionType]]) -> pd.DataFrame:
    result_csv = pd.DataFrame()

    features_csv = build_features_dataframe([pos_iterator, neg_iterator])
    filtered = filter_df(df=features_csv, pos_label=pos_iterator[0], neg_label=neg_iterator[0])

    feature_names = filtered.columns.to_list()
    feature_names.remove("type")
    total_feature_count = len(feature_names)
    each_sample_count = 5
    np.random.seed(42)
    for sample_feature_count in range(1, total_feature_count + 1):
        for iterer in range(each_sample_count):
            sample_feature = np.random.choice(feature_names, sample_feature_count, replace=False).tolist()
            selected_features = features_csv.loc[:, ["type"] + sample_feature]
            clf_res, svm_pipeline, xgb = calculate_svm_and_xgboost(features_csv=selected_features, pos_label=pos_iterator[0], neg_label=neg_iterator[0])
            result_csv.loc[f"svm_accuracy_{sample_feature_count}_{iterer}", [(task_cluster_id, method_name)]] = clf_res["svm_accuracy"]
            result_csv.loc[f"xgb_accuracy_{sample_feature_count}_{iterer}", [(task_cluster_id, method_name)]] = clf_res["xgb_accuracy"]
    return result_csv

def make_feature_table_but_mutual_information_binary(task_cluster_id: int, method_name: str, 
                pos_iterator: Tuple[str, List[SingularActionType]], 
                neg_iterator: Tuple[str, List[SingularActionType]]) -> pd.DataFrame:
    features_csv = build_features_dataframe([pos_iterator, neg_iterator])
    filtered = filter_df(df=features_csv, pos_label=pos_iterator[0], neg_label=neg_iterator[0])
    res = calculate_mutual_information_binary(filtered, pos_label=pos_iterator[0], neg_label=neg_iterator[0])
    if not res.empty:
        res.set_index(keys="feature", inplace=True)
        result_csv = pd.DataFrame()
        result_csv[[(task_cluster_id, method_name)]] = res[["mutual_information"]]
        return result_csv
    else:
        raise ValueError("Resulting mutual information dataframe is empty.")


def make_feature_table_but_mutual_information_multiple(
    task_cluster_id: int, 
    type_and_data_iterators: List[Tuple[str, List[SingularActionType]]],
    output_relative_importance: bool = True,
    ) -> pd.DataFrame:

    features_csv = build_features_dataframe(type_and_data_iterators)
    res = calculate_mutual_information_multiclass(
        features_csv=features_csv, 
        classses_to_consider=[t[0] for t in type_and_data_iterators], 
        output_relative_importance=output_relative_importance,
    )
    if not res.empty:
        res.set_index(keys="feature", inplace=True)
        result_csv = pd.DataFrame()
        result_csv[[(task_cluster_id, "mutual_information_multiclass")]] = res[["mutual_information"]]
        return result_csv
    else:
        raise ValueError("Resulting mutual information dataframe is empty.")

## Plotting helpers

CSV_PATH: Path = DATA_DIR / "label_and_features.csv"
POS_LABEL: str = "user3"
NEG_LABEL: str = "B-spline mobile-agent-e"
ALLOWED_LABELS: Set[str] = {POS_LABEL, NEG_LABEL}

def plot_and_save_roc(
    feature_name: str,
    df: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    pos_label: Optional[str] = None,
    neg_label: Optional[str] = None,
    filename: str | None = None,
) -> str:
    """Plot and save the ROC curve image for a given feature.

    This helper is defined but not executed automatically.
    Returns the path to the saved image.
    """
    if pos_label is None:
        pos_label = POS_LABEL
    if neg_label is None:
        neg_label = NEG_LABEL
    if df is None:
        df = load_filtered_df(CSV_PATH, pos_label=pos_label, neg_label=neg_label)

    if feature_name not in df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in the data.")
    sub = df[["type", feature_name]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        raise ValueError("No valid rows for the selected feature after dropping NaNs/inf.")
    y = (sub["type"] == pos_label).to_numpy(dtype=np.int8)
    if y.size == 0 or y.min() == y.max():
        raise ValueError("Both classes must be present to compute ROC.")
    scores = pd.to_numeric(sub[feature_name], errors="coerce").to_numpy(dtype=float)

    fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
    auc_val = auc(fpr, tpr)

    out_dir = Path(output_dir) if output_dir is not None else (DATA_DIR / "roc_curves")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = filename if filename is not None else f"roc_{_sanitize_filename(feature_name)}.png"
    out_path = out_dir / out_name

    plt.figure(figsize=(5, 5), dpi=150)
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC: {feature_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return str(out_path)

def plot_and_save_hist(
    feature_name: str,
    df: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    pos_label: Optional[str] = None,
    neg_label: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Plot and save a histogram of a given feature for the two classes.

    This helper is defined but not executed automatically.
    Returns the path to the saved image.
    """
    if pos_label is None:
        pos_label = POS_LABEL
    if neg_label is None:
        neg_label = NEG_LABEL
    if df is None:
        df = load_filtered_df(CSV_PATH, pos_label=pos_label, neg_label=neg_label)


    if feature_name not in df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in the data.")

    # Keep only valid rows for this feature
    sub = df[["type", feature_name]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        raise ValueError("No valid rows for the selected feature after dropping NaNs/inf.")

    # Boolean mask for classes
    y = (sub["type"] == pos_label).to_numpy(dtype=np.int8)
    if y.size == 0 or y.min() == y.max():
        raise ValueError("Both classes must be present to plot class-separated histogram.")

    # Coerce scores to float numpy arrays
    scores = pd.to_numeric(sub[feature_name], errors="coerce").to_numpy(dtype=float)
    mask_pos = y.astype(bool)
    pos_scores = scores[mask_pos]
    neg_scores = scores[~mask_pos]

    # If coercion produced NaNs (unlikely after dropna), filter them out
    pos_scores = pos_scores[~np.isnan(pos_scores)]
    neg_scores = neg_scores[~np.isnan(neg_scores)]
    if pos_scores.size == 0 or neg_scores.size == 0:
        raise ValueError("After cleaning, one of the classes has no valid numeric values.")

    out_dir = Path(output_dir) if output_dir is not None else (DATA_DIR / "feature_hists")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = filename if filename is not None else f"hist_{_sanitize_filename(feature_name)}.png"
    out_path = out_dir / out_name

    # Choose bins based on combined data to align histograms
    combined = np.concatenate([pos_scores, neg_scores])
    bins = np.histogram_bin_edges(combined, bins="auto")

    plt.figure(figsize=(6, 4), dpi=150)
    plt.hist(neg_scores, bins=bins, alpha=0.6, label=f"{neg_label} (n={neg_scores.size})", color="tab:orange", edgecolor="none")
    plt.hist(pos_scores, bins=bins, alpha=0.6, label=f"{pos_label} (n={pos_scores.size})", color="tab:blue", edgecolor="none")
    plt.xlabel(feature_name)
    plt.ylabel("Count")
    plt.title(f"Histogram: {feature_name}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return str(out_path)

def plot_tsne_and_save(
        df: pd.DataFrame | None = None, 
        pos_label: Optional[str] = None,
        neg_label: Optional[str] = None,
        output_dir: Path | None = None
    ) -> str:
    """Plot and save a t-SNE visualization of the feature space."""
    if pos_label is None:
        pos_label = POS_LABEL
    if neg_label is None:
        neg_label = NEG_LABEL
    if df is None:
        df = load_filtered_df(CSV_PATH, pos_label=pos_label, neg_label=neg_label)
    if output_dir is None:
        output_dir = DATA_DIR / "tsne_plots"
    output_dir.mkdir(parents=True, exist_ok=True)


    # Perform t-SNE
    df = df[df["type"].isin({pos_label, neg_label})].copy()

    features = df.drop(columns=["type"]).values
    features = sklearn.preprocessing.StandardScaler().fit_transform(features)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features, y=(df["type"] == pos_label).astype(int).values)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=(df["type"] == pos_label).astype(int), cmap="coolwarm", alpha=0.7)
    plt.colorbar(label="Class")
    plt.title("t-SNE Visualization of Feature Space")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Save the plot
    out_path = output_dir / "tsne_plot.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return str(out_path)

if __name__ == "__main__":
    # Allow overriding constants via CLI without changing function signatures
    parser = argparse.ArgumentParser(description="Compute per-feature ROC AUC for two classes from a CSV.")
    parser.add_argument("--csv", type=str, default=str(CSV_PATH), help="Path to input CSV (default: label_and_features.csv)")
    parser.add_argument("--pos_label", type=str, default=POS_LABEL, help="Positive class label (default: user3)")
    parser.add_argument("--neg_label", type=str, default=NEG_LABEL, help="Negative class label (default: B-spline mobile-agent-e)")
    parser.add_argument("--out_csv", type=str, default=None, help="Path to save AUC table CSV (default: test4_auc.csv)")
    args = parser.parse_args()

    # Rebind module-level constants
    CSV_PATH = Path(args.csv)
    POS_LABEL = args.pos_label
    NEG_LABEL = args.neg_label
    ALLOWED_LABELS = {POS_LABEL, NEG_LABEL}
    if args.out_csv is not None:
        out_csv = Path(args.out_csv)
    else:
        out_csv = None

    try:
        df = load_filtered_df(CSV_PATH, POS_LABEL, NEG_LABEL)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    res = compute_auc_per_feature(df, POS_LABEL, NEG_LABEL)
    if res.empty:
        print("No AUCs computed (check data and labels).")
        sys.exit(0)

    # Print and save results
    print(res.to_string(index=False))
    if out_csv is not None:
        res.to_csv(out_csv, index=False)
        print(f"\nSaved AUCs to: {out_csv}")