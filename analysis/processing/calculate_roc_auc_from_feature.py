# from __future__ import annotations

from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Optional, Tuple, TypedDict, Union, Set
import numpy.typing as npt
import argparse
from motionevent_classes import FingerEvent
from extract_feature_of_swipes import build_features_dataframe
from sklearn.feature_selection import mutual_info_classif

# compute svm and xgboost on the filtered data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import sklearn.pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from copy import deepcopy
import random
from sklearn.metrics import precision_recall_curve

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



class ThresholdPosterior(TypedDict):
    """
    Docstring for ThresholdPosterior

    :param threshold: Description
    :param lesser_than_threshold_agent_log_odds: log(P(less than threshold | agent) / P(less than threshold | human))
    :param greater_than_threshold_agent_log_odds: log(P(greater than threshold | agent) / P(greater than threshold | human))
    """
    threshold: float
    lesser_than_threshold_agent_log_odds: float
    greater_than_threshold_agent_log_odds: float



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

        # note that the positive may be irrelevant to agent being 1 or 0; we just care about threshold_posterior which nihilates the discrepancy
        true_positive_rate = np.sum((raw_scores >= thresh) & (y == 1)) / np.sum(y == 1)
        false_positive_rate = np.sum((raw_scores >= thresh) & (y == 0)) / np.sum(y == 0)
        threshold_posterior: ThresholdPosterior = {
            "threshold": float(thresh),
            "greater_than_threshold_agent_log_odds": float(np.log(true_positive_rate) - np.log(false_positive_rate)), # np.log(0.0) is -inf, and throw RuntimeWarning: divide by zero encountered in log
            "lesser_than_threshold_agent_log_odds": float(np.log((1 - true_positive_rate) ) - np.log(1 - false_positive_rate)),
        }
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
        y_pred = (raw_scores <= threshold).astype(int)
        acc = accuracy_score(y, y_pred)

        true_positive_rate = np.sum((y_pred == 1) & (y == 1)) / np.sum(y == 1)
        false_positive_rate = np.sum((y_pred == 1) & (y == 0)) / np.sum(y == 0)

        threshold_posterior: ThresholdPosterior = {
            "threshold": float(threshold),
            "lesser_than_threshold_agent_log_odds": float(np.log(true_positive_rate) - np.log(false_positive_rate)), # np.log(0.0) is -inf, and throw RuntimeWarning: divide by zero encountered in log
            "greater_than_threshold_agent_log_odds": float(np.log((1 - true_positive_rate) ) - np.log(1 - false_positive_rate)),
        }
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
            import matplotlib.pyplot as plt
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
                    import matplotlib.pyplot as plt
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

                    true_positive_rate = np.sum((y_pred == 1) & (y == 1)) / np.sum(y == 1)
                    false_positive_rate = np.sum((y_pred == 1) & (y == 0)) / np.sum(y == 0)
                    threshold_posterior: ThresholdPosterior = {
                        "threshold": float(thresh),
                        "greater_than_threshold_agent_log_odds": float(np.log(true_positive_rate) - np.log(false_positive_rate)), # np.log(0.0) is -inf, and throw RuntimeWarning: divide by zero encountered in log
                        "lesser_than_threshold_agent_log_odds": float(np.log((1 - true_positive_rate) ) - np.log(1 - false_positive_rate)),
                    }
                    threshold_posteriors[feat] = threshold_posterior
            except Exception:
                continue
        
        if best_acc >= 0.36 and best_acc <= 0.37:
            print(f"Feature {feat} has best accuracy {best_acc} at break-even point.")
            if plot_plt:
                # plot the precision-recall curve and find the break-even point
                import matplotlib.pyplot as plt
                # create new figure
                plt.figure()
                # plt.plot(recalls, precisions)
                # plot scatter points instead
                plt.scatter(recalls, precisions, s=0.1, alpha=1.0)
                # input(f"Now comparing {pos_label} vs {neg_label} for feature {feat}. Press Enter to continue...")
                # use that as title
                plt.title(f"Precision-Recall Curve for feature {feat} with pos_label {pos_label} and neg_label {neg_label}")


                import matplotlib.pyplot as plt
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

        threshold_posterior: ThresholdPosterior = {
            "threshold": float(thresh),
            "greater_than_threshold_agent_log_odds": float(np.log(tpr[idx]) - np.log(fpr[idx])),
            "lesser_than_threshold_agent_log_odds": float(np.log((1 - tpr[idx]) ) - np.log(1 - fpr[idx])),
        }
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
    from sklearn.metrics import average_precision_score

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
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        # ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
        
        
        # SVM with scaling
        svm_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42))
        svm_clf.fit(X_train, y_train)
        y_pred_svm = svm_clf.predict(X_test)
        svm_acc: float = accuracy_score(y_test, y_pred_svm)

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
        xgb_acc: float = accuracy_score(y_test, y_pred_xgb)
        
        result: SVMAndXGBoostResult = {"svm_accuracy": svm_acc, "xgb_accuracy": xgb_acc}
        return result, svm_clf, xgb_clf
    else:
        return {"svm_accuracy": None, "xgb_accuracy": None}, None, None


def make_feature_and_learner_table(task_cluster_id: int, method_name: str, 
                pos_iterator: Tuple[str, List[List[FingerEvent]]], 
                neg_iterator: Tuple[str, List[List[FingerEvent]]]) -> pd.DataFrame:
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
    else:
        raise ValueError("Resulting AUC dataframe is empty.")

    return result_csv


def make_feature_and_learner_table_but_acc(task_cluster_id: int, method_name: str, 
                pos_iterator: Tuple[str, List[List[FingerEvent]]], 
                neg_iterator: Tuple[str, List[List[FingerEvent]]],
                acc_obtain_function: Callable[[pd.DataFrame, str, str], Tuple[pd.DataFrame, Dict[str, ThresholdPosterior]]]
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

        return result_csv, svm_pipeline, xgb
    else:
        raise ValueError("Resulting ACC dataframe is empty.")


def plot_acc_increase_as_more_feature_used(task_cluster_id: int, method_name: str, pos_iterator: Tuple[str, List[List[FingerEvent]]], neg_iterator: Tuple[str, List[List[FingerEvent]]]) -> pd.DataFrame:
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
                pos_iterator: Tuple[str, List[List[FingerEvent]]], 
                neg_iterator: Tuple[str, List[List[FingerEvent]]]) -> pd.DataFrame:
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
    type_and_data_iterators: List[Tuple[str, List[List[FingerEvent]]]],
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