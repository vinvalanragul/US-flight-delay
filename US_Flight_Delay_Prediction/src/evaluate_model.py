"""
Model Evaluation Module for US Flight Delay Prediction.
Computes comprehensive classification metrics:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (TN, FP, FN, TP)
- Specificity
- ROC-AUC
- Formatted Classification Report
"""

import math
from typing import List, Dict, Any, Tuple

def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    """Computes TN, FP, FN, TP counts."""
    tn = fp = fn = tp = 0
    for true, pred in zip(y_true, y_pred):
        if true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
        elif true == 1 and pred == 1:
            tp += 1
            
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}

def compute_roc_auc_score(y_true: List[int], y_scores: List[float]) -> float:
    """Computes ROC-AUC score using trapezoidal rule over sorted probabilities."""
    if not y_true or not y_scores or len(y_true) != len(y_scores):
        return 0.5
        
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5
        
    # Pair and sort by score descending
    pairs = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)
    
    tp = 0
    fp = 0
    prev_fp = 0
    prev_tp = 0
    auc = 0.0
    
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            # Trapezoid area
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_fp = fp
            prev_tp = tp
            
    return round(auc / (pos * neg), 4)

def evaluate_classification_predictions(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float] = None
) -> Dict[str, Any]:
    """Calculates all key evaluation metrics for flight delay classification."""
    total = len(y_true)
    if total == 0:
        return {}
        
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm["TN"], cm["FP"], cm["FN"], cm["TP"]
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Class 0 metrics (On-time)
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = (2 * prec_0 * rec_0) / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
    
    roc_auc = compute_roc_auc_score(y_true, y_prob) if y_prob else 0.5
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "classification_report": {
            "0 (On-Time)": {
                "precision": round(prec_0, 4),
                "recall": round(rec_0, 4),
                "f1_score": round(f1_0, 4),
                "support": tn + fp
            },
            "1 (Delayed >=15m)": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "support": tp + fn
            },
            "macro_avg": {
                "precision": round((precision + prec_0) / 2, 4),
                "recall": round((recall + rec_0) / 2, 4),
                "f1_score": round((f1 + f1_0) / 2, 4),
                "support": total
            }
        }
    }
