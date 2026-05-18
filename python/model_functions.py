import matplotlib.pyplot as plt
from sklearn.metrics import (
      accuracy_score, precision_score, average_precision_score,
      recall_score, f1_score, roc_auc_score, classification_report,
      confusion_matrix, roc_curve, ConfusionMatrixDisplay, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import numpy as np


def evaluate_model(model, x_test, y_test, model_name="Model", threshold=0.5, show_plots=True):
    
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int) # Use the threshold value to filter


    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    print(f"{model_name} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score: {pr_auc:.4f}")

    if show_plots:
        
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap="Blues")
        plt.title(f"{model_name} - Confusion Matrix")
        plt.show()

        # ROC & PR curves
        fig, axes = plt.subplots(1,2, figsize=(14,6))
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title(f"{model_name} - ROC Curve")
        axes[0].legend(loc="lower right")
        axes[0].grid(alpha=0.3)
        
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1].plot(rec_curve, prec_curve, label=f"PR-AUC = {pr_auc:.4f}")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title(f"{model_name} - Precision-Recall Curve")
        axes[1].legend(loc="lower left")
        axes[1].grid(alpha=0.3)


        plt.tight_layout()
        plt.show()

    return {
         'Model': model_name,
         'Accuracy': accuracy,
         'Precision': precision,
         'Recall': recall,
         'F1': f1,
         'ROC-AUC': roc_auc,
         'PR-AUC': pr_auc,
         'Threshold': threshold

    }


seed = 31415
np.random.seed(seed)

def make_models(scale_pos_weight, use_class_weights):
        return {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced' if use_class_weights else None,
            max_iter=2000, random_state=seed
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight='balanced' if use_class_weights else None,
            n_jobs=-1, random_state=seed
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight if use_class_weights else 1,
            eval_metric='aucpr', n_jobs=-1, random_state=seed
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            is_unbalance=use_class_weights,
            n_jobs=-1, random_state=seed, verbose=-1
        ),
    }