# Model evaluation functions for the Rain Prediction project.
# Computes and visualizes performance metrics for both algorithms:
# - Accuracy, Precision, Recall, F1-Score
# - Confusion Matrix
# - ROC Curve and AUC
# - Feature Importance (Random Forest)
# - Algorithm Comparison (side-by-side bar chart + overlaid ROC)
# - 5-Fold Cross Validation
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import plot_tree


def evaluate_model(model, X_test_scaled, y_test, model_name: str) -> dict:
    """
    Evaluate a trained model on the test set and print all metrics.

    Metrics explained:
    - Accuracy  : Overall % of correct predictions
    - Precision : Of predicted Rain days, how many were actually Rain
    - Recall    : Of actual Rain days, how many did the model catch
    - F1-Score  : Harmonic mean of Precision and Recall (best for imbalanced data)

    Parameters
    ----------
    model        : Trained sklearn classifier
    X_test_scaled: np.ndarray  Scaled test features
    y_test       : pd.Series   True test labels
    model_name   : str         Display name for printing

    Returns
    -------
    dict  Dictionary containing all metric scores and ROC data
    """
    # ── Generate predictions ──────────────────────────────────────────────────
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of Rain

    # ── Calculate metrics ─────────────────────────────────────────────────────
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')
    f1        = f1_score(y_test, y_pred, average='weighted')

    # ── ROC curve data ────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc(fpr, tpr)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f'\n{model_name} – Evaluation Results')
    print('=' * 45)
    print(f'  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)')
    print(f'  Precision : {precision:.4f}')
    print(f'  Recall    : {recall:.4f}')
    print(f'  F1-Score  : {f1:.4f}')
    print(f'  AUC       : {roc_auc:.4f}')
    print()
    print(classification_report(y_test, y_pred, target_names=['No Rain', 'Rain']))

    return {
        'name'     : model_name,
        'accuracy' : accuracy,
        'precision': precision,
        'recall'   : recall,
        'f1'       : f1,
        'auc'      : roc_auc,
        'fpr'      : fpr,
        'tpr'      : tpr,
        'y_pred'   : y_pred
    }


def plot_confusion_and_roc(metrics: dict, cmap: str = 'Blues', y_test=None):
    """
    Plot confusion matrix and ROC curve side by side for one model.

    Confusion Matrix shows:
    - True Positives (Rain correctly predicted)
    - True Negatives (No Rain correctly predicted)
    - False Positives (Predicted Rain but was No Rain)
    - False Negatives (Predicted No Rain but was Rain)

    Parameters
    metrics : dict Output from evaluate_model()
    cmap : str Colormap for confusion matrix
    y_test : array True labels (needed for confusion matrix)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm   = confusion_matrix(y_test, metrics['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['No Rain', 'Rain'])
    disp.plot(ax=axes[0], cmap=cmap, colorbar=False)
    axes[0].set_title(f'{metrics["name"]} – Confusion Matrix', fontweight='bold')

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    axes[1].plot(metrics['fpr'], metrics['tpr'], lw=2,
                 label=f'AUC = {metrics["auc"]:.4f}')
    axes[1].plot([0, 1], [0, 1], '--', color='navy', lw=1, label='Random Guess')
    axes[1].set_title(f'{metrics["name"]} – ROC Curve', fontweight='bold')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_decision_tree_visual(dt_model, feature_cols: list):
    """
    Visualise the top 3 levels of the trained Decision Tree.

    Parameters
    dt_model : Trained DecisionTreeClassifier
    feature_cols : list Feature names for axis labels
    """
    plt.figure(figsize=(22, 8))
    plot_tree(
        dt_model,
        max_depth     = 3,
        feature_names = feature_cols,
        class_names   = ['No Rain', 'Rain'],
        filled        = True,
        rounded       = True,
        fontsize      = 9
    )
    plt.title('Decision Tree – Top 3 Levels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(rf_model, feature_cols: list):
    """
    Plot Random Forest feature importances as a horizontal bar chart.

    Parameters
    rf_model : Trained RandomForestClassifier
    feature_cols : list Feature names
    """
    # Get importances and sort ascending for horizontal bar chart
    importances = pd.Series(
        rf_model.feature_importances_, index=feature_cols
    ).sort_values()

    # Colour top 25% features green, rest blue
    colors = ['#2ECC71' if v >= importances.quantile(0.75)
              else '#4A90D9' for v in importances.values]

    plt.figure(figsize=(10, 7))
    importances.plot(kind='barh', color=colors, edgecolor='white')
    plt.axvline(
        importances.mean(), color='red', linestyle='--', alpha=0.7,
        label=f'Mean ({importances.mean():.3f})'
    )
    plt.title('Random Forest – Feature Importances', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_comparison(dt_metrics: dict, rf_metrics: dict):
    """
    Plot side-by-side bar chart comparing all metrics for both models.
    Also plots overlaid ROC curves for direct comparison.

    Parameters
    dt_metrics : dict  Output from evaluate_model() for Decision Tree
    rf_metrics : dict  Output from evaluate_model() for Random Forest
    """
    metrics   = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    dt_scores = [dt_metrics['accuracy'], dt_metrics['precision'],
                 dt_metrics['recall'],   dt_metrics['f1'], dt_metrics['auc']]
    rf_scores = [rf_metrics['accuracy'], rf_metrics['precision'],
                 rf_metrics['recall'],   rf_metrics['f1'], rf_metrics['auc']]

    # ── Bar Chart ─────────────────────────────────────────────────────────────
    x = np.arange(len(metrics))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    b1 = axes[0].bar(x - w/2, dt_scores, w,
                     label='Decision Tree',  color='#4A90D9', edgecolor='white')
    b2 = axes[0].bar(x + w/2, rf_scores, w,
                     label='Random Forest', color='#2ECC71', edgecolor='white')

    # Add score labels on top of each bar
    for bar in list(b1) + list(b2):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', fontsize=9
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_title('Algorithm Performance Comparison', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Score')
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)

    # ── Overlaid ROC Curves ───────────────────────────────────────────────────
    axes[1].plot(dt_metrics['fpr'], dt_metrics['tpr'], color='darkorange', lw=2,
                 label=f'Decision Tree  (AUC = {dt_metrics["auc"]:.4f})')
    axes[1].plot(rf_metrics['fpr'], rf_metrics['tpr'], color='green', lw=2,
                 label=f'Random Forest  (AUC = {rf_metrics["auc"]:.4f})')
    axes[1].plot([0, 1], [0, 1], '--', color='navy', lw=1, label='Random Guess')
    axes[1].set_title('ROC Curve Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(loc='lower right')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ── Summary Table ─────────────────────────────────────────────────────────
    results = pd.DataFrame({
        'Algorithm' : ['Decision Tree', 'Random Forest'],
        'Accuracy'  : [dt_metrics['accuracy'],  rf_metrics['accuracy']],
        'Precision' : [dt_metrics['precision'], rf_metrics['precision']],
        'Recall'    : [dt_metrics['recall'],    rf_metrics['recall']],
        'F1-Score'  : [dt_metrics['f1'],        rf_metrics['f1']],
        'AUC'       : [dt_metrics['auc'],        rf_metrics['auc']]
    }).set_index('Algorithm')

    print('\nAlgorithm Comparison Summary')
    print('=' * 60)
    print(results.applymap(lambda x: f'{x:.4f}').to_string())
    winner = results['Accuracy'].idxmax()
    print(f'\nBest Performing Algorithm: {winner}')


def run_cross_validation(dt_model, rf_model, X_train_scaled, y_train):
    """
    Run 5-fold stratified cross validation for both models.

    Parameters
    dt_model : Trained DecisionTreeClassifier
    rf_model : Trained RandomForestClassifier
    X_train_scaled : np.ndarray Scaled training features
    y_train : pd.Series Training labels
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    dt_cv = cross_val_score(dt_model, X_train_scaled, y_train,
                            cv=cv, scoring='accuracy')
    rf_cv = cross_val_score(rf_model, X_train_scaled, y_train,
                            cv=cv, scoring='accuracy')

    print('\n5-Fold Cross Validation Results')
    print('=' * 45)
    print(f'  Decision Tree  : {dt_cv.mean():.4f} +/- {dt_cv.std():.4f}')
    print(f'  Random Forest  : {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}')
