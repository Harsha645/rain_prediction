# =============================================================================
# Generates visualisations to understand the dataset before modelling:
#   - Precipitation distribution
#   - City-level precipitation box plots
#   - Target class distribution
#   - Feature correlation heatmap
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_precipitation_distribution(df: pd.DataFrame):
    """
    - It shows the skewness of the data (most days have low/no rain)
    - City-level differences reveal geographic variation
    - Helps justify the 1mm rain threshold used for RainTomorrow

    Parameters
    ----------
    df : pd.DataFrame  Dataset with 'precipitation_sum' and 'city' columns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ── Left: Histogram of all precipitation values ────────────────────────
    axes[0].hist(
        df['precipitation_sum'], bins=60,
        color='steelblue', edgecolor='white', alpha=0.85
    )
    axes[0].set_title('Precipitation Distribution', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Precipitation (mm)')
    axes[0].set_ylabel('Frequency')

    # ── Right: Box plots per city ──────────
    top_cities = df['city'].value_counts().head(8).index
    df[df['city'].isin(top_cities)].boxplot(
        column='precipitation_sum', by='city', ax=axes[1]
    )
    axes[1].set_title('Precipitation by City (Top 8)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('City')
    axes[1].set_ylabel('Precipitation (mm)')
    plt.xticks(rotation=45, ha='right')
    plt.suptitle('')  
    plt.tight_layout()
    plt.show()


def plot_target_distribution(df: pd.DataFrame, target_col: str = 'RainTomorrow'):
    """
    This is important for understanding class imbalance.
    If one class is much larger than the other, the model may
    be biased towards the majority class — which is why we use
    class_weight='balanced' in both algorithms.

    Parameters
    ----------
    df         : pd.DataFrame  Dataset with target column.
    target_col : str           Name of the target column.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    counts = df[target_col].value_counts()
    bars   = ax.bar(
        ['No Rain (0)', 'Rain (1)'], counts.values,
        color=['#4A90D9', '#2ECC71'], edgecolor='white', linewidth=1.5
    )

    # Add count labels on top of each bar
    for bar, cnt in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f'{cnt:,}', ha='center', fontsize=12, fontweight='bold'
        )

    ax.set_title('Target Variable: RainTomorrow', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(X: pd.DataFrame):
    """
    Correlation analysis helps identify:
    - Highly correlated features (multicollinearity)
    - Which features are most related to each other
    - Features that carry redundant information

    Parameters
    ----------
    X : pd.DataFrame  Feature matrix (numeric columns only).
    """
    plt.figure(figsize=(12, 9))

    # Create upper triangle mask so we only show lower triangle. This avoids duplicate information in the heatmap
    corr_matrix = X.corr()
    mask        = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask       = mask,
        annot      = True,
        fmt        = '.2f',
        cmap       = 'coolwarm',
        center     = 0,
        linewidths = 0.5,
        annot_kws  = {'size': 7}
    )

    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def run_eda(df: pd.DataFrame, X: pd.DataFrame = None):
    """
    Run all EDA visualisations in sequence.

    Called from the notebook to generate all EDA charts
    with a single function call.

    Parameters
    ----------
    df : pd.DataFrame  Full dataset (for precipitation + target plots)
    X  : pd.DataFrame  Feature matrix (for correlation heatmap)
    """
    print('=' * 50)
    print('EXPLORATORY DATA ANALYSIS')
    print('=' * 50)

    # Basic statistics
    print('\nData Types:')
    print(df.dtypes)
    print('\nStatistical Summary:')
    print(df.describe())

    # Visualisations
    plot_precipitation_distribution(df)

    if 'RainTomorrow' in df.columns:
        plot_target_distribution(df)

    if X is not None:
        plot_correlation_heatmap(X)
