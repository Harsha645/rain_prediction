# All data preprocessing steps for the Rain Prediction project:
#

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    RAIN_THRESHOLD, TARGET_COLUMN,
    FEATURE_COLS, TEST_SIZE, RANDOM_STATE
)

#   1. Parse date column and sort by city + date
def parse_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the 'time' column to datetime and sort dataset
    by city and date in ascending order.

    This step is critical before engineering RainTomorrow —
    the shift(-1) operation only works correctly when data
    is sorted chronologically within each city.

    Parameters
    ----------
    df : pd.DataFrame  Raw dataset from data_loader.

    Returns
    -------
    pd.DataFrame  Sorted dataset with datetime 'time' column.
    """
    # Convert string date to datetime object
    df['time'] = pd.to_datetime(df['time'])

    # Sort by city first, then by date within each city
    df = df.sort_values(by=['city', 'time']).reset_index(drop=True)

    print('Date parsed and dataset sorted by city + date.')
    return df

#   2. Engineer target variable (RainTomorrow)
def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the binary target variable 'RainTomorrow'.

    Logic:
    - For each city, shift precipitation_sum by -1 (next day's value)
    - If next day's precipitation > RAIN_THRESHOLD → 1 (Rain)
    - Otherwise → 0 (No Rain)
    - Drop the last row of each city (no 'tomorrow' available)

    The groupby ensures we don't accidentally use the first row
    of the next city as 'tomorrow' for the last row of the current city.

    Parameters
    ----------
    df : pd.DataFrame  Sorted dataset.

    Returns
    -------
    pd.DataFrame  Dataset with RainTomorrow column added.
    """
    # Shift precipitation within each city group to get next day's value
    df[TARGET_COLUMN] = df.groupby('city')['precipitation_sum'].shift(-1)

    # Apply rain threshold: 1 = Rain, 0 = No Rain
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] > RAIN_THRESHOLD).astype(float)

    # Drop rows where next day is unavailable (last row of each city)
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Print class distribution
    rain_count    = df[TARGET_COLUMN].sum()
    no_rain_count = (df[TARGET_COLUMN] == 0).sum()
    total         = len(df)

    print(f'\nTarget variable "{TARGET_COLUMN}" created.')
    print(f'  Rain    (1) : {rain_count:,}  ({rain_count/total*100:.1f}%)')
    print(f'  No Rain (0) : {no_rain_count:,}  ({no_rain_count/total*100:.1f}%)')

    return df

#   3. Select feature columns
def get_feature_cols(df: pd.DataFrame) -> list:
    """
    Return the list of available feature columns from FEATURE_COLS
    that actually exist in the dataset.

    This is a safety check — if the dataset version has slightly
    different column names, this prevents KeyErrors.

    Parameters
    ----------
    df : pd.DataFrame  Preprocessed dataset.

    Returns
    -------
    list  Available feature column names.
    """
    available = [col for col in FEATURE_COLS if col in df.columns]
    missing   = [col for col in FEATURE_COLS if col not in df.columns]

    print(f'\nFeatures available : {len(available)}')
    if missing:
        print(f'Features missing   : {missing}')

    return available

#   4. Build feature matrix (X) and target vector (y)
def build_X_y(df: pd.DataFrame, feature_cols: list):
    """
    Build the feature matrix X and target vector y.

    Fills any remaining missing values with column medians.
    Using median (not mean) because weather data can be skewed.

    Parameters
    ----------
    df           : pd.DataFrame  Full preprocessed dataset.
    feature_cols : list          Selected feature column names.

    Returns
    -------
    X : pd.DataFrame  Feature matrix
    y : pd.Series     Target vector
    """
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    # Fill any remaining missing values with column median
    X.fillna(X.median(), inplace=True)

    print(f'\nFeature matrix X : {X.shape}')
    print(f'Target vector  y : {y.shape}')
    print(f'Missing values   : {X.isnull().sum().sum()}')

    return X, y

#   5. Train/test split (80/20 stratified)
def split_data(X, y):
    """
    Split data into training and test sets.

    Uses stratified sampling (stratify=y) to ensure both train
    and test sets have the same Rain/No Rain class ratio.
    This is important when classes are imbalanced.

    Parameters
    ----------
    X : pd.DataFrame  Feature matrix.
    y : pd.Series     Target vector.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y           # Preserve class balance in both sets
    )

    print(f'\nTrain/Test Split (stratified):')
    print(f'  Training set : {X_train.shape[0]:,} samples')
    print(f'  Test set     : {X_test.shape[0]:,} samples')
    print(f'  Train balance: {y_train.value_counts().to_dict()}')
    print(f'  Test  balance: {y_test.value_counts().to_dict()}')

    return X_train, X_test, y_train, y_test

#   6. Feature scaling (StandardScaler - fit on train only)
def scale_features(X_train, X_test):
    """
    Apply StandardScaler to features.

    IMPORTANT: The scaler is fit ONLY on the training set,
    then used to transform both train and test sets.

    Fitting on the full dataset would cause DATA LEAKAGE —
    the model would indirectly 'see' test set statistics during training,
    giving an overly optimistic accuracy estimate.

    Parameters
    ----------
    X_train : Training feature matrix (scaler is fit here)
    X_test  : Test feature matrix (only transformed, never fit)

    Returns
    -------
    X_train_scaled : np.ndarray  Scaled training features
    X_test_scaled  : np.ndarray  Scaled test features
    scaler         : StandardScaler  Fitted scaler (needed for prediction)
    """
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit + transform train
    X_test_scaled  = scaler.transform(X_test)        # Only transform test

    print(f'\nStandardScaler applied.')
    print(f'  Fit on training set only (prevents data leakage).')

    return X_train_scaled, X_test_scaled, scaler


def run_preprocessing(df: pd.DataFrame):
    """
    Run the complete preprocessing pipeline in order.

    This is the main function called from the notebook.
    It chains all preprocessing steps and returns everything
    needed for model training and evaluation.

    Parameters
    ----------
    df : pd.DataFrame  Raw dataset from data_loader.

    Returns
    -------
    df            : pd.DataFrame  Fully preprocessed dataset
    X_train_scaled: np.ndarray
    X_test_scaled : np.ndarray
    y_train       : pd.Series
    y_test        : pd.Series
    scaler        : StandardScaler (fitted)
    feature_cols  : list
    """
    print('=' * 50)
    print('PREPROCESSING PIPELINE')
    print('=' * 50)

    # Step 1: Parse dates and sort
    df = parse_and_sort(df)

    # Step 2: Engineer RainTomorrow target variable
    df = engineer_target(df)

    # Step 3: Identify available feature columns
    feature_cols = get_feature_cols(df)

    # Step 4: Build X and y
    X, y = build_X_y(df, feature_cols)

    # Step 5: Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 6: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print('\nPreprocessing complete.')
    return df, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols
