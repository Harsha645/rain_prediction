
# Loading the SPerforms initial validation: shape check,Sri Lanka Weather Dataset from CSV.
# Check columns , missing value report.

import pandas as pd
from src.config import DATASET_PATH

def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the Sri Lanka Weather Dataset from a CSV file.
    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    """

    # ── Load the CSV file ─────────────────────────────────────────────────────
    print(f'Loading dataset from: {path}')
    df = pd.read_csv(path)

    # ── Basic validation ──────────────────────────────────────────────────────
    print(f'\nDataset loaded successfully.')
    print(f'  Rows    : {df.shape[0]:,}')
    print(f'  Columns : {df.shape[1]}')
    print(f'  Columns : {list(df.columns)}')

    # ── Missing values report ─────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        print(f'\nMissing values found:')
        print(missing)
    else:
        print(f'\nNo missing values found.')

    # ── City information ──────────────────────────────────────────────────────
    print(f'\nUnique cities ({df["city"].nunique()}): {sorted(df["city"].unique())}')

    return df
