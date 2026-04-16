
# Configuration file for the Rain Prediction project.
# All constants, file paths, model parameters and feature definitions

# ─── Dataset ──────────────────────────────────────────────────────────────────
import os as _os
DATASET_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data', 'SriLanka_Weather_Dataset_V1.csv')

# ─── Target Variable ──────────────────────────────────────────────────────────
# A day is classified as 'rainy' if precipitation exceeds this threshold (mm)
# 1.0mm is the standard World Meteorological Organization (WMO) threshold
RAIN_THRESHOLD = 1.0
TARGET_COLUMN  = 'RainTomorrow'

# ─── Feature Columns ──────────────────────────────────────────────────────────
# These are the 14 weather features used as input to the ML models.
# All are numeric daily measurements from the Open-Meteo API.
FEATURE_COLS = [
    'temperature_2m_max',           # Maximum temperature at 2m height (°C)
    'temperature_2m_min',           # Minimum temperature at 2m height (°C)
    'temperature_2m_mean',          # Mean temperature at 2m height (°C)
    'apparent_temperature_max',     # Maximum feels-like temperature (°C)
    'apparent_temperature_min',     # Minimum feels-like temperature (°C)
    'apparent_temperature_mean',    # Mean feels-like temperature (°C)
    'precipitation_sum',            # Total daily precipitation (mm)
    'rain_sum',                     # Total daily rain (mm)
    'precipitation_hours',          # Hours with precipitation
    'windspeed_10m_max',            # Maximum wind speed at 10m (km/h)
    'windgusts_10m_max',            # Maximum wind gusts at 10m (km/h)
    'winddirection_10m_dominant',   # Dominant wind direction (degrees)
    'shortwave_radiation_sum',      # Daily solar radiation (MJ/m²)
    'et0_fao_evapotranspiration'    # Reference evapotranspiration (mm)
]

# ─── Train / Test Split ───────────────────────────────────────────────────────
TEST_SIZE    = 0.20   # 20% of data reserved for testing
RANDOM_STATE = 42     # Fixed seed for reproducibility

# ─── Decision Tree Hyperparameters ────────────────────────────────────────────
DT_PARAMS = {
    'max_depth'         : 10,        # Maximum depth of the tree
    'min_samples_split' : 20,        # Minimum samples required to split a node
    'min_samples_leaf'  : 10,        # Minimum samples required at a leaf node
    'class_weight'      : 'balanced',# Handle class imbalance automatically
    'random_state'      : RANDOM_STATE
}

# ─── c Hyperparameters ────────────────────────────────────────────
RF_PARAMS = {
    'n_estimators'      : 100,       # Number of trees in the forest
    'max_depth'         : 15,        # Maximum depth of each tree
    'min_samples_split' : 10,        # Minimum samples required to split a node
    'min_samples_leaf'  : 5,         # Minimum samples required at a leaf node
    'max_features'      : 'sqrt',    # Features to consider per split (sqrt rule)
    'class_weight'      : 'balanced',# Handle class imbalance automatically
    'n_jobs'            : -1,        # Use all available CPU cores
    'random_state'      : RANDOM_STATE
}