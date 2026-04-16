
# Configuration file for the Rain Prediction project.
# All constants, file paths, model parameters and feature definitions

# ─── Dataset ──────────────────────────────────────────────────────────────────
import os as _os
DATASET_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'data', 'SriLanka_Weather_Dataset_V1.csv')
# All are numeric daily measurements from the Open-Meteo API.
