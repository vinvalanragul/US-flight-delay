"""
Configuration settings for US Flight Delay Prediction Big Data Pipeline.
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Dataset locations (checked in order)
DATASET_CANDIDATES = [
    os.path.join(BASE_DIR, "archive (4).zip"),
    os.path.join(BASE_DIR, "archive.zip"),
    os.path.join(DATA_DIR, "flights_sample_3m.csv"),
    os.path.join(DATA_DIR, "flights_sample.csv")
]

# Operational Threshold: US DOT / BTS standard for flight delay is 15 minutes
DELAY_THRESHOLD_MINUTES = 15.0

# Supported Grouping Keys for MapReduce
SUPPORTED_KEYS = [
    "AIRLINE",
    "ORIGIN",
    "DEST",
    "MONTH",
    "CRS_DEP_HOUR"
]

# Pre-departure features (Strictly prevents Data Leakage)
# Excludes: DEP_DELAY, ARR_DELAY, TAXI_OUT, TAXI_IN, WHEELS_OFF, WHEELS_ON,
# and all DELAY_DUE_* post-flight breakdown columns.
FEATURE_COLUMNS = [
    "AIRLINE",
    "ORIGIN",
    "DEST",
    "MONTH",
    "DAY_OF_WEEK",
    "CRS_DEP_HOUR",
    "CRS_ARR_HOUR",
    "DISTANCE"
]

TARGET_COLUMN = "is_delayed"

# Processing settings
DEFAULT_SAMPLE_LIMIT = 50000
CHUNK_SIZE = 10000
RANDOM_STATE = 42

# Ensure required directories exist
for path in [DATA_DIR, OUTPUTS_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(path, exist_ok=True)
