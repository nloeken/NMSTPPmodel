# config.py
import os
import numpy as np

# --- Fast Debug Mode Settings ---
# If True, the pipeline runs on a smaller subset of data for quick checks.
FAST_DEBUG = False 
FAST_MAX_MATCHES = 50       # Max number of matches to process in debug mode
FAST_MAX_ROWS = 400_000   # Max number of rows to use from the combined file in debug mode

# --- Model Hyperparameters ---
EPOCHS = 25                 # Maximum number of training epochs
PATIENCE = 4                # Number of epochs to wait for improvement before early stopping
SEQ_LEN = 40                # The length of each event sequence fed to the model
GRAD_CLIP = 1.0             # Gradient clipping value to prevent exploding gradients

# --- File and Directory Paths ---
BASE_DIR = r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data"
EVENTS_DIR = os.path.join(BASE_DIR, "events")
THREE_SIXTY_DIR = os.path.join(BASE_DIR, "three-sixty")
MERGED_DIR = os.path.join(BASE_DIR, "merged")
COMBINED_FILE = os.path.join(BASE_DIR, "combined", "combined1.csv")
SAMPLE_FILE = os.path.join(BASE_DIR, "samplefile.csv")
PREDICTION_FILE = os.path.join(BASE_DIR, "predictions", "prediction.csv")
SAMPLE_PREDICTION_FILE = os.path.join(BASE_DIR, "predictions", "sample_prediction.csv")

# --- Constants for Feature Engineering ---
# List of event types to be dropped as they are considered noise or irrelevant for the model's task.
DROP_EVENT_NAMES = [
    'Starting XI', 'Ball Receipt*', 'Pressure', 'Foul Committed', 'Foul Won', 
    'Ball Recovery', 'Block', 'Interception', 'Goal Keeper', 'Dribbled Past', 
    'Miscontrol', 'Dispossessed', 'Injury Stoppage', 'Substitution', 
    'Tactical Shift', 'Player Off', 'Player On', 'Shield', 'Bad Behaviour', 
    'Offside', 'Half End', 'Referee Ball-Drop'
]

# Types of own goals used to filter out entire matches containing them.
OWN_GOAL_TYPES = ['Own Goal For', 'Own Goal Against']

# --- Zone Centroids for Fuzzy C-Means Zoning ---
# Centroid coordinates are first defined on a 100x100 grid.
_centroid_x_100 = [8.5, 25.25, 41.75, 58.25, 74.75, 91.5, 8.5, 25.25, 41.75, 58.25, 74.75, 91.5, 33.5, 66.5, 33.5, 66.5, 33.5, 66.5, 8.5, 91.5]
_centroid_y_100 = [89.45, 89.45, 89.45, 89.45, 89.45, 89.45, 10.55, 10.55, 10.55, 10.55, 10.55, 10.55, 71.05, 71.05, 50., 50., 28.95, 28.95, 50., 50.]

# Then, they are scaled to a standard pitch size of 105x68 meters.
ZONE_CENTROIDS_X = np.array([x * 105 / 100 for x in _centroid_x_100])
ZONE_CENTROIDS_Y = np.array([y * 68 / 100 for y in _centroid_y_100])
ZONE_CENTROIDS = {i: (ZONE_CENTROIDS_X[i], ZONE_CENTROIDS_Y[i]) for i in range(len(ZONE_CENTROIDS_X))}

# --- Constants for Data Preprocessing ---
# Columns that are stored as string representations of dicts/lists in the CSV
# and need to be converted back to Python objects.
COLUMNS_TO_EVAL = [
    "type", "pass", "clearance", "shot", 
    "carry", "dribble", "duel", "freeze_frame"
]

# --- Directory Creation ---
# Ensure that necessary output directories exist.
os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(COMBINED_FILE), exist_ok=True)
os.makedirs(os.path.dirname(PREDICTION_FILE), exist_ok=True)