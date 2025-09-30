import os
import numpy as np

# fast debug mode settings
FAST_DEBUG = False 
FAST_MAX_MATCHES = 50
FAST_MAX_ROWS = 400_000

# model hyperparameters
EPOCHS = 25
PATIENCE = 4
SEQ_LEN = 40 
GRAD_CLIP = 1.0

#paths
EVENTS_DIR = r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data\events"
THREE_SIXTY_DIR = r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data\three-sixty"
MERGED_DIR = r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data\merged"
COMBINED_FILE =  r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data\combined\combined1.csv"
SAMPLE_FILE = r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data\samplefile.csv"
PREDICTION_FILE = r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data\predictions\prediction.csv"
SAMPLE_PREDICTION_FILE = r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data\predictions\sample_prediction.csv"

# constants
DROP_EVENT_NAMES = [
    'Starting XI', 'Ball Receipt*', 'Pressure', 'Foul Committed', 'Foul Won', 
    'Ball Recovery', 'Block', 'Interception', 'Goal Keeper', 'Dribbled Past', 
    'Miscontrol', 'Dispossessed', 'Injury Stoppage', 'Substitution', 
    'Tactical Shift', 'Player Off', 'Player On', 'Shield', 'Bad Behaviour', 
    'Offside', 'Half End', 'Referee Ball-Drop'
]
OWN_GOAL_TYPES = ['Own Goal For', 'Own Goal Against']
_centroid_x_100 = [8.5, 25.25, 41.75, 58.25, 74.75, 91.5, 8.5, 25.25, 41.75, 58.25, 74.75, 91.5, 33.5, 66.5, 33.5, 66.5, 33.5, 66.5, 8.5, 91.5]
_centroid_y_100 = [89.45, 89.45, 89.45, 89.45, 89.45, 89.45, 10.55, 10.55, 10.55, 10.55, 10.55, 10.55, 71.05, 71.05, 50., 50., 28.95, 28.95, 50., 50.]
ZONE_CENTROIDS_X = np.array([x * 105 / 100 for x in _centroid_x_100])
ZONE_CENTROIDS_Y = np.array([y * 68 / 100 for y in _centroid_y_100])
ZONE_CENTROIDS = {i: (ZONE_CENTROIDS_X[i], ZONE_CENTROIDS_Y[i]) for i in range(len(ZONE_CENTROIDS_X))}

os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(COMBINED_FILE), exist_ok=True)
os.makedirs(os.path.dirname(PREDICTION_FILE), exist_ok=True)