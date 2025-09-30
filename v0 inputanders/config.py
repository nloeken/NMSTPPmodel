# config.py
import os
import numpy as np

# ---- Fast/Debug Switches ----
FAST_DEBUG = False 
FAST_MAX_MATCHES = 50
FAST_MAX_ROWS = 400_000

# ---- Feste Parameter für die Studie ----
EPOCHS = 25
PATIENCE = 4
SEQ_LEN = 40 
GRAD_CLIP = 1.0

# ---- Pfade ----
EVENTS_DIR = r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data\events"
# MODIFIZIERT: Korrigierter Pfad für Windows (kein führender Backslash)
THREE_SIXTY_DIR = r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data\three-sixty"
MERGED_DIR = r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data\merged"
# MODIFIZIERT: Pfad angepasst für Konsistenz
COMBINED_FILE =  r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data\combined\combined1.csv"
SAMPLE_FILE = r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data\samplefile.csv"
PREDICTION_FILE = r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data\predictions\prediction.csv"
SAMPLE_PREDICTION_FILE = r"C:\Users\nloeken\Downloads\open-data-master\open-data-master\data\predictions\sample_prediction.csv"

# ---- Event-Filter (MODIFIZIERT) ----
# Wir verwenden jetzt eine Blacklist anstelle einer Whitelist, um mehr Rauschen zu entfernen
DROP_EVENT_NAMES = [
    'Starting XI', 'Ball Receipt*', 'Pressure', 'Foul Committed', 'Foul Won', 
    'Ball Recovery', 'Block', 'Interception', 'Goal Keeper', 'Dribbled Past', 
    'Miscontrol', 'Dispossessed', 'Injury Stoppage', 'Substitution', 
    'Tactical Shift', 'Player Off', 'Player On', 'Shield', 'Bad Behaviour', 
    'Offside', 'Half End', 'Referee Ball-Drop'
]

# NEU: Liste für Eigentor-Typen zur einfachen Filterung
OWN_GOAL_TYPES = ['Own Goal For', 'Own Goal Against']

# NEU: Zonen-Zentroide für "Fuzzy C-Means"-Ansatz
# Koordinaten sind von 0-100 und werden auf 105x68 m Spielfeldgröße skaliert
_centroid_x_100 = [8.5, 25.25, 41.75, 58.25, 74.75, 91.5, 8.5, 25.25, 41.75, 58.25, 74.75, 91.5,
                   33.5, 66.5, 33.5, 66.5, 33.5, 66.5, 8.5, 91.5]
_centroid_y_100 = [89.45, 89.45, 89.45, 89.45, 89.45, 89.45, 10.55, 10.55, 10.55, 10.55, 10.55, 10.55,
                   71.05, 71.05, 50., 50., 28.95, 28.95, 50., 50.]

# Skalierte Zentroide für direkte Verwendung
ZONE_CENTROIDS_X = np.array([x * 105 / 100 for x in _centroid_x_100])
ZONE_CENTROIDS_Y = np.array([y * 68 / 100 for y in _centroid_y_100])
ZONE_CENTROIDS = {i: (ZONE_CENTROIDS_X[i], ZONE_CENTROIDS_Y[i]) for i in range(len(ZONE_CENTROIDS_X))}

# Sicherstellen, dass die Verzeichnisse existieren
os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(COMBINED_FILE), exist_ok=True)
os.makedirs(os.path.dirname(PREDICTION_FILE), exist_ok=True)