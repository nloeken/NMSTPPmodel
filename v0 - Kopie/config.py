# config.py
import os

# ---- Fast/Debug Switches ----
# Für die Optuna-Suche sollte FAST_DEBUG auf False stehen, 
# aber die Datenmenge kann hier zum Testen noch begrenzt werden.
FAST_DEBUG = False 
FAST_MAX_MATCHES = 50 # Begrenzt die Anzahl der geladenen Spiele
FAST_MAX_ROWS = 400_000 # Begrenzt die Anzahl der Zeilen nach dem Laden

# ---- Feste Parameter für die Studie ----
EPOCHS = 25 # Maximale Anzahl an Epochen (EarlyStopping greift vorher)
PATIENCE = 4 # Patience für EarlyStopping
SEQ_LEN = 40 
GRAD_CLIP = 1.0

# ---- Pfade ----
BASE_DIR = r"C:\Users\nloeken\StatsbombData"
EVENTS_DIR = os.path.join(BASE_DIR, 'data/events/')
THREE_SIXTY_DIR = os.path.join(BASE_DIR, '\three-sixty')
MERGED_DIR = os.path.join(BASE_DIR, 'merged/')
COMBINED_FILE = r"C:\Users\nloeken\StatsbombData\combined_all.csv"
SAMPLE_FILE = os.path.join(BASE_DIR, 'combined/combined_sample.csv')
PREDICTION_FILE = os.path.join(BASE_DIR, 'predictions/predictions.csv')
SAMPLE_PREDICTION_FILE = os.path.join(BASE_DIR, 'predictions/sample_predictions.csv')

# ---- Event-Filter ----
MAIN_EVENT_TYPES = ['Half Start', 'Pass', 'Clearance', 'Carry', 'Duel', 'Dribble', '50/50', 'Shot']

os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(COMBINED_FILE), exist_ok=True)