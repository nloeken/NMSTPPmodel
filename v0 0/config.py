import os


# ---- Fast/Debug Switches ----
FAST_DEBUG = False # kurze Läufe / kleiner Datensatz
FAST_MAX_MATCHES = 30 # max. Anzahl Matches laden
FAST_MAX_ROWS = 300_000#400_000 # truncate combined csv
EPOCHS = 4 if FAST_DEBUG else 25
BATCH_SIZE = 32 if FAST_DEBUG else 256
SEQ_LEN = 40 # 
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PATIENCE = 4


# ---- Pfade ----
BASE_DIR = r"C:\Users\nloeken\StatsbombData"
EVENTS_DIR = os.path.join(BASE_DIR, 'data/events/')
THREE_SIXTY_DIR = os.path.join(BASE_DIR, '\three-sixty')
MERGED_DIR = os.path.join(BASE_DIR, 'merged/')
COMBINED_FILE = r"C:\Users\nloeken\StatsbombData\combined_all.csv"
SAMPLE_FILE = os.path.join(BASE_DIR, 'combined/combined_sample.csv')
PREDICTION_FILE = os.path.join(BASE_DIR, 'predictions/predictions.csv')
SAMPLE_PREDICTION_FILE = os.path.join(BASE_DIR, 'predictions/sample_predictions.csv')


# ---- Konstante / Filter ----
MAX_EVENT_GAP = 10 # Sek.
# Eventtypen (StatsBomb) – ähnlich wie im Repo
MAIN_EVENT_TYPES = ['Half Start', 'Pass', 'Clearance', 'Carry', 'Duel', 'Dribble', '50/50', 'Shot']

os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(COMBINED_FILE), exist_ok=True)