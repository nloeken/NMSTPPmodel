import pandas as pd
import os
import glob
from tqdm import tqdm
from utils import safe_literal_eval
from config import MERGED_DIR, COMBINED_FILE, COLUMNS_TO_EVAL

def combine_and_process_csvs():
    """
    (Pipeline Stage 2)
    Combines all individual match CSVs into a single large file.
    It processes string-encoded columns back into Python objects in a 
    memory-efficient, streaming manner.
    """
    # Find all intermediate CSV files
    all_files = sorted(glob.glob(os.path.join(MERGED_DIR, "*.csv")))
    if not all_files:
        print(f"ERROR: No CSV files found in '{MERGED_DIR}'. Please run the data loading step first (e.g. `python run_pipeline.py --load`).")
        return

    # Ensure the target file does not already exist to prevent appending to an old file
    if os.path.exists(COMBINED_FILE):
        os.remove(COMBINED_FILE)
        print(f"Removed existing combined file at '{COMBINED_FILE}'.")
    
    print(f"Found {len(all_files)} CSV files to combine into a single file.")

    # --- Process the first file to create the header ---
    print(f"Processing first file: {os.path.basename(all_files[0])}")
    first_df = pd.read_csv(all_files[0])
    for col in COLUMNS_TO_EVAL:
        if col in first_df.columns:
            first_df[col] = first_df[col].apply(safe_literal_eval)
            
    # Write the processed first file WITH header to the new combined file
    first_df.to_csv(COMBINED_FILE, index=False)
    del first_df # Free up memory immediately

    # --- Process and append the remaining files ---
    for filename in tqdm(all_files[1:], desc="Stage 2: Combining and preprocessing CSVs"):
        df = pd.read_csv(filename)
        for col in COLUMNS_TO_EVAL:
            if col in df.columns:
                df[col] = df[col].apply(safe_literal_eval)
        
        # Append the processed dataframe WITHOUT header to the combined file
        df.to_csv(COMBINED_FILE, mode='a', header=False, index=False)
        del df # Free up memory

    print("\nStage 2: Combining and preprocessing complete.")
    print(f"Final file successfully created at '{COMBINED_FILE}'.")

# Spalten, auf die die safe_eval-Funktion angewendet werden soll
COLUMNS_TO_EVAL = [
    "type", "pass", "clearance", "shot", "50/50", 
    "carry", "dribble", "duel", "freeze_frame"
]

def preprocess():
    """
    Kombiniert CSV-Dateien speichereffizient, indem sie einzeln gelesen,
    verarbeitet und an die Zieldatei angehängt werden.
    """
    # Stelle sicher, dass die Zieldatei nicht bereits existiert
    if os.path.exists(COMBINED_FILE):
        os.remove(COMBINED_FILE)
        print(f"Alte Datei '{COMBINED_FILE}' wurde gelöscht.")

    all_files = sorted(glob.glob(os.path.join(MERGED_DIR, "*.csv")))

    if not all_files:
        print(f"FEHLER: Keine CSV-Dateien im Ordner '{MERGED_DIR}' gefunden.")
        return

    print(f"{len(all_files)} CSV-Dateien gefunden, die kombiniert werden.")

    # --- Verarbeitung der ersten Datei ---
    print(f"Verarbeite und schreibe die erste Datei: {os.path.basename(all_files[0])}")
    first_df = pd.read_csv(all_files[0])
    
    # Wende deine `safe_eval`-Logik an
    for col in COLUMNS_TO_EVAL:
        if col in first_df.columns:
            first_df[col] = first_df[col].apply(safe_eval)
            
    # Schreibe die verarbeitete erste Datei MIT dem Header in die Zieldatei
    first_df.to_csv(COMBINED_FILE, index=False)
    del first_df # Gib Speicher sofort frei

    # --- Verarbeitung der restlichen Dateien ---
    print("Verarbeite und hänge die restlichen Dateien an...")
    for filename in tqdm(all_files[1:]):
        df = pd.read_csv(filename)
        
        # Wende deine `safe_eval`-Logik an
        for col in COLUMNS_TO_EVAL:
            if col in df.columns:
                df[col] = df[col].apply(safe_eval)
        
        # Hänge das verarbeitete DataFrame OHNE Header an die Zieldatei an
        df.to_csv(COMBINED_FILE, mode='a', header=False, index=False)
        del df # Gib Speicher sofort frei

    print(f"\nKombinieren der Daten abgeschlossen.")
    print(f"Finale Datei wurde erfolgreich unter '{COMBINED_FILE}' erstellt.")
