import os
import pandas as pd
import glob
from tqdm import tqdm
from config import MERGED_DIR, COMBINED_FILE, COLUMNS_TO_EVAL
from utils import safe_literal_eval

# function to load merged data file and preprocess for feature engineering
def preprocess():

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

    print(f"Final file successfully created at '{COMBINED_FILE}'.")