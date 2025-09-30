# data_loading.py
import os
import pandas as pd
from tqdm import tqdm
from config import EVENTS_DIR, THREE_SIXTY_DIR, MERGED_DIR
from utils import extract_name

def load_and_merge():

    event_files = [f for f in os.listdir(EVENTS_DIR) if f.endswith('.json')]
    match_ids = [os.path.splitext(f)[0] for f in event_files]

    print(f"Found {len(match_ids)} matches to process.")
    for match_id in tqdm(match_ids, desc="Stage 1: Merging raw match data"):
        try:
            events_path = os.path.join(EVENTS_DIR, f'{match_id}.json')
            positional_path = os.path.join(THREE_SIXTY_DIR, f'{match_id}.json')

            # Skip matches that are missing 360 data
            if not os.path.exists(positional_path):
                print(f"[{match_id}] 360 data not found. Match skipped.")
                continue

            events_df = pd.read_json(events_path)
            positional_df = pd.read_json(positional_path)

            # Merge the event and positional dataframes
            df = pd.merge(events_df, positional_df, left_on='id', right_on='event_uuid', how='inner')

            # Extract human-readable names from dictionary columns
            df["type_name"] = df["type"].apply(extract_name)
            df["player_name"] = df["player"].apply(extract_name)
            df["team_name"] = df["possession_team"].apply(extract_name)
            df["position_name"] = df["position"].apply(extract_name)
            df["match_id"] = match_id
            
            # Keep only the necessary columns for the next steps
            keep_cols = [
                "id", "index", "period", "timestamp", "minute", "second", "duration", "type", 
                "type_name", "team_name", "position_name", "possession", "possession_team", 
                "player_name", "location", "pass", "carry", "dribble", "shot", "duel", 
                "clearance", "error", "freeze_frame", "match_id"
            ]
            df = df[[col for col in keep_cols if col in df.columns]]
            
            # Sort chronologically
            df = df.sort_values(by=['period', 'minute', 'second', 'index']).reset_index(drop=True)

            # Save the merged dataframe as a CSV
            out_path = os.path.join(MERGED_DIR, f"contextualevents_{match_id}.csv")
            df.to_csv(out_path, index=False)

        except Exception as e:
            print(f"[{match_id}] An error occurred: {e}")
    print("\nStage 1: Merging raw data complete.")