# data_loading.py
import os
import pandas as pd
from tqdm import tqdm
from config import EVENTS_DIR, THREE_SIXTY_DIR, MERGED_DIR
from utils import extract_name

# function to sequentially load and merge event and positional data
def load_and_merge():

    event_files = [f for f in os.listdir(EVENTS_DIR) if f.endswith('.json')]
    match_ids = [os.path.splitext(f)[0] for f in event_files]

    for match_id in tqdm(match_ids, desc="Stage 1: Merging raw match data"):
        try:
            events_path = os.path.join(EVENTS_DIR, f'{match_id}.json')
            positional_path = os.path.join(THREE_SIXTY_DIR, f'{match_id}.json')

            # skip matches only having event data
            if not os.path.exists(positional_path):
                continue

            events_df = pd.read_json(events_path)
            positional_df = pd.read_json(positional_path)

            # merge the event and positional df
            df = pd.merge(
                events_df,
                positional_df,
                left_on='id',
                right_on='event_uuid',
                how='inner'
            )

            # extract and store names seperately 
            df["type_name"] = df["type"].apply(extract_name)
            df["player_name"] = df["player"].apply(extract_name)
            df["team_name"] = df["possession_team"].apply(extract_name)
            df["position_name"] = df["position"].apply(extract_name)
            
            # add column match_id
            df["match_id"] = match_id
            
            # only keep needed columns
            keep_cols = [
                "id", "index", "period", "timestamp", "minute", "second", "duration", "type", 
                "type_name", "team_name", "position_name", "possession", "possession_team", 
                "player_name", "location", "pass", "carry", "dribble", "shot", "duel", 
                "clearance", "error", "freeze_frame", "match_id"
            ]
            df = df[[col for col in keep_cols if col in df.columns]]
            
            # Sort chronologically
            df = df.sort_values(by=['period', 'minute', 'second', 'index']).reset_index(drop=True)

            # save merged df as csv
            out_path = os.path.join(MERGED_DIR, f"contextualevents_{match_id}.csv")
            df.to_csv(out_path, index=False)
            print(f"[{match_id}] saved at {out_path}")

        except Exception as e:
            print(f"[{match_id}] Error: {e}")