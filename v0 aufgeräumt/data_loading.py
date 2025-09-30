import os
import pandas as pd
from tqdm import tqdm
from utils import extract_name
from config import EVENTS_DIR, THREE_SIXTY_DIR, MERGED_DIR, FAST_DEBUG, FAST_MAX_MATCHES, OWN_GOAL_TYPES

# function to sequentially load and merge event and positional data
def load_and_merge():
    event_files = [f for f in os.listdir(EVENTS_DIR) if f.endswith('.json')]
    if FAST_DEBUG:
        event_files = event_files[:FAST_MAX_MATCHES]
    match_ids = [os.path.splitext(f)[0] for f in event_files]

    for match_id in tqdm(match_ids, desc='Processing matches'):
        try:
            events_path = os.path.join(EVENTS_DIR, f'{match_id}.json')
            positional_path = os.path.join(THREE_SIXTY_DIR, f'{match_id}.json')

            # skip matches only having event data
            if not os.path.exists(positional_path):
                continue

            events_df = pd.read_json(events_path)
            positional_df = pd.read_json(positional_path)
            df = pd.merge(events_df, positional_df, left_on='id', right_on='event_uuid', how='inner')

            # Namen extrahieren
            for col, newcol in [
                ('type','type_name'), ('player','player_name'), 
                ('possession_team','team_name'), ('position','position_name')
            ]:
                df[newcol] = df[col].apply(extract_name)

            # MODIFIZIERT: Spiele mit Eigentoren werden komplett übersprungen
            if df['type_name'].isin(OWN_GOAL_TYPES).any():
                print(f'[{match_id}] Own goal found. Match skipped.')
                continue

            df['match_id'] = match_id
            
            # MODIFIZIERT: Das aggressive Filtern wird in feature_engineering.py verschoben.
            # Hier behalten wir alle Spalten, die potenziell nützlich sind.
            keep_cols = [
                'id', 'index', 'period', 'timestamp', 'minute', 'second', 'duration', 
                'type', 'type_name', 'team', 'team_name', 'position', 'position_name', 
                'possession', 'possession_team', 'player', 'player_name', 'location', 
                'pass', 'carry', 'dribble', 'shot', 'duel', 'clearance', 'freeze_frame', 
                'under_pressure', 'off_camera', 'match_id', 'error' # NEU: 'error' hinzugefügt für Shot-Kategorisierung
            ]
            df = df[[c for c in keep_cols if c in df.columns]]
            df = df.sort_values(by=['period', 'minute', 'second', 'index']).reset_index(drop=True)

            out_path = os.path.join(MERGED_DIR, f'contextualevents_{match_id}.csv')
            df.to_csv(out_path, index=False)
            print(f'[{match_id}] saved at {out_path}')
        except Exception as e:
            print(f'[{match_id}] Error: {e}')