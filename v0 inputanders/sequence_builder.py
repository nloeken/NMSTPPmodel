# sequence_builder.py
import numpy as np
import pandas as pd
from tqdm import tqdm

# MODIFIZIERT: Wir definieren jetzt explizit kategoriale und kontinuierliche Features
CATEGORICAL_FEATURES = ['zone', 'act']
# NEU: Die 20 "Fuzzy Zone"-Zugehörigkeiten werden als kontinuierliche Features hinzugefügt
CONTINUOUS_FEATURES_BASE = [
    'zone_deltax', 'zone_deltay', 'zone_s', 'zone_sg', 'zone_thetag', 'deltaT_log1p'
] + [f'zone_degree_{i}' for i in range(20)]


def get_feature_cols(df):
    cont_cols = [c for c in CONTINUOUS_FEATURES_BASE if c in df.columns]
    sb360 = sorted([c for c in df.columns if c.startswith('sb360_')])
    return cont_cols, sb360

def build_sequences(df, seq_len=40):
    print("Baue Sequenzen für das Modell...")
    # WICHTIG: Wir brauchen eine konsistente numerische Kodierung für 'act'
    df['act_cat_code'] = df['act'].astype('category').cat.codes

    # Wir brauchen jetzt separate Listen für kategoriale und kontinuierliche Daten
    X_cat, X_cont, X_360 = [], [], []
    y_zone, y_act, y_time = [], [], []
    
    CONT_COLS, SB360_COLS = get_feature_cols(df)
    CAT_COLS = ['zone', 'act_cat_code'] # Wir verwenden die numerischen Codes

    for match_id, group in tqdm(df.groupby('match_id'), desc="Sequenzen bauen"):
        group = group.reset_index(drop=True)
        # Stelle sicher, dass die zu verwendenden Spalten keine NaNs enthalten
        group.loc[:, CONT_COLS + SB360_COLS] = group.loc[:, CONT_COLS + SB360_COLS].fillna(0)

        for i in range(len(group) - seq_len):
            # Input-Sequenz (die ersten 40 Events)
            seq_df = group.iloc[i:i+seq_len]
            # Ziel-Event (das 41. Event)
            next_row = group.iloc[i+seq_len]

            # Überspringe Sequenz, falls das Ziel-Label ungültig ist
            if pd.isna(next_row['zone']) or pd.isna(next_row['act_cat_code']) or pd.isna(next_row['deltaT_log1p']):
                continue
            
            # Trenne die Input-Daten
            X_cat.append(seq_df[CAT_COLS].values)
            X_cont.append(seq_df[CONT_COLS].values)
            X_360.append(seq_df[SB360_COLS].values)
            
            # Hänge die Ziel-Labels an
            y_zone.append(int(next_row['zone']))
            y_act.append(int(next_row['act_cat_code']))
            y_time.append(float(next_row['deltaT_log1p']))

    # Konvertiere alles in Numpy-Arrays
    X = {
        'cat': np.array(X_cat, dtype=np.int64),
        'cont': np.array(X_cont, dtype=np.float32),
        '360': np.array(X_360, dtype=np.float32)
    }
    y = {
        'zone': np.array(y_zone, dtype=np.int64),
        'act': np.array(y_act, dtype=np.int64),
        'time': np.array(y_time, dtype=np.float32)
    }
    
    # Wir geben auch den LabelEncoder für Actions zurück
    le_action = dict(enumerate(df['act'].astype('category').cat.categories))
    
    print(f"Sequenzbau abgeschlossen. {len(X['cat'])} Sequenzen erstellt.")
    return X, y, le_action