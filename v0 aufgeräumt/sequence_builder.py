# sequence_builder.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler # NEU

# ... (CATEGORICAL_FEATURES und CONTINUOUS_FEATURES_BASE bleiben gleich) ...
CATEGORICAL_FEATURES = ['zone', 'act']
CONTINUOUS_FEATURES_BASE = [
    'zone_deltax', 'zone_deltay', 'zone_s', 'zone_sg', 'zone_thetag', 'deltaT_log1p'
] + [f'zone_degree_{i}' for i in range(20)]

def get_feature_cols(df):
    cont_cols = [c for c in CONTINUOUS_FEATURES_BASE if c in df.columns]
    sb360 = sorted([c for c in df.columns if c.startswith('sb360_')])
    return cont_cols, sb360

def build_sequences(df, seq_len=40):
    print("Baue Sequenzen für das Modell...")
    df['act_cat_code'] = df['act'].astype('category').cat.codes

    CONT_COLS, SB360_COLS = get_feature_cols(df)
    CAT_COLS = ['zone', 'act_cat_code']

    # NEU: Feature Scaling anwenden (wie im Paper-Code)
    print("Skaliere kontinuierliche Features mit MinMaxScaler...")
    # Features, die zwischen -1 und 1 skaliert werden sollen
    scaler_neg_one = MinMaxScaler(feature_range=(-1, 1))
    scale_cols_neg_one = ['zone_deltax', 'zone_deltay']
    df[scale_cols_neg_one] = scaler_neg_one.fit_transform(df[scale_cols_neg_one])
    
    # Features, die zwischen 0 und 1 skaliert werden sollen
    scaler_zero_one = MinMaxScaler(feature_range=(0, 1))
    scale_cols_zero_one = [c for c in CONT_COLS if c not in scale_cols_neg_one]
    df[scale_cols_zero_one] = scaler_zero_one.fit_transform(df[scale_cols_zero_one])
    print("Skalierung abgeschlossen.")
    
    X_cat, X_cont, X_360 = [], [], []
    y_zone, y_act, y_time = [], [], []

    for match_id, group in tqdm(df.groupby('match_id'), desc="Sequenzen bauen"):
        group = group.reset_index(drop=True)
        group.loc[:, CONT_COLS + SB360_COLS] = group.loc[:, CONT_COLS + SB360_COLS].fillna(0)

        for i in range(len(group) - seq_len):
            seq_df = group.iloc[i:i+seq_len]
            next_row = group.iloc[i+seq_len]

            if pd.isna(next_row['zone']) or pd.isna(next_row['act_cat_code']) or pd.isna(next_row['deltaT_log1p']):
                continue
            
            X_cat.append(seq_df[CAT_COLS].values)
            X_cont.append(seq_df[CONT_COLS].values)
            X_360.append(seq_df[SB360_COLS].values)
            
            y_zone.append(int(next_row['zone']))
            y_act.append(int(next_row['act_cat_code']))
            # WICHTIG: Das Ziel y_time muss auch skaliert werden, wir verwenden hier das unskalierte Original
            # für die Loss-Berechnung, was in Ordnung ist, da der RMSE-Loss damit umgehen kann.
            y_time.append(float(next_row['deltaT_log1p'])) 

    # ... (Rest der Funktion bleibt gleich) ...
    X = { 'cat': np.array(X_cat, dtype=np.int64), 'cont': np.array(X_cont, dtype=np.float32), '360': np.array(X_360, dtype=np.float32) }
    y = { 'zone': np.array(y_zone, dtype=np.int64), 'act': np.array(y_act, dtype=np.int64), 'time': np.array(y_time, dtype=np.float32) }
    le_action = dict(enumerate(df['act'].astype('category').cat.categories))
    
    print(f"Sequenzbau abgeschlossen. {len(X['cat'])} Sequenzen erstellt.")
    return X, y, le_action