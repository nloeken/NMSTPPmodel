# sequence_builder.py
import numpy as np
import pandas as pd

# MODIFIZIERT: Wir definieren jetzt explizit kategoriale und kontinuierliche Features
CATEGORICAL_FEATURES = ['zone', 'act']
CONTINUOUS_FEATURES_BASE = ['zone_deltax', 'zone_deltay', 'zone_sg', 'zone_thetag', 'deltaT_log1p']

def get_feature_cols(df):
    cont_cols = [c for c in CONTINUOUS_FEATURES_BASE if c in df.columns]
    sb360 = sorted([c for c in df.columns if c.startswith('sb360_')])
    return cont_cols, sb360

def build_sequences(df, seq_len=40):
    # WICHTIG: Wir brauchen eine konsistente numerische Kodierung für 'act'
    df['act_cat_code'] = df['act'].astype('category').cat.codes

    # Wir brauchen jetzt separate Listen für kategoriale und kontinuierliche Daten
    X_cat, X_cont, X_360 = [], [], []
    y_zone, y_act, y_time = [], [], []
    
    CONT_COLS, SB360_COLS = get_feature_cols(df)
    CAT_COLS = ['zone', 'act_cat_code'] # Wir verwenden die numerischen Codes

    for match_id, group in df.groupby('match_id'):
        group = group.reset_index(drop=True)
        for i in range(len(group) - seq_len):
            # Input-Sequenz (die ersten 40 Events)
            seq_df = group.iloc[i:i+seq_len]
            # Ziel-Event (das 41. Event)
            next_row = group.iloc[i+seq_len]

            if np.any(pd.isna(next_row[CAT_COLS + ['deltaT_log1p']])):
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
        'cat': np.array(X_cat),
        'cont': np.array(X_cont),
        '360': np.array(X_360)
    }
    y = {
        'zone': np.array(y_zone),
        'act': np.array(y_act),
        'time': np.array(y_time)
    }
    
    # Wir geben auch den LabelEncoder für Actions zurück
    le_action = dict(enumerate(df['act'].astype('category').cat.categories))
    
    return X, y, le_action