import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from utils import safe_parse_pass, extract_xy, parse_freezeframe
from config import DROP_EVENT_NAMES, ZONE_CENTROIDS_X, ZONE_CENTROIDS_Y, ZONE_CENTROIDS, SEQ_LEN

# --- Feature Columns Definition ---
CATEGORICAL_FEATURES = ['zone', 'act']
CONTINUOUS_FEATURES_BASE = [
    'zone_deltax', 'zone_deltay', 'zone_s', 'zone_sg', 'zone_thetag', 'deltaT_log1p'
] + [f'zone_degree_{i}' for i in range(len(ZONE_CENTROIDS))]

def add_features(df):

    print("Starting advanced feature engineering...")

    # Step 1: Rigorous Event Filtering (Noise Reduction)
    print(f"Rows before filtering: {len(df)}")
    df = df[~df['type_name'].isin(DROP_EVENT_NAMES)].copy().reset_index(drop=True)
    print(f"Rows after filtering event types: {len(df)}")

    # Step 2: Robust Time Calculation and Injury Time Filtering
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.time
    cutoff_time = pd.to_datetime("00:45:00.000").time()
    df = df[df['timestamp'] <= cutoff_time]
    print(f"Rows after filtering injury time: {len(df)}")
    
    df['time_seconds'] = df['timestamp'].apply(lambda t: t.hour*3600 + t.minute*60 + t.second + t.microsecond/1e6)
    seconds_in_first_half = 45 * 60
    df.loc[df['period'] == 2, 'time_seconds'] += seconds_in_first_half

    # Step 3: Detailed Action Categorization
    df['act'] = df['act'].astype('object') if 'act' in df.columns else ''
    df.loc[df['type_name'].isin(["Carry", "Duel", "Dribble", "50/50"]), 'act'] = 'Dribble'
    df.loc[df['type_name'] == "Shot", 'act'] = 'Shot'
    if 'error' in df.columns:
        df.loc[df['error'].notna(), 'act'] = 'Shot'
    
    pass_mask = df['type_name'].isin(["Pass", "Half Start", "Clearance"])
    df.loc[pass_mask, 'act'] = 'Pass'

    def is_cross(pass_val):
        pass_info = safe_parse_pass(pass_val)
        return pass_info.get("type", {}).get("name") == "Corner" or pass_info.get("cross", False)
    
    cross_mask = df['pass'].fillna('{}').apply(is_cross)
    df.loc[pass_mask & cross_mask, 'act'] = 'Cross'

    df = df[df['act'] != ''].copy()

    df = df.dropna(subset=['act']).reset_index(drop=True)
    print(f"Rows after action categorization and filtering empty actions: {len(df)}")

    # Step 4: Add "Possession End" Events
    print("Adding 'Possession End' events...")
    end_rows = [
        group.iloc[-1].copy() for _, group in tqdm(df.groupby(['match_id', 'possession']))
    ]
    if end_rows:
        end_df = pd.DataFrame(end_rows)
        end_df['time_seconds'] += 0.01
        end_df['act'] = 'Possession End'
        end_df['duration'] = 0.0
        df = pd.concat([df, end_df], ignore_index=True)
    
    df = df.sort_values(by=['match_id', 'period', 'time_seconds']).reset_index(drop=True)

    # Step 5: Extract and Normalize Coordinates
    coords = df['location'].apply(extract_xy).tolist()
    df[['x', 'y']] = pd.DataFrame(coords, index=df.index).dropna()
    df['x'] = df['x'] * 105 / 120
    df['y'] = df['y'] * 68 / 80

    # Step 6: Fuzzy C-Means Zoning
    print("Applying Fuzzy C-Means zoning...")
    x_coords = df[['x']].values
    y_coords = df[['y']].values
    distances = np.sqrt((x_coords - ZONE_CENTROIDS_X)**2 + (y_coords - ZONE_CENTROIDS_Y)**2)
    
    dist_sq = distances**2 + 1e-9
    inv_dist_sq_sum = (1 / dist_sq).sum(axis=1, keepdims=True)
    
    for i in range(len(ZONE_CENTROIDS)):
        degree = (1 / dist_sq[:, i]) / inv_dist_sq_sum.squeeze()
        df[f'zone_degree_{i}'] = degree
        
    df['zone'] = np.argmin(distances, axis=1).astype(int)

    # Step 7: Sequential Features based on Zone Centroids
    print("Calculating sequential features...")
    df['zone_x'] = df['zone'].map(lambda z: ZONE_CENTROIDS[z][0])
    df['zone_y'] = df['zone'].map(lambda z: ZONE_CENTROIDS[z][1])
    
    grouped = df.groupby(['match_id', 'period'])
    df['deltaT'] = grouped['time_seconds'].diff().fillna(0).clip(lower=0, upper=60)
    df['zone_deltax'] = grouped['zone_x'].diff().fillna(0)
    df['zone_deltay'] = grouped['zone_y'].diff().fillna(0)
    
    df['zone_s'] = np.sqrt(df['zone_deltax']**2 + df['zone_deltay']**2)
    
    goal_x, goal_y = 105, 34 # Goal center in 105x68m
    df['zone_sg'] = np.sqrt((df['zone_x'] - goal_x)**2 + (df['zone_y'] - goal_y)**2)
    df['zone_thetag'] = np.arctan2(df['zone_y'] - goal_y, df['zone_x'] - goal_x)
    
    # Step 8: 360 Freeze-Frame Features
    print("Processing 360 freeze-frame data...")
    ff_feats = df['freeze_frame'].apply(parse_freezeframe).tolist()
    sb_cols = [f'sb360_{feat}_{i}' for i in range(22) for feat in ['x', 'y', 'actor', 'teammate', 'keeper']]
    df = pd.concat([df, pd.DataFrame(ff_feats, columns=sb_cols, index=df.index)], axis=1)

    # Step 9: Stabilized Time Feature for the Model
    df['deltaT_log1p'] = np.log1p(df['deltaT'])

    print("Feature engineering complete.")
    return df.reset_index(drop=True)

def build_sequences(df):
    """
    Builds sequences from the feature-engineered DataFrame.
    Scales continuous features and prepares data for model input.
    """
    print("Building sequences for the model...")
    df['act_cat_code'] = df['act'].astype('category').cat.codes

    cont_cols = [c for c in CONTINUOUS_FEATURES_BASE if c in df.columns]
    sb360_cols = sorted([c for c in df.columns if c.startswith('sb360_')])
    cat_cols = ['zone', 'act_cat_code']

    # Apply feature scaling
    print("Scaling continuous features with MinMaxScaler...")
    df[cont_cols] = df[cont_cols].fillna(0)
    scaler_neg_one = MinMaxScaler(feature_range=(-1, 1))
    df[['zone_deltax', 'zone_deltay']] = scaler_neg_one.fit_transform(df[['zone_deltax', 'zone_deltay']])
    
    scaler_zero_one = MinMaxScaler(feature_range=(0, 1))
    scale_cols_zero_one = [c for c in cont_cols if c not in ['zone_deltax', 'zone_deltay']]
    df[scale_cols_zero_one] = scaler_zero_one.fit_transform(df[scale_cols_zero_one])
    print("Scaling complete.")
    
    X_cat, X_cont, X_360 = [], [], []
    y_zone, y_act, y_time = [], [], []

    for _, group in tqdm(df.groupby('match_id'), desc="Building sequences per match"):
        group = group.reset_index(drop=True)
        group.loc[:, cont_cols + sb360_cols] = group.loc[:, cont_cols + sb360_cols].fillna(0)

        for i in range(len(group) - SEQ_LEN):
            seq_df = group.iloc[i:i+SEQ_LEN]
            next_row = group.iloc[i+SEQ_LEN]
            
            X_cat.append(seq_df[cat_cols].values)
            X_cont.append(seq_df[cont_cols].values)
            X_360.append(seq_df[sb360_cols].values)
            
            y_zone.append(int(next_row['zone']))
            y_act.append(int(next_row['act_cat_code']))
            y_time.append(float(next_row['deltaT_log1p']))

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
    le_action = dict(enumerate(df['act'].astype('category').cat.categories))
    
    print(f"Sequence building complete. {len(X['cat'])} sequences created.")
    return X, y, le_action