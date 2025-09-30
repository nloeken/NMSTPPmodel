# feature_engineering.py
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from utils import safe_parse_pass
from config import DROP_EVENT_NAMES, ZONE_CENTROIDS_X, ZONE_CENTROIDS_Y, ZONE_CENTROIDS

def add_features(df):
    print("Starte erweitertes Feature Engineering...")

    # --------------------
    # Schritt 1: Rigoroses Filtern von Events (Rauschen reduzieren)
    # --------------------
    print(f"Zeilen vor dem Filtern: {len(df)}")
    df = df[~df['type_name'].isin(DROP_EVENT_NAMES)].copy().reset_index(drop=True)
    print(f"Zeilen nach dem Filtern von Event-Typen: {len(df)}")

    # --------------------
    # Schritt 2: Robuste Zeitberechnung und Filtern der Nachspielzeit
    # --------------------
    # Konvertiere 'timestamp' in ein Zeitobjekt
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.time
    # Entferne Events nach der 45. Minute jeder Halbzeit
    cutoff_time = pd.to_datetime("00:45:00.000").time()
    df = df[df['timestamp'] <= cutoff_time]
    print(f"Zeilen nach dem Filtern der Nachspielzeit: {len(df)}")

    # Erstelle eine absolute Sekundenanzahl, die die 2. Halbzeit berücksichtigt
    df['time_seconds'] = df['timestamp'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)
    # Addiere 45 Minuten * 60 Sekunden für die zweite Halbzeit
    # Annahme: Zeitstempel der 2. HZ beginnen wieder bei 0.
    # Wir benutzen 'period' zur Unterscheidung.
    # Zeit in 2. HZ = Zeit in 2. HZ + maximale Zeit der 1. HZ (konservativ 45*60)
    seconds_in_first_half = 45 * 60
    df.loc[df['period'] == 2, 'time_seconds'] += seconds_in_first_half

    # --------------------
    # Schritt 3: Detaillierte Aktions-Kategorisierung
    # --------------------
    df['act'] = np.nan
    # Dribble-Aktionen
    dribble_types = ["Carry", "Duel", "Dribble", "50/50"]
    df.loc[df['type_name'].isin(dribble_types), 'act'] = 'Dribble'
    # Schuss-Aktionen (inklusive Fehler, die zu Schüssen führen)
    shot_types = ["Shot"]
    df.loc[df['type_name'].isin(shot_types), 'act'] = 'Shot'
    if 'error' in df.columns:
        df.loc[df['error'].notna(), 'act'] = 'Shot' # Gemäß Paper-Logik
    # Pass-Aktionen
    pass_types = ["Pass", "Half Start", "Clearance"]
    pass_mask = df['type_name'].isin(pass_types)
    df.loc[pass_mask, 'act'] = 'Pass'
    # Flanken-Aktionen (überschreibt "Pass")
    def is_cross(pass_val):
        pass_info = safe_parse_pass(pass_val)
        is_corner = pass_info.get("type", {}).get("name") == "Corner"
        is_cross_flag = pass_info.get("cross", False)
        return is_corner or is_cross_flag
    
    cross_mask = df['pass'].apply(is_cross)
    df.loc[pass_mask & cross_mask, 'act'] = 'Cross'
    
    df = df.dropna(subset=['act']).reset_index(drop=True)
    print(f"Zeilen nach Aktions-Kategorisierung: {len(df)}")

    # --------------------
    # Schritt 4: "Possession End"-Event hinzufügen
    # --------------------
    print("Füge 'Possession End'-Events hinzu...")
    end_possession_rows = []
    df = df.sort_values(by=['match_id', 'period', 'possession', 'time_seconds'])
    for _, group in tqdm(df.groupby(['match_id', 'possession'])):
        last_event = group.iloc[-1].copy()
        last_event['time_seconds'] += 0.01 
        last_event['act'] = 'Possession End'
        last_event['duration'] = 0.0 
        end_possession_rows.append(last_event)
    
    end_rows_df = pd.DataFrame(end_possession_rows)
    df = pd.concat([df, end_rows_df], ignore_index=True)
    df = df.sort_values(by=['match_id', 'period', 'time_seconds']).reset_index(drop=True)

    # --------------------
    # Schritt 5: Koordinaten extrahieren und auf 105x68m normalisieren
    # --------------------
    def extract_xy(loc):
        if isinstance(loc, str):
            try: loc = ast.literal_eval(loc)
            except: return [np.nan, np.nan]
        return loc[:2] if isinstance(loc, list) and len(loc) >= 2 else [np.nan, np.nan]

    coords = df['location'].apply(extract_xy).tolist()
    df[['x', 'y']] = pd.DataFrame(coords, index=df.index)
    df.dropna(subset=['x', 'y'], inplace=True)
    
    # Konvertiere von StatsBomb-Koordinaten (120x80) zu Standard-Metern (105x68)
    df['x'] = df['x'] * 105 / 120
    df['y'] = df['y'] * 68 / 80

    # --------------------
    # Schritt 6: NEU - Fuzzy C-Means Zoning
    # --------------------
    print("Wende 'Fuzzy C-Means'-Zonierung an...")
    # Berechne Distanz zu jedem Zonen-Zentroid (vektorisiert)
    x_coords = df[['x']].values
    y_coords = df[['y']].values
    distances = np.sqrt((x_coords - ZONE_CENTROIDS_X)**2 + (y_coords - ZONE_CENTROIDS_Y)**2)
    
    for i in range(len(ZONE_CENTROIDS_X)):
        df[f'zone_dist_{i}'] = distances[:, i]
        
    # Berechne Zugehörigkeitsgrad zu jeder Zone (vektorisiert)
    # Vermeide Division durch Null, falls eine Distanz exakt 0 ist
    dist_sq = distances**2 + 1e-9
    inv_dist_sq_sum = (1 / dist_sq).sum(axis=1)
    
    for i in range(len(ZONE_CENTROIDS_X)):
        degree = (1 / dist_sq[:, i]) / inv_dist_sq_sum
        df[f'zone_degree_{i}'] = degree
    
    # Bestimme die "harte" Zone mit der höchsten Zugehörigkeit
    df['zone'] = np.argmax(distances, axis=1)
    df['zone'] = df['zone'].astype(int)

    # --------------------
    # Schritt 7: MODIFIZIERT - Sequenzielle Features basierend auf Zonen-Zentroiden
    # --------------------
    print("Berechne verbesserte sequenzielle Features...")
    # Map zone to its centroid coordinates
    df['zone_x'] = df['zone'].map(lambda z: ZONE_CENTROIDS[z][0])
    df['zone_y'] = df['zone'].map(lambda z: ZONE_CENTROIDS[z][1])
    
    grouped = df.groupby(['match_id', 'period'])
    df['deltaT'] = grouped['time_seconds'].diff().fillna(0)
    df['zone_deltax'] = grouped['zone_x'].diff().fillna(0)
    df['zone_deltay'] = grouped['zone_y'].diff().fillna(0)
    
    # Negative Zeitdifferenzen (sollten durch Gruppierung vermieden werden) auf 0 setzen
    df.loc[df['deltaT'] < 0, 'deltaT'] = 0
    df['deltaT'].clip(upper=60, inplace=True) # Max inter-event time auf 60s begrenzen

    df['zone_s'] = np.sqrt(df['zone_deltax']**2 + df['zone_deltay']**2)
    
    goal_x, goal_y = 105, 34 # Tor-Mittelpunkt in 105x68m
    df['zone_sg'] = np.sqrt((df['zone_x'] - goal_x)**2 + (df['zone_y'] - goal_y)**2)
    df['zone_thetag'] = np.arctan2(df['zone_y'] - goal_y, df['zone_x'] - goal_x)
    
    # --------------------
    # Schritt 8: 360° Freeze-Frame Features (mit Koordinaten-Normalisierung)
    # --------------------
    print("Verarbeite 360 Freeze-Frame Daten...")
    max_players = 22
    sb_cols = [f'sb360_{feat}_{i}' for i in range(max_players) for feat in ['x', 'y', 'actor', 'teammate', 'keeper']]

    def parse_freezeframe(ff_val):
        ff = ff_val
        if isinstance(ff, str):
            try: ff = ast.literal_eval(ff)
            except: ff = None
        
        if not isinstance(ff, list): return [0.0] * (5 * max_players)

        features = []
        for i in range(max_players):
            if i < len(ff) and isinstance(ff[i], dict):
                p = ff[i]
                loc = p.get('location', [0.0, 0.0])
                # Koordinaten hier ebenfalls normalisieren
                norm_x = loc[0] * 105 / 120
                norm_y = loc[1] * 68 / 80
                features.extend([norm_x, norm_y, float(p.get('actor', False)), float(p.get('teammate', False)), float(p.get('keeper', False))])
            else:
                features.extend([0.0] * 5)
        return features

    ff_feats = df['freeze_frame'].apply(parse_freezeframe).tolist()
    sb_df = pd.DataFrame(ff_feats, columns=sb_cols, index=df.index)
    df = pd.concat([df, sb_df], axis=1)

    # --------------------
    # Schritt 9: Stabilisiertes Zeit-Feature für das Modell
    # --------------------
    df['deltaT_log1p'] = np.log1p(df['deltaT'])

    print("Feature Engineering abgeschlossen.")
    return df.reset_index(drop=True)