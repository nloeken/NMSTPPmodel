# feature_engineering.py
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from v0.utils import safe_parse_pass
from v0.config import MAIN_EVENT_TYPES

def extract_name(field):
    if isinstance(field, dict):
        return field.get("name")
    return field

# NEU: Implementierung der "Juego de Posición"-Zonierung gemäß Paper/Repo
# Ersetzt die alte, uniforme Grid-Funktion.
# Diese Funktion teilt das Spielfeld in 20 taktisch relevante Zonen ein.
def assign_juego_de_posicion_zone(x, y):
    if pd.isna(x) or pd.isna(y):
        return np.nan

    # Eigene Hälfte
    if x <= 60:
        if y <= 13.84: return 15
        if y <= 27.68: return 10
        if y <= 52.32: return 11
        if y <= 66.16: return 12
        else: return 16
    # Gegnerische Hälfte
    else:
        # Zone vor dem Strafraum
        if x <= 80:
            if y <= 13.84: return 9
            if y <= 27.68: return 5
            if y <= 52.32: return 0
            if y <= 66.16: return 6
            else: return 13
        # Strafraum-Bereich
        elif x <= 100:
            if y <= 13.84: return 9
            if y <= 27.68: return 7
            if y <= 52.32: return 2
            if y <= 66.16: return 8
            else: return 13
        # Nähe der Grundlinie
        else:
            if y <= 13.84: return 19
            if y <= 27.68: return 17
            if y <= 52.32: return 4
            if y <= 66.16: return 18
            else: return 14

def add_features(df):
    # --------------------
    # Filter nur Haupt-Events
    # --------------------
    df = df[df['type_name'].isin(MAIN_EVENT_TYPES)].copy().reset_index(drop=True)

    # --------------------
    # Zeitstempel in Sekunden für Sortierung
    # --------------------
    if 'time_seconds' not in df.columns:
        df['time_seconds'] = df['minute']*60 + df['second']

    # --------------------
    # Action Kategorien (Schritt 1)
    # --------------------
    def get_action_cat(row):
        type_name = row.get("type_name", "")
        if type_name in ["Carry", "Duel", "Dribble", "50/50"]:
            return "Dribble"
        if type_name == "Shot":
            return "Shot"
        if type_name in ["Pass", "Half Start", "Clearance"]:
            pass_info = safe_parse_pass(row.get("pass", {}))
            if pass_info.get("type", {}).get("name", "") == "Corner":
                return "Cross"
            if pass_info.get("cross", False):
                return "Cross"
            return "Pass"
        return None

    df['action_cat'] = df.apply(get_action_cat, axis=1)
    df = df.dropna(subset=['action_cat']).reset_index(drop=True)

    # NEU: "Possession End"-Event hinzufügen (gemäß Paper)
    # Dies ist ein entscheidender Schritt, um dem Modell das Ende von Sequenzen zu signalisieren.
    print("Füge 'Possession End'-Events hinzu...")
    end_possession_rows = []
    # Gruppieren nach Spiel und Ballbesitzphase
    for _, group in tqdm(df.groupby(['match_id', 'possession'])):
        last_event = group.iloc[-1].copy()
        # Erstelle ein neues Event, das direkt nach dem letzten echten Event stattfindet
        last_event['time_seconds'] += 0.01 
        last_event['action_cat'] = 'Possession End'
        # Setze Dauer auf 0, da es ein künstliches Event ist
        last_event['duration'] = 0.0 
        end_possession_rows.append(last_event)
    
    # Kombiniere die neuen Events mit dem originalen DataFrame
    end_rows_df = pd.DataFrame(end_possession_rows)
    df = pd.concat([df, end_rows_df], ignore_index=True)
    
    # Erneutes Sortieren nach der Zeit, um die neuen Events korrekt einzufügen
    df = df.sort_values(by=['match_id', 'period', 'time_seconds']).reset_index(drop=True)
    df['act'] = df['action_cat']


    # --------------------
    # x,y Koordinaten
    # --------------------
    def extract_xy(loc):
        if isinstance(loc, str):
            try: loc = ast.literal_eval(loc)
            except: return [np.nan, np.nan]
        if isinstance(loc, list) and len(loc) >= 2: return loc[:2]
        if isinstance(loc, dict) and "x" in loc and "y" in loc: return [loc["x"], loc["y"]]
        return [np.nan, np.nan]

    coords = df['location'].apply(extract_xy).tolist()
    df[['x', 'y']] = pd.DataFrame(coords, index=df.index)
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    
    # Droppe Zeilen, wo keine Koordinaten extrahiert werden konnten
    df.dropna(subset=['x', 'y'], inplace=True)


    # MODIFIZIERT: Wende die neue Zonierungsfunktion an
    print("Wende 'Juego de Posición'-Zonierung an...")
    df['zone'] = df.apply(lambda r: assign_juego_de_posicion_zone(r['x'], r['y']), axis=1)
    df.dropna(subset=['zone'], inplace=True) # Zonen, die NaN sind, entfernen
    df['zone'] = df['zone'].astype(int)

    # --------------------
    # Sequenzielle Features (Delta Time, Zone Deltas)
    # Diese müssen nach dem Hinzufügen der 'Possession End'-Events berechnet werden
    # --------------------
    print("Berechne sequenzielle Features (deltaT, deltaZone)...")
    df['deltaT'] = df.groupby('match_id')['time_seconds'].diff().fillna(0)
    
    # Negative Zeitdifferenzen (z.B. bei Halbzeitwechsel) auf 0 setzen
    df.loc[df['deltaT'] < 0, 'deltaT'] = 0

    df['zone_s'] = df['zone']
    df['zone_deltax'] = df.groupby('match_id')['x'].diff().fillna(0)
    df['zone_deltay'] = df.groupby('match_id')['y'].diff().fillna(0)
    df['zone_sg'] = np.sqrt(df['zone_deltax']**2 + df['zone_deltay']**2)
    df['zone_thetag'] = np.arctan2(df['zone_deltay'], df['zone_deltax'])

    # --------------------
    # 360° Freeze-Frame Features
    # --------------------
    print("Verarbeite 360 Freeze-Frame Daten...")
    max_players = 22
    sb_cols = [f'sb360_{feat}_{i}' for i in range(max_players) for feat in ['x', 'y', 'actor', 'teammate', 'keeper']]

    def parse_freezeframe(ff_val):
        ff = ff_val
        if isinstance(ff, str):
            try: ff = ast.literal_eval(ff)
            except: ff = None
        
        if not isinstance(ff, list):
            return [0.0] * (5 * max_players)

        features = []
        for i in range(max_players):
            if i < len(ff) and isinstance(ff[i], dict):
                p = ff[i]
                loc = p.get('location', [0.0, 0.0])
                features.extend([loc[0], loc[1], float(p.get('actor', False)), float(p.get('teammate', False)), float(p.get('keeper', False))])
            else:
                features.extend([0.0] * 5)
        return features

    ff_feats = df['freeze_frame'].apply(parse_freezeframe).tolist()
    sb_df = pd.DataFrame(ff_feats, columns=sb_cols, index=df.index)
    df = pd.concat([df, sb_df], axis=1)
    df.attrs['sb360_cols'] = sb_cols

    # --------------------
    # Stabilisiertes Zeit-Feature für das Modell
    # --------------------
    df['deltaT_stable'] = df['deltaT'].clip(lower=0.0)
    df['deltaT_log1p'] = np.log1p(df['deltaT_stable'])

    print("Feature Engineering abgeschlossen.")
    return df.reset_index(drop=True)