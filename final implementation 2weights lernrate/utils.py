# utils.py
# hier geändert: mulitplitkaton bei class weights 
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def extract_name(field):
    """
    Extracts the 'name' key from a dictionary field.
    If the field is not a dict, returns the field itself.
    """
    if isinstance(field, dict):
        return field.get("name")
    return field

def safe_literal_eval(val):
    """
    Safely evaluate a string containing a Python literal (dict, list, etc.).
    Returns the original value if it's not a string or if evaluation fails.
    """
    if isinstance(val, str):
        try:
            # ast.literal_eval is a safe way to evaluate strings containing Python literals
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            # If it's not a valid literal (e.g., just a regular string), return it as is
            return val
    return val

def safe_parse_pass(val):
    """
    Safely parses a string that should be a dictionary (e.g., from a CSV).
    Returns an empty dict if parsing fails or input is not a string/dict.
    """
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return {}
    return {}

def extract_xy(loc):
    """
    Extracts x and y coordinates from a location field.
    Handles both list and string-formatted list inputs.
    """
    if isinstance(loc, str):
        try:
            loc = ast.literal_eval(loc)
        except (ValueError, SyntaxError):
            return [np.nan, np.nan]
    return loc[:2] if isinstance(loc, list) and len(loc) >= 2 else [np.nan, np.nan]
    
def parse_freezeframe(ff_val, max_players=22):
    """
    Parses the 'freeze_frame' data into a flat feature vector.
    Normalizes coordinates from 120x80 to 105x68.
    Pads with zeros if fewer than max_players are present.
    """
    ff = ff_val
    if isinstance(ff, str):
        try:
            ff = ast.literal_eval(ff)
        except (ValueError, SyntaxError):
            ff = None
    
    if not isinstance(ff, list):
        return [0.0] * (5 * max_players)

    features = []
    for i in range(max_players):
        if i < len(ff) and isinstance(ff[i], dict):
            p = ff[i]
            loc = p.get('location', [0.0, 0.0])
            # Normalize coordinates from StatsBomb (120x80) to standard (105x68)
            norm_x = loc[0] * 105 / 120
            norm_y = loc[1] * 68 / 80
            features.extend([
                norm_x, 
                norm_y, 
                float(p.get('actor', False)), 
                float(p.get('teammate', False)), 
                float(p.get('keeper', False))
            ])
        else:
            # Pad with zeros for missing players
            features.extend([0.0] * 5)
    return features

def get_manual_scaled_weights(df_featured):
    """
    Berechnet die Klassengewichte und skaliert sie manuell.
    Diese Version ist robust und passt sich an die tatsächlichen Daten an.
    """
    # Hole die numerisch kodierten Action-Labels und die dazugehörigen Namen
    y_act_codes = df_featured['act'].astype('category').cat.codes
    action_categories = df_featured['act'].astype('category').cat.categories
    
    print(f"Gefundene Action-Klassen in der richtigen Reihenfolge: {list(action_categories)}")

    # 1. Berechne die "balanced" Gewichte als Basis
    balanced_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_act_codes),
        y=y_act_codes
    )
    print(f"Automatisch 'balanced' Gewichte: {balanced_weights}")

    # 2. Definiere die manuellen Skalierungsfaktoren in einem Dictionary
    scaler_map = {
        'Pass': 1.0,
        'Dribble': 1.0,
        'Carry': 1.0, # Fügen Sie alle möglichen Namen Ihrer Klassen hinzu
        'Shot': 1.0,
        'Cross': 1.0,
        'Possession End': 1.5  # Dies ist der Wert aus dem Paper-Code
    }
    # Fügen Sie einen Default-Wert hinzu, falls eine unerwartete Klasse auftaucht
    default_scaler = 1.0

    # 3. Baue den 'scalers'-Array dynamisch auf, basierend auf den gefundenen Klassen
    scalers = [scaler_map.get(cat, default_scaler) for cat in action_categories]
    scalers = np.array(scalers)
    
    print(f"Manuelle Skalierer in der richtigen Reihenfolge: {scalers}")
    
    # 4. Wende die Skalierung an (die Shapes werden jetzt immer übereinstimmen)
    final_weights = balanced_weights * scalers
    
    print(f"Manuell skalierte Gewichte nach Anpassung: {final_weights}")
    
    return torch.tensor(final_weights, dtype=torch.float32)

# --- ANGEPASSTE FUNKTION: Erstellt nur noch die DataLoaders ---
def make_dataloaders(X, y, batch_size):
    """
    Erstellt Trainings- und Validierungs-DataLoader-Objekte aus den Sequenzdaten.
    Die Berechnung der Klassengewichte wurde entfernt und ausgelagert.
    """
    y_act = y['act']
    
    # Die Berechnung von class_weights und action_weights_tensor wurde entfernt.

    # Erstelle ein PyTorch TensorDataset
    dataset = TensorDataset(
        torch.tensor(X['cat'], dtype=torch.long),
        torch.tensor(X['cont'], dtype=torch.float32),
        torch.tensor(X['360'], dtype=torch.float32),
        torch.tensor(y['zone'], dtype=torch.long),
        torch.tensor(y_act, dtype=torch.long),
        torch.tensor(y['time'], dtype=torch.float32)
    )
    
    # Teile die Indizes für Trainings- und Validierungssets auf
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)), 
        test_size=0.2, 
        random_state=42, 
        shuffle=True,
        stratify=y_act # Hinzugefügt für eine bessere Verteilung der Klassen
    )
    
    # Erstelle die DataLoaders
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices), 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=0
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # Gibt nur noch die Loader zurück
    return train_loader, val_loader
