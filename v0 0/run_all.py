# run_all.py
import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from v0.config import *
from v0.feature_engineering import add_features
from v0.sequence_builder import build_sequences 
from v0.model_nmstpp import NMSTPP_Sequential

# --- Die Funktionen load_and_preprocess_data und make_dataloaders bleiben unverändert ---

def load_and_preprocess_data(file_path, use_fast_debug, max_rows, seq_len):
    """
    Lädt die Daten, führt das Feature Engineering und die Sequenzerstellung durch
    und gibt danach den Speicher des ursprünglichen DataFrames frei.
    """
    print("--- Schritt 1: Daten laden und vorverarbeiten ---")
    df = pd.read_csv(file_path)
    if use_fast_debug and len(df) > max_rows:
        print(f"FAST_DEBUG Modus: Begrenze auf {max_rows} Zeilen.")
        df = df.iloc[:max_rows].copy()

    print("Führe Feature Engineering durch (dies kann bei großen Datenmengen dauern)...")
    df = add_features(df)

    print("Erstelle Sequenzen für das Modell...")
    X, y, le_action_dict = build_sequences(df, seq_len=seq_len)

    print(f"Großen DataFrame (Größe: ~{df.memory_usage(deep=True).sum() / 1e9:.2f} GB) aus dem Speicher entfernen...")
    del df
    gc.collect()
    print("Speicher freigegeben.")

    return X, y, le_action_dict

def make_dataloaders(X, y, le_action, batch=BATCH_SIZE):
    print("\n--- Schritt 2: DataLoaders erstellen ---")
    y_act = y['act']
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_act), y=y_act)
    action_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Klassen: {le_action}")
    print(f"Verwendete Klassengewichte: {action_weights_tensor.numpy()}")

    Xc = torch.tensor(X['cat'], dtype=torch.long)
    Xcont = torch.tensor(X['cont'], dtype=torch.float32)
    X360 = torch.tensor(X['360'], dtype=torch.float32)
    yz = torch.tensor(y['zone'], dtype=torch.long)
    ya = torch.tensor(y_act, dtype=torch.long)
    yt = torch.tensor(y['time'], dtype=torch.float32)

    ds = TensorDataset(Xc, Xcont, X360, yz, ya, yt)
    
    tr_idx, va_idx = train_test_split(np.arange(len(ds)), test_size=0.2, random_state=42, shuffle=True)
    
    train_loader = DataLoader(torch.utils.data.Subset(ds, tr_idx), batch_size=batch, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(torch.utils.data.Subset(ds, va_idx), batch_size=batch, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, action_weights_tensor


def train(model, train_loader, val_loader, action_weights, epochs=EPOCHS, lr=LR):
    print("\n--- Schritt 3: Modell trainieren ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training auf Gerät: {device}")
    model = model.to(device)
    action_weights = action_weights.to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    # KORRIGIERT: Das Argument 'verbose=True' wurde aus der folgenden Zeile entfernt.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    best_loss, patience_counter = float('inf'), PATIENCE
    for ep in range(1, epochs+1):
        model.train()
        train_loss = 0
        for xc, xcont, x360, yz, ya, yt in train_loader:
            xc, xcont, x360 = xc.to(device), xcont.to(device), x360.to(device)
            yz, ya, yt = yz.to(device), ya.to(device), yt.to(device)
            
            opt.zero_grad()
            targets = {'time': yt, 'zone': yz, 'action': ya}
            out = model(xc, xcont, x360, targets=targets, teacher_forcing=True)
            loss, _ = model.loss(out, (yz, ya, yt), action_weights=action_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for xc, xcont, x360, yz, ya, yt in val_loader:
                xc, xcont, x360 = xc.to(device), xcont.to(device), x360.to(device)
                yz, ya, yt = yz.to(device), ya.to(device), yt.to(device)
                out = model(xc, xcont, x360, targets=None, teacher_forcing=False)
                loss, _ = model.loss(out, (yz, ya, yt), action_weights=action_weights)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # Manuelle Ausgabe, wenn die Lernrate angepasst wird
        current_lr = opt.param_groups[0]['lr']
        print(f'Epoch {ep}/{epochs} - train_loss {train_loss:.4f} - val_loss {val_loss:.4f} - LR: {current_lr}')
        
        sched.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = PATIENCE
            torch.save(model.state_dict(), 'best_nmstpp.pt')
        else:
            patience_counter -= 1
            if patience_counter == 0: 
                print("Early stopping.")
                break
            
    model.load_state_dict(torch.load('best_nmstpp.pt', map_location=device))
    return model

# --- Die Funktion evaluate bleibt unverändert ---

def evaluate(model, loader, le_action_dict):
    print("\n--- Schritt 4: Modell evaluieren ---")
    device = next(model.parameters()).device
    model.eval()
    all_z, all_z_hat, all_a, all_a_hat = [], [], [], []
    with torch.no_grad():
        for xc, xcont, x360, yz, ya, yt in loader:
            xc, xcont, x360 = xc.to(device), xcont.to(device), x360.to(device)
            
            zlog, alog, tpred = model(xc, xcont, x360, targets=None, teacher_forcing=False)
            
            all_z.extend(yz.numpy())
            all_a.extend(ya.numpy())
            all_z_hat.extend(zlog.argmax(dim=-1).cpu().numpy())
            all_a_hat.extend(alog.argmax(dim=-1).cpu().numpy())

    print('\n=== Zone Prediction Report ===')
    zone_labels = sorted(list(np.unique(all_z)))
    print(classification_report(all_z, all_z_hat, labels=zone_labels, digits=2, zero_division=0))

    print('\n=== Action Prediction Report (mit Klassengewichtung) ===')
    target_names = [le_action_dict[i] for i in sorted(le_action_dict.keys())]
    print(classification_report(all_a, all_a_hat, target_names=target_names, zero_division=0, digits=2))

def main():
    X, y, le_action_dict = load_and_preprocess_data(
        file_path=COMBINED_FILE,
        use_fast_debug=FAST_DEBUG,
        max_rows=FAST_MAX_ROWS,
        seq_len=SEQ_LEN
    )
    
    train_loader, val_loader, action_weights = make_dataloaders(X, y, le_action_dict)
    
    print("\n--- Modell wird initialisiert ---")
    model = NMSTPP_Sequential(
        cont_dim=X['cont'].shape[2],
        dim_360=X['360'].shape[2],
        seq_len=SEQ_LEN,
        num_zones=20,
        num_actions=len(le_action_dict),
        d_model=128, nhead=4, num_layers=2
    )

    model = train(model, train_loader, val_loader, action_weights=action_weights)
    
    evaluate(model, val_loader, le_action_dict)
    
    print("\n--- Skript erfolgreich beendet ---")

if __name__ == '__main__':
    main()