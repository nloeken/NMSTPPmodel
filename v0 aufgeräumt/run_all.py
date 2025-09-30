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
import optuna 
from tqdm import tqdm

from config import *
from feature_engineering import add_features
from sequence_builder import build_sequences 
from model_nmstpp import NMSTPP_Sequential

def load_and_preprocess_data(file_path, use_fast_debug, max_rows, seq_len):
    print("--- Schritt 1: Daten laden und vorverarbeiten ---")
    df = pd.read_csv(file_path)
    if use_fast_debug and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).copy()
    
    # Ruft das erweiterte Feature Engineering auf
    df = add_features(df)
    
    # Ruft den Sequence Builder auf, der jetzt auch das Scaling enthält
    X, y, le_action_dict = build_sequences(df, seq_len=seq_len)
    
    del df
    gc.collect()
    return X, y, le_action_dict

def make_dataloaders(X, y, le_action, batch_size):
    y_act = y['act']
    class_weights = compute_class_weight('balanced', classes=np.unique(y_act), y=y_act)
    action_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    ds = TensorDataset(
        torch.tensor(X['cat'], dtype=torch.long),
        torch.tensor(X['cont'], dtype=torch.float32),
        torch.tensor(X['360'], dtype=torch.float32),
        torch.tensor(y['zone'], dtype=torch.long),
        torch.tensor(y_act, dtype=torch.long),
        torch.tensor(y['time'], dtype=torch.float32)
    )
    
    # Split in Trainings- und Validierungsset
    tr_idx, va_idx = train_test_split(np.arange(len(ds)), test_size=0.2, random_state=42, shuffle=True)
    
    train_loader = DataLoader(torch.utils.data.Subset(ds, tr_idx), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(torch.utils.data.Subset(ds, va_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, action_weights_tensor

def train(model, train_loader, val_loader, action_weights, lr, weight_decay):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training auf Gerät: {device}")
    model = model.to(device)
    action_weights = action_weights.to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    best_loss = float('inf')
    patience_counter = PATIENCE
    for ep in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoche {ep}/{EPOCHS} [Training]")
        for xc, xcont, x360, yz, ya, yt in pbar:
            xc, xcont, x360, yz, ya, yt = xc.to(device), xcont.to(device), x360.to(device), yz.to(device), ya.to(device), yt.to(device)
            
            opt.zero_grad()
            
            # MODIFIZIERT: Vereinfachter Modellaufruf ohne Teacher Forcing
            out = model(xc, xcont, x360)
            
            loss, _ = model.loss(out, (yz, ya, yt), action_weights=action_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        train_loss /= len(train_loader)
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for xc, xcont, x360, yz, ya, yt in tqdm(val_loader, desc=f"Epoche {ep}/{EPOCHS} [Validierung]"):
                xc, xcont, x360, yz, ya, yt = xc.to(device), xcont.to(device), x360.to(device), yz.to(device), ya.to(device), yt.to(device)
                
                # MODIFIZIERT: Vereinfachter Modellaufruf ohne Teacher Forcing
                out = model(xc, xcont, x360)
                
                loss, _ = model.loss(out, (yz, ya, yt), action_weights=action_weights)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f'Epoch {ep}/{EPOCHS} - train_loss {train_loss:.4f} - val_loss {val_loss:.4f}')
        sched.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = PATIENCE
        else:
            patience_counter -= 1
            if patience_counter == 0: 
                print("Early stopping.")
                break
    return model

def evaluate(model, loader, le_action_dict):
    device = next(model.parameters()).device
    model.eval()
    all_a, all_a_hat = [], []
    with torch.no_grad():
        for xc, xcont, x360, yz, ya, yt in loader:
            xc, xcont, x360 = xc.to(device), xcont.to(device), x360.to(device)
            
            # MODIFIZIERT: Vereinfachter Modellaufruf ohne Teacher Forcing
            zone_logits, action_logits, time_pred = model(xc, xcont, x360)
            
            all_a.extend(ya.numpy())
            all_a_hat.extend(action_logits.argmax(dim=-1).cpu().numpy())

    print("\n--- Evaluationsergebnis des Trials ---")
    target_names = [le_action_dict[i] for i in sorted(le_action_dict.keys())]
    report_dict = classification_report(all_a, all_a_hat, target_names=target_names, zero_division=0, output_dict=True)
    
    print(classification_report(all_a, all_a_hat, target_names=target_names, zero_division=0, digits=2))
    macro_f1 = report_dict['macro avg']['f1-score']
    print(f"===> Macro F1-Score für diesen Trial: {macro_f1:.4f}")
    return macro_f1

def objective(trial: optuna.Trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    d_model = trial.suggest_categorical('d_model', [128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])

    print(f"\n===== Starte Trial {trial.number} mit Parametern: =====")
    print(f"lr: {lr:.6f}, d_model: {d_model}, num_layers: {num_layers}, dropout: {dropout_rate:.2f}, weight_decay: {weight_decay:.6f}, batch_size: {batch_size}")

    X, y, le_action_dict = load_and_preprocess_data(
        file_path=COMBINED_FILE,
        use_fast_debug=FAST_DEBUG,
        max_rows=FAST_MAX_ROWS,
        seq_len=SEQ_LEN
    )
    
    train_loader, val_loader, action_weights = make_dataloaders(X, y, le_action_dict, batch_size)
    
    model = NMSTPP_Sequential(
        cont_dim=X['cont'].shape[2],
        dim_360=X['360'].shape[2],
        seq_len=SEQ_LEN,
        num_zones=20, # Annahme: 20 Zonen
        num_actions=len(le_action_dict),
        d_model=d_model,
        nhead=8 if d_model % 8 == 0 else 4,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )

    model = train(model, train_loader, val_loader, action_weights, lr=lr, weight_decay=weight_decay)
    
    macro_f1_score = evaluate(model, val_loader, le_action_dict)
    
    del model, X, y, train_loader, val_loader, action_weights
    gc.collect()
    
    return macro_f1_score


def main():
    n_trials = 50 
    
    print(f"Starte Optuna-Studie für {n_trials} Trials...")
    
    study = optuna.create_study(direction='maximize')
    
    study.optimize(objective, n_trials=n_trials)

    print("\n\n===== Optimierung abgeschlossen =====")
    print(f"Bester Macro F1-Score: {study.best_value:.4f}")
    print("Beste Hyperparameter:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()