# model_training.py
import torch
from tqdm import tqdm
from config import EPOCHS, PATIENCE, GRAD_CLIP

def train_model(model, train_loader, val_loader, action_weights):
    """
    Angepasste Version: Verwendet einen festen Adam-Optimizer mit lr=0.01 
    und einen ReduceLROnPlateau-Scheduler.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    model = model.to(device)
    action_weights = action_weights.to(device)
    
    # --- ÄNDERUNG: Optimizer und Scheduler ---
    # Fest definiert nach dem Vorbild des Paper-Codes.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    best_val_loss = float('inf')
    patience_counter = PATIENCE
    
    for epoch in range(1, EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Training]")
        for xc, xcont, x360, yz, ya, yt in pbar_train:
            xc, xcont, x360, yz, ya, yt = (t.to(device) for t in [xc, xcont, x360, yz, ya, yt])
            
            optimizer.zero_grad()
            predictions = model(xc, xcont, x360)
            loss, _ = model.loss(predictions, (yz, ya, yt), action_weights=action_weights)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            
            train_loss += loss.item()
            pbar_train.set_postfix(loss=loss.item())
        train_loss /= len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Validation]")
            for xc, xcont, x360, yz, ya, yt in pbar_val:
                xc, xcont, x360, yz, ya, yt = (t.to(device) for t in [xc, xcont, x360, yz, ya, yt])
                
                predictions = model(xc, xcont, x360)
                loss, _ = model.loss(predictions, (yz, ya, yt), action_weights=action_weights)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
        scheduler.step(val_loss)

        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = PATIENCE
        else:
            patience_counter -= 1
            if patience_counter == 0:
                print("Early stopping triggered.")
                break
                
    return model