# model_nmstpp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NMSTPP_Sequential(nn.Module):
    # MODIFIZIERT: Neue Parameter für Vokabulargrößen und Embedding-Dimensionen
    def __init__(self, cont_dim, dim_360, seq_len, 
                 num_zones, num_actions, zone_emb_dim=16, action_emb_dim=16,
                 d_model=128, nhead=4, num_layers=2, hidden_dim=128, use_360=True):
        super().__init__()
        self.use_360 = use_360

        # NEU: Embedding-Schichten für kategoriale Input-Features
        self.zone_embedding = nn.Embedding(num_zones, zone_emb_dim)
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim)

        # Lineare Schichten für kontinuierliche Features
        self.fc_cont = nn.Linear(cont_dim, d_model // 2) # Wir teilen die Dimension auf
        if use_360:
            self.fc_360 = nn.Linear(dim_360, d_model // 2)

        # Eine Schicht, um die konkatenierten Embeddings und kontinuierlichen Features auf d_model zu bringen
        total_input_dim = zone_emb_dim + action_emb_dim + (d_model // 2)
        if use_360:
            total_input_dim += (d_model // 2)
        self.input_projection = nn.Linear(total_input_dim, d_model)

        self.pos_emb = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Output Heads (bleiben strukturell gleich) ---
        self.time_net = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.zone_net = nn.Sequential(nn.Linear(d_model + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_zones))
        
        # Embedding für die *vorhergesagte* Zone als Input für den Action-Head
        self.target_zone_emb = nn.Embedding(num_zones, zone_emb_dim)
        self.action_net = nn.Sequential(nn.Linear(d_model + 1 + zone_emb_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_actions))

    # MODIFIZIERT: Die Forward-Methode akzeptiert jetzt getrennte Tensoren
    def forward(self, x_cat, x_cont, x_360=None, targets=None, teacher_forcing=True):
        # 1. Embeddings für kategoriale Inputs erstellen
        # x_cat hat die Form [Batch, SeqLen, NumCatFeatures (2)]
        zone_emb = self.zone_embedding(x_cat[:, :, 0])    # -> [B, L, zone_emb_dim]
        action_emb = self.action_embedding(x_cat[:, :, 1]) # -> [B, L, action_emb_dim]

        # 2. Kontinuierliche Inputs durch lineare Schichten verarbeiten
        cont_proj = F.relu(self.fc_cont(x_cont)) # -> [B, L, d_model/2]

        # 3. Alle Inputs konkatenieren
        x = torch.cat([zone_emb, action_emb, cont_proj], dim=-1)
        if self.use_360 and x_360 is not None:
            x360_proj = F.relu(self.fc_360(x_360)) # -> [B, L, d_model/2]
            x = torch.cat([x, x360_proj], dim=-1)

        # 4. Auf d_model projizieren und Transformer anwenden
        x = self.input_projection(x)
        B, L, D = x.shape
        pos = self.pos_emb.unsqueeze(0).expand(B, -1, -1)
        x = x + pos
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[-1]

        # --- Output Heads (Logik bleibt gleich) ---
        time_pred = self.time_net(x).squeeze(-1)
        time_for_zone = targets['time'] if teacher_forcing and targets is not None and 'time' in targets else time_pred
        
        zone_input = torch.cat([x, time_for_zone.unsqueeze(-1)], dim=-1)
        zone_logits = self.zone_net(zone_input)
        
        zone_for_action_idx = targets['zone'] if teacher_forcing and targets is not None and 'zone' in targets else zone_logits.argmax(dim=-1)
        zone_emb_for_action = self.target_zone_emb(zone_for_action_idx)
        
        action_input = torch.cat([x, time_for_zone.unsqueeze(-1), zone_emb_for_action], dim=-1)
        action_logits = self.action_net(action_input)

        return zone_logits, action_logits, time_pred

    def loss(self, preds, targets, action_weights=None):
        zone_logits, action_logits, time_pred = preds
        y_zone, y_action, y_time = targets
        l_zone = F.cross_entropy(zone_logits, y_zone)
        l_action = F.cross_entropy(action_logits, y_action, weight=action_weights)
        mse = F.mse_loss(time_pred, y_time)
        l_time = torch.sqrt(mse + 1e-8)
        total_loss = 10.0 * l_time + l_zone + l_action
        return total_loss, (l_zone.item(), l_action.item(), l_time.item())