# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NMSTPP_Sequential(nn.Module):
    """
    Neural Marked Spatio-Temporal Point Process (NMSTPP) model.
    Uses a Transformer encoder to learn from sequences of football events.
    """
    def __init__(self, cont_dim, dim_360, seq_len, 
                 num_zones, num_actions, zone_emb_dim=16, action_emb_dim=16,
                 d_model=128, nhead=4, num_layers=2, hidden_dim=128, 
                 use_360=True, dropout_rate=0.4):
        super().__init__()
        self.use_360 = use_360
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Embedding layers for categorical features
        self.zone_embedding = nn.Embedding(num_zones, zone_emb_dim)
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim)

        # Projection layers for continuous and 360 features
        self.fc_cont = nn.Linear(cont_dim, d_model // 2)
        if use_360:
            self.fc_360 = nn.Linear(dim_360, d_model // 2)

        # Final projection into the model's main dimension
        total_input_dim = zone_emb_dim + action_emb_dim + (d_model // 2)
        if use_360:
            total_input_dim += (d_model // 2)
        self.input_projection = nn.Linear(total_input_dim, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction heads (autoregressive structure)
        self.time_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        self.zone_net = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim), # Input: Transformer output + time prediction
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_zones)
        )
        self.action_net = nn.Sequential(
            nn.Linear(d_model + 1 + num_zones, hidden_dim), # Input: Transformer out + time out + zone logits
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_actions)
        )

    def positional_encoding(self, src):
        """Generates fixed sinusoidal positional encodings."""
        batch_size, seq_len, d_model = src.shape
        pos_encoding = torch.zeros(seq_len, d_model, device=self.device)
        
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0).repeat(batch_size, 1, 1)

    def forward(self, x_cat, x_cont, x_360=None):
        zone_emb = self.zone_embedding(x_cat[:, :, 0])
        action_emb = self.action_embedding(x_cat[:, :, 1])
        cont_proj = F.relu(self.fc_cont(x_cont))

        # Concatenate all input features
        x = torch.cat([zone_emb, action_emb, cont_proj], dim=-1)
        if self.use_360 and x_360 is not None:
            x360_proj = F.relu(self.fc_360(x_360))
            x = torch.cat([x, x360_proj], dim=-1)

        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding(x)

        # Pass through the Transformer
        x = self.transformer(x)
        x = x[:, -1, :] # Use only the output of the last time step

        # Autoregressive Prediction Heads
        time_pred = self.time_net(x)
        zone_input = torch.cat([x, time_pred], dim=-1)
        zone_logits = self.zone_net(zone_input)
        action_input = torch.cat([x, time_pred, zone_logits], dim=-1)
        action_logits = self.action_net(action_input)

        return zone_logits, action_logits, time_pred.squeeze(-1)

    def loss(self, preds, targets, action_weights=None):
        """Calculates the combined loss for time, zone, and action predictions."""
        zone_logits, action_logits, time_pred = preds
        y_zone, y_action, y_time = targets
        
        l_zone = F.cross_entropy(zone_logits, y_zone)
        l_action = F.cross_entropy(action_logits, y_action, weight=action_weights)
        
        mse = F.mse_loss(time_pred, y_time)
        l_time = torch.sqrt(mse + 1e-8) # Root Mean Squared Error
        
        # Weighted sum of individual losses
        total_loss = 10.0 * l_time + l_zone + l_action
        
        return total_loss, (l_zone.item(), l_action.item(), l_time.item())