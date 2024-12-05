import torch
import torch.nn as nn

class AudioTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(AudioTransformer, self).__init__()
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected layer to reduce to a single vector
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output a scalar for binary classification
            nn.Sigmoid() # casts between 0 and 1
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = x.permute(1, 0, 2)  # Shape: (sequence_length, batch_size, input_dim)
        x = self.transformer_encoder(x)  # Shape: (sequence_length, batch_size, input_dim)
        
        # Mean pooling to aggregate sequence dimension
        x = x.mean(dim=0)  # Shape: (batch_size, input_dim)
        
        # Pass through fully connected layer
        x = self.fc(x).squeeze(-1)  # Shape: (batch_size,)
        return x