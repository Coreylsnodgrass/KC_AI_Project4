# model.py

import torch
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=2,
        num_layers=2,
        dim_feedforward=256,
        max_len=512,
        dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, hidden=None):
        # x: [batch, seq_len]
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        out = self.transformer(x)              # [batch, seq_len, d_model]
        logits = self.fc(out)                  # [batch, seq_len, vocab_size]
        return logits, None
