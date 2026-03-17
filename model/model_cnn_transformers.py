import torch
import torch.nn as nn
from torch import Tensor
from utils.cnn_transformers_create_training_examples import *

class ConvTransformerHybrid(nn.Module):
    """
    A hybrid model combining convolutional layers and transformer layers for sequence processing.
    Input:
        - tokens: shape (B, 40) - 
        - maskpos: shape (B) long
        - pad_mask: shape (B, 40) bool (True for padding positions)
    Output:
        - logits: shape (B, 26) - class scores for letters a-z
    """
    def __init__(
            self, 
            *,
            vocab_size: int = 29,
            d_model: int = 256, # Dimension of the embedding
            max_len: int = 40,
            cnn_layers: int = 4,
            dilation_base: int =2,
            n_heads: int = 8,
            num_encoder_layers: int = 4,
            ff_dim: int = 512,
            dropout: float = 0.1
            ):
        super().__init__()
        # Embdedings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_len, d_model)

        # Dilated Convolutional layers
        conv = []
        dilation = 1
        for _ in range(cnn_layers):
            conv.append(
                nn.Conv1d(
                    in_channels=d_model, 
                    out_channels=d_model, 
                    kernel_size=3, 
                    padding=dilation, 
                    dilation=dilation
                )
            )
            conv.append(nn.GELU())
            conv.append(nn.Dropout(dropout))
            dilation *= dilation_base
        self.cnn = nn.Sequential(*conv)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer, 
            num_layers=num_encoder_layers
        )
        
        # Classsification head
        self.classifier = nn.Linear(d_model, 26)
    

    def forward(self, tokens, maskpos, pad_mask):
        B, L = tokens.shape
        device = tokens.device

        # Embedding
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(tokens) + self.positional_embedding(pos_ids)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # (B, C, L) -> (B, L, C)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        gather_idx = maskpos.view(B, 1, 1).expand(-1, -1, x.size(2))  # (B, 1, C)
        hidden = x.gather(1, gather_idx).squeeze(1)

        return self.classifier(hidden)


def build_model(**kwargs) -> ConvTransformerHybrid:
    return ConvTransformerHybrid(**kwargs)


# if __name__ == "__main__":
#     # Test the model with dummy data
#     model = ConvTransformerHybrid()
#     tokens = torch.zeros(2, 40, dtype=torch.long)
#     maskpos = torch.zeros(2, dtype=torch.long)
#     pad_mask = tokens.eq(PAD_ID)
#     out = model(tokens, maskpos, pad_mask)
#     assert out.shape == (2, 26), "Output shape mismatch"
#     print('Model output:', out)
#     print('Model output shape:', out.shape)    




