from models.transformer import TransformerEncoder, TransformerEncoderLayer
from torch import nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, num_embed, d_model=64, nhead=4, dim_feedforward=256, num_layers=2, out_dim=2, drop_rate=0.5):
        super(TransformerModel, self).__init__()
        self.embed=nn.Embedding(num_embed, d_model)
        self.encoder=TransformerEncoder(TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=drop_rate,
        ), num_layers=num_layers)
        self.linear=nn.Linear(d_model, out_dim)
        self.dropout=nn.Dropout(p=drop_rate)
    
    def forward(self, x):
        pad_mask=self._get_pad_mask(x)
        x=self.embed(x)
        x=self.dropout(x)
        x=x.transpose(0,1) # seq dimension first
        x, attns= self.encoder(x, src_key_padding_mask=pad_mask)
        x=x.transpose(0,1) # batch first
        x=x.mean(dim=1) # seq, embed
        x=torch.softmax(self.linear(x), dim=-1)
        return x, attns
    
    def _get_pad_mask(self, x):
        return x==1