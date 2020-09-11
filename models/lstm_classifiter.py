from torch import nn
import torch
from  torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmModel(nn.Module):
    def __init__(self, num_embed, embed_dim, out_dim, lstm_layer_num=1, drop_rate=0.5):
        super(LstmModel, self).__init__()
        self.embed=nn.Embedding(num_embed, embed_dim)
        self.lstm=nn.LSTM(embed_dim, embed_dim, bidirectional=True, dropout=drop_rate,
            batch_first=True, num_layers=lstm_layer_num)
        self.linear=nn.Sequential(
            nn.Linear(2*embed_dim, out_dim)
        )
        self.dropout=nn.Dropout(p=drop_rate)
    
    def forward(self, x):
        # pad:1
        lens=(x!=1).sum(dim=-1)
        x=self.embed(x)
        x=self.dropout(x)
        packed_x=pack_padded_sequence(x,lens, batch_first=True)
        x,_=self.lstm(packed_x)
        padded_x,_=pad_packed_sequence(x, batch_first=True)
        x=padded_x.mean(dim=1)
        x=self.dropout(x)
        x=torch.softmax(self.linear(x), dim=-1)
        return x