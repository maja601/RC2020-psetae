import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.positional_encoding import PositionalEncoding
from models.mlps import MLP3


class TemporalAttentionEncoder(nn.Module):
    """
    TAE with hints from the original code
    """

    def __init__(self, heads, d_e):
        super(TemporalAttentionEncoder, self).__init__()

        self.d_e = d_e
        self.d_k = d_e // heads
        self.h = heads

        self.pos_encoding = PositionalEncoding()
        self.fc1_q = nn.Linear(128, 128)
        self.fc1_k = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mlp3 = MLP3()

    def forward(self, x):  # [batch_size x seq_len x hidden_state:128]

        batch_size, seq_len, hidden_state = x.size()
        e_p = self.pos_encoding(x)  # [batch_size x seq_len x hidden_state:128]

        # Queries
        q = self.fc1_q(e_p)         # [batch_size x seq_len x hidden_state:128]
        q_mean = torch.mean(q, 1)   # [batch_size x hidden_state:128]
        q_hat = self.fc2(q_mean).view(batch_size, self.h, self.d_k)     # [batch_size x num_heads x d_k]
        q_hat = q_hat.contiguous().view(-1, self.d_k).unsqueeze(1)      # [batch_size * num_heads x 1 x d_k]

        # Keys
        k = self.fc1_k(e_p)                                                 # [batch_size x seq_len x hidden_state:128]
        k = k.view(batch_size, seq_len, self.h, self.d_k)                   # [batch_size x seq_len x num_heads x d_k]
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, seq_len, self.d_k)  # [batch_size * num_heads x seq_len x d_k]

        # Values (like original garnot code)
        v = e_p.repeat(self.h, 1, 1)    # [batch_size * num_heads x seq_len x hidden:128]

        # Attention
        attention_scores = q_hat.matmul(k.transpose(-2, -1)) / math.sqrt(self.d_k)      # [batch_size * num_heads x 1 x seq_len]
        attention_probs = F.softmax(attention_scores, dim=-1)               # [batch_size * num_heads x 1 x seq_len]
        attention_output = torch.matmul(attention_probs, v).squeeze()       # [batch_size * num_heads x hidden_state:128]
        attention_output = attention_output.contiguous().view(batch_size, self.h, -1)   # [batch_size x num_h x hidden_state:128]
        attention_output = attention_output.contiguous().view(batch_size, -1)           # [batch_size x hidden:512]

        # Output
        o_hat = self.mlp3(attention_output)                                 # [batch_size x hidden_state:128]

        return o_hat
