import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.positional_encoding import PositionalEncoding
from models.mlps import MLP3


class TemporalAttentionEncoder(nn.Module):
    """
    TAE without hints from the original code
    """

    def __init__(self, heads, d_e):
        super(TemporalAttentionEncoder, self).__init__()

        self.d_e = d_e
        self.d_k = d_e // heads
        self.h = heads

        self.pos_encoding = PositionalEncoding()
        self.fc1_q_h1 = nn.Linear(128, 32)
        self.fc1_q_h2 = nn.Linear(128, 32)
        self.fc1_q_h3 = nn.Linear(128, 32)
        self.fc1_q_h4 = nn.Linear(128, 32)

        self.fc1_k_h1 = nn.Linear(128, 32)  # Does k get the same weights as q?
        self.fc1_k_h2 = nn.Linear(128, 32)
        self.fc1_k_h3 = nn.Linear(128, 32)
        self.fc1_k_h4 = nn.Linear(128, 32)

        self.fc2_h1 = nn.Linear(32, 32)
        self.fc2_h2 = nn.Linear(32, 32)
        self.fc2_h3 = nn.Linear(32, 32)
        self.fc2_h4 = nn.Linear(32, 32)

        self.mlp3 = MLP3()

    def forward(self, x):  # [batch_size x seq_len x hidden_state:128]

        batch_size, seq_len, hidden_state = x.size()
        e_p = self.pos_encoding(x)  # [batch_size x seq_len x hidden_state:128]

        # Queries
        q_h1 = self.fc1_q_h1(e_p)  # [batch_size x seq_len x hidden_state:32]
        q_h2 = self.fc1_q_h2(e_p)  # [batch_size x seq_len x hidden_state:32]
        q_h3 = self.fc1_q_h3(e_p)  # [batch_size x seq_len x hidden_state:32]
        q_h4 = self.fc1_q_h4(e_p)  # [batch_size x seq_len x hidden_state:32]

        q_mean_h1 = torch.mean(q_h1, 1)  # [batch_size x hidden_state:32]
        q_mean_h2 = torch.mean(q_h2, 1)  # [batch_size x hidden_state:32]
        q_mean_h3 = torch.mean(q_h3, 1)  # [batch_size x hidden_state:32]
        q_mean_h4 = torch.mean(q_h4, 1)  # [batch_size x hidden_state:32]

        q_hat_h1 = self.fc2_h1(q_mean_h1)  # [batch_size x hidden_state:32]
        q_hat_h2 = self.fc2_h2(q_mean_h2)  # [batch_size x hidden_state:32]
        q_hat_h3 = self.fc2_h3(q_mean_h3)  # [batch_size x hidden_state:32]
        q_hat_h4 = self.fc2_h4(q_mean_h4)  # [batch_size x hidden_state:32]

        # Keys
        k_h1 = self.fc1_k_h1(e_p)  # [batch_size x seq_len x hidden_state:32]
        k_h2 = self.fc1_k_h1(e_p)  # [batch_size x seq_len x hidden_state:32]
        k_h3 = self.fc1_k_h1(e_p)  # [batch_size x seq_len x hidden_state:32]
        k_h4 = self.fc1_k_h1(e_p)  # [batch_size x seq_len x hidden_state:32]

        # Attention
        attention_scores_h1 = q_hat_h1.matmul(k_h1.transpose(-2, -1)) / math.sqrt(
            hidden_state)  # [batch_size x seq_len]
        attention_scores_h2 = q_hat_h2.matmul(k_h2.transpose(-2, -1)) / math.sqrt(
            hidden_state)  # [batch_size x seq_len]
        attention_scores_h3 = q_hat_h3.matmul(k_h3.transpose(-2, -1)) / math.sqrt(
            hidden_state)  # [batch_sizex seq_len]
        attention_scores_h4 = q_hat_h4.matmul(k_h4.transpose(-2, -1)) / math.sqrt(
            hidden_state)  # [batch_size x seq_len]

        attention_probs_h1 = F.softmax(attention_scores_h1, dim=-1)  # [batch_size x seq_len]
        attention_probs_h2 = F.softmax(attention_scores_h2, dim=-1)  # [batch_size x seq_len]
        attention_probs_h3 = F.softmax(attention_scores_h3, dim=-1)  # [batch_size x seq_len]
        attention_probs_h4 = F.softmax(attention_scores_h4, dim=-1)  # [batch_size x seq_len]

        attention_output_h1 = torch.matmul(attention_probs_h1, e_p)  # [batch_size x hidden_state:128]
        attention_output_h2 = torch.matmul(attention_probs_h2, e_p)  # [batch_size x hidden_state:128]
        attention_output_h3 = torch.matmul(attention_probs_h3, e_p)  # [batch_size x hidden_state:128]
        attention_output_h4 = torch.matmul(attention_probs_h4, e_p)  # [batch_size x hidden_state:128]

        attention_output = torch.cat((attention_output_h1,
                                      attention_output_h2,
                                      attention_output_h3,
                                      attention_output_h4),
                                     1)  # [batch_size x hidden_state:512]

        o_hat = self.mlp3(attention_output)  # [batch_size x hidden_state:128]

        return o_hat
