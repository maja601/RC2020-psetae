import os
import json
import math
import torch
import torch.nn as nn
import datetime


class PositionalEncoding(nn.Module):

    def __init__(self, d_e=128, max_len=24):
        super(PositionalEncoding, self).__init__()

        dates_file = '/home/maja/ssd/rc2020dataset/pixelset/META/dates.json'
        with open(dates_file) as f:
            self.dates_json = json.load(f)

        # Instead of taking the position, the numbers of days since the first observation is used
        days = torch.zeros(max_len)
        date_0 = self.dates_json["0"]
        date_0 = datetime.datetime.strptime(str(date_0), "%Y%m%d")
        days[0] = 0
        for i in range(max_len - 1):
            date = self.dates_json[str(i + 1)]
            date = datetime.datetime.strptime(str(date), "%Y%m%d")
            days[i + 1] = (date - date_0).days
        days = days.unsqueeze(1)

        # Calculate the positional encoding p
        p = torch.zeros(max_len, d_e)
        div_term = torch.exp(torch.arange(0, d_e, 2).float() * (-math.log(1000.0) / d_e))
        p[:, 0::2] = torch.sin(days * div_term)
        p[:, 1::2] = torch.cos(days * div_term)
        p = p.unsqueeze(0)
        self.register_buffer('p', p)

    def forward(self, x):
        x = x + self.p
        return x



# pe = PositionalEncoding()
# x = torch.zeros(128, 24, 128)
# x = pe.forward(x)
# print(x)
# print("hi")
