import torch.nn as nn

from models.pse import SpatialEncoder
from models.tae import TemporalAttentionEncoder
from models.mlps import MLP4


class PSE_TAE(nn.Module):

    def __init__(self, device):
        super(PSE_TAE, self).__init__()
        self.spatial_encoder = SpatialEncoder(device)
        self.temporal_attention_encoder = TemporalAttentionEncoder(4, 128)
        self.decoder = MLP4()

    def forward(self, x, geom, pixels_in_parcel, mask):
        encoding = self.spatial_encoder(x, geom, pixels_in_parcel, mask)
        output_tae = self.temporal_attention_encoder(encoding)
        output = self.decoder(output_tae)
        return output
