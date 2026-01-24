import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch.nn.functional as F

class SparseFeatureExtractor(nn.Module):
    """
    Feature extractor derived from MySSCAE encoder,
    corrected for anomaly detection.
    """

    def __init__(self, input_channels=2, base_channels=32, out_channels=64):
        super().__init__()

        C = base_channels
        C2 = C * 2

        self.conv1 = spconv.SubMConv2d(
            input_channels, C, 3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(C)

        self.conv2 = spconv.SubMConv2d(
            C, C2, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(C2)

        self.conv3 = spconv.SubMConv2d(
            C2, out_channels, 3, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x: spconv.SparseConvTensor):
        x = self.conv1(x)
        x = x.replace_feature(
            F.leaky_relu(self.bn1(x.features), 0.01)
        )

        x = self.conv2(x)
        x = x.replace_feature(
            F.leaky_relu(self.bn2(x.features), 0.01)
        )

        x = self.conv3(x)
        x = x.replace_feature(
            F.leaky_relu(self.bn3(x.features), 0.01)
        )

        # IMPORTANT: return per-pixel features
        return x.indices, x.features
