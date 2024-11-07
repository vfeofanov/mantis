import torch

from torch import nn


class LinearChannelCombiner(nn.Module):
    """
    A differentiable adapter that implements a linear projector along the channel axis.
    Given time series dataset with `num_channels`, it transforms it into a dataset with `new_num_channels`,
    where each new channel is a linear combination of the original ones.
    This adapter is a pytorch module, which can be trained together with the prediction head 
    or during the full fine-tuning of Mantis.

    Parameters
    ----------
    num_channels: int
        The original number of channels in a time series dataset.
    new_num_channels: int
        The number of channels after transformation.
    """
    def __init__(self, num_channels, new_num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.new_num_channels = new_num_channels
        self.reduction_matrix = nn.Parameter(torch.rand(1, new_num_channels, num_channels))

    def forward(self, x):
        return self.reduction_matrix @ x
