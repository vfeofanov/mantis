from torch import nn


class Convolution(nn.Module):
    """
    Calculates the cross-correlation of a bunch of shapelets to a data set, implemented via convolution and
    performs global max-pooling.

    Parameters
    ----------
    shapelets_size : int
        the size of the shapelets / the number of time steps
    num_shapelets : int
        the number of shapelets that the block should contain
    in_channels : int
        the number of input channels of the dataset
    """

    def __init__(self,
                 kernel_size,
                 out_channels,
                 dilation,
                 ):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, dilation=1, padding=padding)

    def forward(self, x):
        """
        1) Apply 1D convolution
        2) Apply global max-pooling

        Parameters
        ----------
        x : the data set of time series
            array(float) of shape (num_samples, in_channels, len_ts)
        """
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out
