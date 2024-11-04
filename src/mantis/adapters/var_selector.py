import numpy as np

from sklearn.feature_selection import VarianceThreshold


class VarianceBasedSelector:
    """
    Perform filter feature selection approach to the multichannel time series data
    Class to reduce the dimensionality of input channels by selecting features with the highest variance.

    Parameters
    ----------
    new_num_channels: int 
        Number of channels in the latent tensor.
    """

    def __init__(self, new_num_channels):
        self.new_num_channels = new_num_channels
        self.support_ = None
        
    
    def fit(self, X):
        """
        Fit the feature selector based on the variance of the input data.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, sequence_length).
        """
        # Flatten the tensor to 2D for sklearn compatibility
        X_transposed = np.swapaxes(X, 1, 2)

        num_samples, seq_len, num_channels = X_transposed.shape

        
        X_2d = X_transposed.reshape(num_samples * seq_len, num_channels)

        # Calculate variances and select top features
        variances = np.var(X_2d, axis=0)
        # Select top features by variance
        self.support_ = np.argsort(variances)[::-1][:self.new_num_channels]
        return self.support_

    def transform(self, x):
        """
        Transform the input data by selecting features based on precomputed variances.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, sequence_length).

        Returns:
            torch.Tensor: Reduced tensor with selected features.
        """
        if self.support_ is None:
            raise RuntimeError("You must call fit at least once before calling transform.")

        # Select features based on precomputed indices
        selected_features = x[:, self.support_, :]

        return selected_features
