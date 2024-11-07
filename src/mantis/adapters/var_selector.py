import numpy as np

from sklearn.feature_selection import VarianceThreshold


class VarianceBasedSelector:
    """
    Perform a filter feature selection approach to reduce the number of channels in a multichannel time series data set.
    More specifically, it selects those channels that have the highest variance.

    Parameters
    ----------
    new_num_channels: int 
        The number of selected channels.
    """
    def __init__(self, new_num_channels):
        self.new_num_channels = new_num_channels
        self.support_ = None
        
    def fit(self, X):
        # Flatten the tensor to 2D for sklearn compatibility
        X_transposed = np.swapaxes(X, 1, 2)

        num_samples, seq_len, num_channels = X_transposed.shape

        
        X_2d = X_transposed.reshape(num_samples * seq_len, num_channels)

        # Calculate variances and select top features
        variances = np.var(X_2d, axis=0)
        # Select top features by variance
        self.support_ = np.argsort(variances)[::-1][:self.new_num_channels]
        return self.support_

    def transform(self, X):
        if self.support_ is None:
            raise RuntimeError("You must call fit at least once before calling transform.")

        # Select features based on precomputed indices
        selected_features = X[:, self.support_, :]

        return selected_features
