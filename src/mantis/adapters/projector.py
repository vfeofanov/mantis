import numpy as np

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection


class MultichannelProjector:
    def __init__(self, new_num_channels, patch_window_size=None, base_projector=None):
        # init dimensions
        self.new_num_channels = new_num_channels
        self.patch_window_size = patch_window_size
        self.patch_window_size_ = 1 if patch_window_size is None else patch_window_size
        # init base projector 
        self.base_projector = base_projector
        n_components = patch_window_size * new_num_channels
        if base_projector in [None, 'pca']:
            self.base_projector_ = PCA(n_components=n_components)
        elif base_projector == 'svd':
            self.base_projector_ = TruncatedSVD(n_components=n_components)
        elif base_projector == 'rand':
            self.base_projector_ = SparseRandomProjection(n_components=n_components)
        # you can give your own base_projector with fit() and transform() methods, and it should have the argument `n_components`.
        else:
            self.base_projector_ = base_projector(n_components=n_components)

    def fit(self, X):
        X_transposed = np.swapaxes(X, 1, 2)
        
        num_samples, seq_len, num_channels = X_transposed.shape
        num_patches = seq_len // self.patch_window_size_
        assert num_patches * self.patch_window_size_ == seq_len

        X_2d = X_transposed.reshape(num_samples * num_patches, self.patch_window_size_ * num_channels)
        return self.base_projector_.fit(X_2d)

    def transform(self, X):
        """
        Apply the PCA transform on the input data.
        """
        # X_transposed = X.transpose([1, 2])
        X_transposed = np.swapaxes(X, 1, 2)

        num_samples, seq_len, num_channels = X_transposed.shape
        num_patches = seq_len // self.patch_window_size_
        assert num_patches * self.patch_window_size_ == seq_len

        X_2d = X_transposed.reshape(num_samples * num_patches, self.patch_window_size_ * num_channels)

        X_transformed = self.base_projector_.transform(X_2d)
        X_transformed = X_transformed.reshape([num_samples, seq_len, self.new_num_channels])
        return np.swapaxes(X_transformed, 1, 2)
