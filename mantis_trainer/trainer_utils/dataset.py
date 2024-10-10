import torch

from torch.utils.data import Dataset


class LabeledDataset(Dataset):
    def __init__(self, x, y):
        """
        Converts from a numpy data set to a torch one
        Args:
            x (np.array): data matrix
            y (np.array): class labels
        """
        self.x = x.to(torch.float32) if isinstance(x, torch.Tensor) else torch.FloatTensor(x)
        self.y = y.to(torch.long) if isinstance(x, torch.Tensor) else torch.LongTensor(y)

    def transform(self, x):
        return x.to(torch.float32) if isinstance(x, torch.Tensor) else torch.FloatTensor(x)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        examples = self.x[idx]
        labels = self.y[idx]
        return examples, labels
