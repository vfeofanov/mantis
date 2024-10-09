import torch

from torch import nn


class FineTuningNetwork(nn.Module):
    def __init__(self, encoder, head, adapter=None):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.adapter = adapter

    def forward(self, x):
        if self.encoder is None:
            return self.head(x)
        elif self.adapter is None:
            if x.shape[1] > 1:
                Warning("The data is multi-variate! Applying encoder to all channels independently")
                return self.head(torch.cat([self.encoder(x[:, [i], :]) for i in range(x.shape[1])], dim=-1))
            else:
                return self.head(self.encoder(x))
        else:
            adapter_output = self.adapter(x)
            return self.head(torch.cat([self.encoder(adapter_output[:, [i], :]) for i in range(adapter_output.shape[1])], dim=-1))
