import numpy as np
import pytest
from torch import nn

from mantis.trainer import MantisTrainer


class DummyEncoder(nn.Module):
    hidden_dim = 4

    def forward(self, x):
        return x.mean(dim=-1).expand(-1, self.hidden_dim)


def test_default_fine_tuning_head_uses_batch_normalization():
    x = np.random.randn(4, 1, 8).astype(np.float32)
    y = np.array([0, 1, 0, 1])
    trainer = MantisTrainer(device="cpu", network=DummyEncoder())

    model = trainer.fit(
        x,
        y,
        fine_tuning_type="head",
        num_epochs=0,
        learning_rate_adjusting=False,
    )

    assert isinstance(model.head[0], nn.BatchNorm1d)
    assert model.head[0].num_features == DummyEncoder.hidden_dim
    assert isinstance(model.head[1], nn.Linear)
    assert model.head[1].out_features == 2


@pytest.mark.parametrize("fine_tuning_type", ['head', 'full'])
def test_trailing_single_sample_batch_is_dropped(fine_tuning_type):
    """The default batch-normalized head cannot handle a batch of one sample."""
    x = np.random.randn(5, 1, 8).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0])
    trainer = MantisTrainer(device="cpu", network=DummyEncoder())

    trainer.fit(
        x,
        y,
        fine_tuning_type=fine_tuning_type,
        num_epochs=1,
        batch_size=4,
        learning_rate_adjusting=False,
    )
