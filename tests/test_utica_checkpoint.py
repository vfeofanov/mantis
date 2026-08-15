import pytest
import torch

import numpy as np

from mantis.architecture import MantisV1
from mantis.trainer import MantisTrainer


UTICA_REPO = 'fegounna/Utica'
UTICA_REVISION = '3cff4f954191b5bf9839b7a41117e2b24e7693ab'

# tensors of the UTICA pre-training objective that are not part of the encoder
TRAINING_ONLY_KEYS = {
    'transf_unit.mask_token',
    'transf_unit.norm.weight',
    'transf_unit.norm.bias',
}
# the pre-training projector is not needed to extract features
PROJECTOR_KEYS = {'prj.0.weight', 'prj.0.bias', 'prj.1.weight', 'prj.1.bias'}


@pytest.mark.parametrize("output_token,expected_dim", [('cls_token', 256), ('combined', 512)])
@pytest.mark.parametrize("device", ['cpu'])
def test_utica_from_pretrained(output_token, expected_dim, device):
    """The UTICA repository ships no config.json, so the device must come from the instance."""
    network = MantisV1(device=device, return_transf_layer=2, output_token=output_token)
    network = network.from_pretrained(UTICA_REPO, revision=UTICA_REVISION)

    assert network.device == device
    assert network.return_transf_layer == 2
    assert network.output_token == output_token
    assert network.hidden_dim == expected_dim

    model = MantisTrainer(device=device, network=network)
    x = np.random.randn(4, 1, 512).astype(np.float32)
    z = model.transform(x)

    assert z.shape == (4, expected_dim)
    assert np.isfinite(z).all()


@pytest.mark.parametrize("device", ['cpu'])
def test_utica_checkpoint_keys_are_as_expected(device):
    """Guard the non-strict load: only the known pre-training tensors may mismatch."""
    from huggingface_hub import hf_hub_download

    weights_path = hf_hub_download(
        UTICA_REPO, 'pytorch_model.bin', revision=UTICA_REVISION)
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)

    network = MantisV1(device=device)
    incompatible = network.load_state_dict(state_dict, strict=False)

    assert set(incompatible.missing_keys) == PROJECTOR_KEYS
    assert set(incompatible.unexpected_keys) == TRAINING_ONLY_KEYS
