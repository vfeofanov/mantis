import pytest
import torch

from mantis.architecture import MantisV1, MantisV2


@pytest.mark.parametrize("Model", [MantisV1, MantisV2])
@pytest.mark.parametrize("output_token", ["cls_token", "mean_token", "combined"])
@pytest.mark.parametrize("return_layer", [-1, 0, 2])
def test_intermediate_layer_and_output_token_shapes(Model, output_token, return_layer):
    # Use CPU for tests
    device = 'cpu'

    # instantiate model with small batch and defaults
    model = Model(device=device, output_token=output_token, return_transf_layer=return_layer)
    model.eval()

    # random single-channel time series with seq_len compatible with defaults (512)
    x = torch.randn(2, 1, 512, device=device)

    with torch.no_grad():
        out = model(x)

    # expected feature dimension: combined -> 2*hidden_dim, else hidden_dim
    hidden_dim = 256
    expected_dim = 2 * hidden_dim if output_token == "combined" else hidden_dim

    assert out.shape == (2, expected_dim), (
        f"Unexpected output shape for {Model.__name__} with output_token={output_token} "
        f"and return_layer={return_layer}: got {out.shape}, expected (2, {expected_dim})"
    )

    # If return_layer is not -1, ensure output differs from the final-layer output
    if return_layer != -1:
        model_final = Model(device=device, output_token=output_token, return_transf_layer=-1)
        model_final.eval()
        with torch.no_grad():
            out_final = model_final(x)
        # It's extremely unlikely that outputs from different layers are identical.
        assert not torch.allclose(out, out_final), (
            f"Outputs for return_layer={return_layer} and return_layer=-1 are identical for {Model.__name__} "
            f"(output_token={output_token}). This likely indicates return_layer is ignored."
        )


def test_combined_token_concatenation_consistency():
    """Verify that 'combined' output equals concatenation of cls and mean tokens."""
    device = 'cpu'
    model = MantisV1(device=device, output_token='combined', return_transf_layer=-1)
    model.eval()
    x = torch.randn(3, 1, 512, device=device)

    # get combined output
    with torch.no_grad():
        combined = model(x)

    # get cls and mean separately from TransformerUnit by calling transf_unit directly
    with torch.no_grad():
        cls = model.transf_unit(x_embeddings := model.tokgen_unit(x), output_token='cls_token')
        mean = model.transf_unit(x_embeddings, output_token='mean_token')

    recon_cat = torch.cat([cls, mean], dim=1)
    assert combined.shape == recon_cat.shape
    assert torch.allclose(combined, recon_cat), "Combined output does not match concatenated cls+mean tokens"
