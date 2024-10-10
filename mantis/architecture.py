import torch

from torch import nn
from einops import repeat, pack, unpack
from huggingface_hub import PyTorchModelHubMixin

from .tokgen_utils.convolution import Convolution
from .tokgen_utils.encoders import MultiScaledScalarEncoder, LinearEncoder
from .vit_utils.positional_encoding import PositionalEncoding
from .vit_utils.transformer import Transformer


# ==============================
# ==       Organization:      ==
# ==============================
# == class TokenGeneratorUnit ==
# == class ViTUnit            ==
# == class Mantis             ==
# ==============================


class TokenGeneratorUnit(nn.Module):
    def __init__(self, hidden_dim, num_patches, patch_window_size, scalar_scales, hidden_dim_scalar_enc,
                 epsilon_scalar_enc):
        super().__init__()
        self.num_patches = num_patches
        # token generator for time series objects
        num_ts_feats = 2  # original ts + its diff
        kernel_size = patch_window_size + \
            1 if patch_window_size % 2 == 0 else patch_window_size
        self.convs = nn.ModuleList([
            Convolution(kernel_size=kernel_size,
                        out_channels=hidden_dim, dilation=1)
            for i in range(num_ts_feats)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-5)
            for i in range(num_ts_feats)
        ])

        # token generator for scalar statistics
        if scalar_scales is None:
            scalar_scales = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        num_scalar_stats = 2  # mean + std
        self.scalar_encoders = nn.ModuleList([
            MultiScaledScalarEncoder(
                scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc)
            for i in range(num_scalar_stats)
        ])

        # final token projector
        self.linear_encoder = LinearEncoder(
            hidden_dim_scalar_enc * num_scalar_stats + hidden_dim * (num_ts_feats), hidden_dim)

        # scales each time-series w.r.t. its mean and std
        self.ts_scaler = lambda x: (
            x - torch.mean(x, axis=2, keepdim=True)) / (torch.std(x, axis=2, keepdim=True) + 1e-5)

    def forward(self, x):
        with torch.no_grad():
            # compute statistics for each patch
            x_patched = x.reshape(x.shape[0], self.num_patches, -1)
            mean_patched = torch.mean(x_patched, axis=-1, keepdim=True)
            std_patched = torch.std(x_patched, axis=-1, keepdim=True)
            statistics = [mean_patched, std_patched]

        # for each encoder output is (batch_size, num_sub_ts, hidden_dim_scalar_enc)
        scalar_embeddings = [self.scalar_encoders[i](
            statistics[i]) for i in range(len(statistics))]

        # apply convolution for original ts and its diff
        ts_var_embeddings = []
        # diff
        with torch.no_grad():
            diff_x = torch.diff(x, n=1, axis=2)
            # pad by zeros to have same dimensionalities as x
            diff_x = torch.nn.functional.pad(diff_x, (0, 1))
            # dim(bs, hidden_dim, len_ts-patch_window_size-1)
            embedding = self.convs[0](self.ts_scaler(diff_x))
            ts_var_embeddings.append(embedding)
        # original ts
        # dim(bs, hidden_dim, len_ts-patch_window_size-1)
        embedding = self.convs[1](self.ts_scaler(x))
        ts_var_embeddings.append(embedding)

        # split ts_var_embeddings into patches
        patched_ts_var_embeddings = []
        for i, embedding in enumerate(ts_var_embeddings):
            embedding = self.layer_norms[i](embedding)
            embedding = embedding.reshape(
                embedding.shape[0], self.num_patches, -1, embedding.shape[2])
            embedding = torch.mean(embedding, dim=2)
            patched_ts_var_embeddings.append(embedding)

        # concatenate diff_x, x, mu and std embeddinga and send them to the linear projector
        x_embeddings = torch.cat([
            torch.cat(patched_ts_var_embeddings, dim=-1),
            torch.cat(scalar_embeddings, dim=-1)
        ], dim=-1)
        x_embeddings = self.linear_encoder(x_embeddings)

        return x_embeddings


class ViTUnit(nn.Module):
    def __init__(self, hidden_dim, num_patches, depth, heads, mlp_dim, dim_head, dropout, device):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim, dropout=dropout, max_len=num_patches+1)
        self.cls_token = nn.Parameter(torch.randn(hidden_dim).to(device))
        self.transformer = Transformer(
            hidden_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x_embeddings, ps = pack([cls_tokens, x], 'b * d')
        x_embeddings = self.pos_encoder(
            x_embeddings.transpose(0, 1)).transpose(0, 1)
        x_embeddings = self.transformer(x_embeddings)
        cls_tokens, _ = unpack(x_embeddings, ps, 'b * d')
        return cls_tokens.reshape(cls_tokens.shape[0], -1)


class Mantis(
    nn.Module,
    PyTorchModelHubMixin,
    # optionally, you can add metadata which gets pushed to the model card
    library_name="mantis",
    repo_url="https://huggingface.co/paris-noah/Mantis-8M/tree/main",
    pipeline_tag="time-series-foundation-model",
    license="mit",
    tags=["time-series-foundation-model"],
):
    def __init__(self, seq_len=512, hidden_dim=256, num_patches=32, scalar_scales=None, hidden_dim_scalar_enc=32,
                 epsilon_scalar_enc=1.1, transf_depth=6, transf_num_heads=8, transf_mlp_dim=512, transf_dim_head=128,
                 transf_dropout=0.1, device='cuda:0', pre_training=False):
        """
        The architecture of Mantis foundation model.
        Parameters
        ----------
        seq_len: length of time series
        hidden_dim: size of a token, i.e., what is the hidden dimension each patch is projected to
        num_patches: number of patches
        scalar_scales: list of scales used for MultiScaledScalarEncoder in TokenGeneratorUnit
        hidden_dim_scalar_enc: hidden dimension used for MultiScaledScalarEncoder in TokenGeneratorUnit
        epsilon_scalar_enc: esplilon used for MultiScaledScalarEncoder in TokenGeneratorUnit
        transf_depth: layers of transformer used for Transformer in ViTUnit
        transf_num_heads: number of self-attention heads used for Transformer in ViTUnit
        transf_mlp_dim: hidden dim of mlp of feed forward layer of transformer used for Transformer in ViTUnit
        transf_dim_head: q,k,v dim used for Transformer in ViTUnit
        transf_dropout: dropout value used for Transformer in ViTUnit
        device: device
        pre_training: if True, applies a MLP projector after the ViTUnit, which originally was used to pre-train
        the model using InfoNCE contrastive loss.
        """

        super().__init__()
        assert (seq_len % num_patches) == 0, print(
            'Seq_len must be the multiple of num_patches')
        patch_window_size = int(seq_len / num_patches)

        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.scalar_scales = scalar_scales
        self.hidden_dim_scalar_enc = hidden_dim_scalar_enc
        self.epsilon_scalar_enc = epsilon_scalar_enc
        self.seq_len = seq_len
        self.pre_training = pre_training

        self.tokgen_unit = TokenGeneratorUnit(hidden_dim=hidden_dim,
                                              num_patches=num_patches,
                                              patch_window_size=patch_window_size,
                                              scalar_scales=scalar_scales,
                                              hidden_dim_scalar_enc=hidden_dim_scalar_enc,
                                              epsilon_scalar_enc=epsilon_scalar_enc)
        self.vit_unit = ViTUnit(hidden_dim=hidden_dim, num_patches=num_patches, depth=transf_depth,
                                heads=transf_num_heads, mlp_dim=transf_mlp_dim, dim_head=transf_dim_head,
                                dropout=transf_dropout, device=device)

        self.prj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.to(device)

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, x):
        x_embeddings = self.tokgen_unit(x)
        vit_out = self.vit_unit(x_embeddings)
        if self.pre_training:
            return self.prj(vit_out)
        else:
            return vit_out
