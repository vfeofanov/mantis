# Changelog

## [0.1.0] - 2025-02-25
### Added
- Initial release with basic functionality.

## [0.2.0] - 2025-08-20
### Added
- `pretrain`, `load` and `save` methods to `MantisTrainer`.
- auxiliary functions for pre-training: `RandomCropResize`, `ContrastiveLoss`, `UnlabeledDataset`.
- `getting_started/pretrain.py` that demonstrates how the model can be pre-trained.

## [1.0.0] - 2026-02-19
### Added
- `Mantis8M` is renamed to `MantisV1`. We still keep `Mantis8M` for legacy, but it is adviced to use `MantisV1` that contains new functionality (`return_transf_layer` and `output_token` arguments).
- In the same spirit, `ViTUnit` is renamed to `TransformerUnit`, `Mantis8M.vit_unit` to `Mantis8M.transf_unit`.
- `architecture.py` is renamed to `version1.py`, `vit_utils` to `transformer_v1_utils`.
- added `MantisV2` with supporting `transformer_v2_utils`.
- new functionality for architectures: (a) `return_transf_layer=i` means that the network outputs the embedding of the i-th transformer layer; (b) `output_token` decides how to aggregate the output: `"cls_token"` returns the classification token only (default), `"mean_token"` calculates the mean over non-classification tokens, `"combined"` returns the concatenation of `"cls_token"` and `"mean_token"`.
- `MantisTrainer.pretrain`: supports now a Hugging Face dataset instead of a numpy array. In this case, `x` is directly sent to `DataLoader`.
- `MantisTrainer.fit`: set `requires_grad=False` for those parameters that are not fine-tuned.
- `MantisTrainer.fit`: fixed head fine-tuning: the forward pass over the encoder is performed only once to save computational time.
- `getting_started/intermediate_layers.ipynb` demonstrates how to use `return_transf_layer` and `output_token` arguments.
- updated tests.

## [1.1.0] - 2026-08-15
### Added
- support of the UTICA checkpoint (`fegounna/Utica`) for the `MantisV1` architecture, with `getting_started/utica.ipynb` demonstrating how to load it and how `return_transf_layer` and `output_token` affect its accuracy.
- `getting_started/mantis_and_tivit.ipynb` shows how to concatenate our embeddings with those of [TiViT](https://github.com/ExplainableML/TiViT). The notebook is self-contained and does not add any dependency to the package.
- `getting_started/self_ensembling.ipynb` demonstrates self-ensembling: the input sequence is resized to several lengths, each of them is passed through the network, and the outputs are concatenated to form the final embedding.

### Fixed
- `MantisV1.from_pretrained` and `MantisV2.from_pretrained`: the device of the network is no longer taken from the checkpoint repository. Previously, loading from a repository without `config.json` built the network on the default device instead of the requested one, which failed on machines without CUDA.

### Changed
- `MantisTrainer.fit`: the default prediction head is now a linear layer preceded by `BatchNorm1d` instead of `LayerNorm`, as it delivers superior performance. A trailing batch with a single sample is dropped, since it cannot be batch-normalized. **This changes the results of fine-tuning runs that rely on the default head**; pass `head` explicitly to `fit` to keep the previous behavior.
- relaxed the `safetensors` requirement from `>=0.4,<0.5` to `>=0.4`, which is needed to support Python 3.13 and 3.14.
- switched dependency management from Poetry to [uv](https://docs.astral.sh/uv/): dependencies are declared in the standard `[project]` table of `pyproject.toml` and pinned in `uv.lock`.
- fixed the ambiguity with the license: it is Apache 2.0.
