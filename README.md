> [!WARNING]
> This package is in a phase of the first release preparation. The package can be already used, but some changes may be applied. We appreciate your understanding. 
> 

# Mantis: Foundation Model with Adapters for Multichannel Time Series Classification

<p align="center">
  <img src="figures/mantis_logo_white_with_font.png" alt="Logo" height="300"/>
</p>

## Overview

**MANTIS** (foundation **M**odel with **A**dapters for multicha**N**nel **TI**me **S**eries Classification) is an open-source python package with a pre-trained time series classification foundation model implemented by Huawei Noah's Ark Lab.

Please find out technical report on arxiv (available soon) and the model checkpoint on [Hugging Face](https://huggingface.co/paris-noah/Mantis-8M).

## Installation

```
pip install mantis
```

### Editable mode using poetry

First, install poetry and add the path to the binary file to your shell configuration file. 
For example, on linux systems it can be performed by running the following commands:
```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/username/.local/bin:$PATH"
```
Now you can create a virtual environment that is based on one of your already installed python interpeter.
For example, if your default python is 3.9, then create the environment by running:
```bash
poetry use env 3.9
```
Alternatively, you can give a path to the interpreter, for example, if you want to use Anaconda python interpreter:
```bash
poetry env use /path/to/anaconda3/envs/my_env/bin/python
```
Activate a new shell session within the environment by
```bash
poetry shell
```
Then, install the package (in editable mode) with the dependencies by running:
```bash
poetry install
```
If it doesn't work for some reason you can try to re-generate the lock file:
```bash
poetry lock
poetry install
```


## Getting started

Please refer to `getting_started/` folder to see reproducible examples of how the package can be used.

Below we summarize the basic commands needed to use the package.

### Initialization.

To load our pre-trained model with 8M parameters from the Hugging Face, it is sufficient to run:

``` python
from mantis.architecture import Mantis8M

network = Mantis8M(device='cuda')
network = network.from_pretrained("paris-noah/Mantis-8M")
```

### Feature Extraction.

We provide a scikit-learn-like wrapper `MantisTrainer` that allows to use Mantis as a feature extractor by running the following commands:

``` python
from mantis.trainer import MantisTrainer

model = MantisTrainer(device='cuda', network=network)
Z = model.transform(X) # X is your time series dataset
```

### Fine-tuning.

If you want to fine-tune the model on your supervised dataset, you can use `fit` method of `MantisTrainer`:

``` python
from mantis.trainer import MantisTrainer

model = MantisTrainer(device='cuda', network=network)
model.fit(X, y) # y is a vector with class labels
probs = model.predict_proba(X)
y_pred = model.predict(X)
```

### Adapters.

We have integrated into the framework the possibility to pass the input to an adapter before sending it to the foundation model. This may be useful for time series data sets with a large number of channels. More specifically, large number of channels may induce the curse of dimensionality or make model's fine-tuning unfeasible. 

A straightforward way to overcome these issues is to use a dimension reduction approach like PCA:
``` python
from mantis.adapters import MultichannelProjector

adapter = MultichannelProjector(new_num_channels=5, base_projector='pca')
adapter.fit(X)
X_transformed = adapter.transform(X)

model = MantisTrainer(device='cuda', network=network)
Z = model.transform(X_transformed)
```

Another wat is to add learnable layers before the foundation model and fine-tune them with the prediction head:
``` python
from mantis.adapters import LinearChannelCombiner

model = MantisTrainer(device='cuda', network=network)
adapter = LinearChannelCombiner(num_channels=X.shape[1], new_num_channels=5)
model.fit(X, y, adapter=adapter, fine_tuning_type='adapter_head')
```

## Structure

```
├── data/                <-- two datasets for demonstration
├── getting_started/     <-- jupyter notebooks with tutorials
└── src/mantis/          <-- the main package
    ├── adapters/        <-- adapters for multichannel time series
    ├── architecture/    <-- foundation model architectures
    └── trainer/         <-- a scikit-learn-like wrapper for feature extraction or fine-tuning
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Open-source Participation

We would be happy to receive feedback and integrate any suggestion, so do not hesitate to contribute to this project by raising a GitHub issue or contacting us by email:

 - Vasilii Feofanov - vasilii [dot] feofanov [at] huawei [dot] com


## Citing Mantis 📚

If you use Mantis in your work, please cite this technical report:

```bibtex
@misc{feofanov2024mantis,
      title={TODO}, 
      author={TODO},
      year={2024},
      eprint={TODO},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={TODO}, 
}
```
