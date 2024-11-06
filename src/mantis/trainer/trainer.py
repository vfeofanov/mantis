import torch
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from mantis.architecture import Mantis8M

from .trainer_utils.architecture import FineTuningNetwork
from .trainer_utils.dataset import LabeledDataset
from .trainer_utils.scheduling import adjust_learning_rate


class MantisTrainer:
    """
    A scikit-learn-like class to use Mantis as a feature extractor or fine-tune it to the downstream task.

    Parameters
    ----------
    device: {'cpu', 'cuda'}
        On which device the model is located and trained.
    network: Mantis, default=None
        The foundation model. If None, the class initializes a Mantis object by itself (so weights are randomly
        initialized). Otherwise, pass a pre-trained model.
    """
    def __init__(self, device, network=None):
        self.device = device
        if network is None:
            network = Mantis8M(seq_len=512, hidden_dim=256, num_patches=32, scalar_scales=None, hidden_dim_scalar_enc=32,
                             epsilon_scalar_enc=1.1, transf_depth=6, transf_num_heads=8, transf_mlp_dim=512,
                             transf_dim_head=128, transf_dropout=0.1, device=device, pre_training=False)
        self.network = network.to(device)

    def fit(self, x, y, fine_tuning_type='full', adapter=None, head=None, num_epochs=500, batch_size=256,
            base_learning_rate=2e-4, init_optimizer=None, criterion=None, learning_rate_adjusting=True):
        """
        Fit (fine-tune) the foundation model to the downstream task.

        Parameters
        ----------
        x: array-like of shape (n_samples, n_channels, seq_len)
            The input samples. If data is univariate case, the shape should be ``(n_samples, 1, seq_len)``.
            ``seq_len`` should correspond to ``self.network.seq_len``.
        y: array-like of shape (n_samples,)
            The class labels with the following unique_values: ``[i for i in range(n_classes)]``
        fine_tuning_type: {'full', 'adapter_head', 'head', 'scratch'}, default='full'
            fine-tuning type
        adapter: nn.Module, default=None
            Adapter is a part of the network that precedes the foundation model and reduces the original data matrix
            of shape ``(n_samples, n_channels, seq_len)`` to ``(n_samples, new_n_channels, seq_len)``. By default,
            adapter is not used.
        head: nn.Module, default=None
            Head is a part of the network that follows the foundation model and projects from the embedding space
            to the probability matrix of shape ``(n_samples, n_classes)``. By default, head is a linear layer ``Linear``
            preceded by the layer normalization ``LayerNorm``.
        num_epochs: int, default=500
            Number of training epochs.
        batch_size: int, default=256
            Batch size.
        base_learning_rate: float, default=2e-4
            Learning rate that optimizer starts from. If ``learning_rate_adjusting`` is ``False``,
            it remains to be fixed
        init_optimizer: callable, default=None
            Function that initializes the optimizer. By default, ``AdamW`` 
            with pre-defined hyperparameters (except the learning rate) is used.
        criterion: nn.Module, default=None
            Learning criterion. By default, ``CrossEntropyLoss`` is used. 
        learning_rate_adjusting: bool, default=True
            Whether to use the implemented scheduling scheme.
        
        Returns
        -------
        self.fine_tuned_model: nn.Module
            Network fine-tuned to the downstream task.
        """
        
        self.fine_tuning_type = fine_tuning_type
        # ==== get the whole fine-tuning architecture ====
        # init head
        if head is None:
            head = nn.Sequential(
                nn.LayerNorm(self.network.hidden_dim * x.shape[1]),
                nn.Linear(self.network.hidden_dim *
                          x.shape[1], np.unique(y).shape[0])
            ).to(self.device)
        else:
            head = head.to(self.device)
        # init adapter
        if adapter is not None:
            adapter = adapter.to(self.device)
        else:
            adapter = None
        # when fine-tuning head, the forward pass over the encoder will be done only once (see init data_loader below)
        self.fine_tuned_model = FineTuningNetwork(
            deepcopy(self.network), head, adapter).to(self.device)

        # ==== get params to fine-tune and set them into the training model ====
        parameters = self._get_fine_tuning_params(
            fine_tuning_type=fine_tuning_type)
        self.fine_tuned_model.eval()
        self._set_train(fine_tuning_type=fine_tuning_type)

        # ==== init criterion, optimizer and dataloader ====
        # init criterion if None
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # init optimizer by init_optimizer
        if init_optimizer is None:
            optimizer = torch.optim.AdamW(
                parameters, lr=base_learning_rate, betas=(0.9, 0.999), weight_decay=0.05)
        else:
            optimizer = init_optimizer(parameters)

        # init dataloader, for the head fine-tuning we directly load the embeddings
        train_dataset = LabeledDataset(x, y)
        data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

        # ==== training loop ====
        progress_bar = tqdm(range(num_epochs))
        step = 1
        for epoch in progress_bar:
            loss_list = []
            for (x_batch, y_batch) in data_loader:
                # adjust learning rate
                if learning_rate_adjusting:
                    adjust_learning_rate(
                        num_epochs, optimizer, data_loader, step, base_learning_rate)
                # read data
                x_batch, y_batch = x_batch.to(
                    self.device), y_batch.to(self.device)
                step += 1
                # forward
                output = self.fine_tuned_model(x_batch)
                loss = criterion(output, y_batch)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            # report average loss over all batches
            avg_loss_in_epoch = np.mean(loss_list)
            progress_bar.set_description("Epoch {:d}: Train Loss {:.4f}".format(
                epoch, avg_loss_in_epoch), refresh=True)
        return self.fine_tuned_model

    def transform(self, x, batch_size=256, three_dim=False, to_numpy=True):
        """
        Projects to the embedding space using self.network.

        Parameters
        ----------
        x: array-like of shape (n_samples, n_channels, seq_len)
            The input samples. If data is univariate case, the shape should be (n_samples, 1, seq_len).
            ``seq_len`` should correspond to ``self.network.seq_len``. In the multivariate case, each channel is sent
            independently to the foundation model.
        batch_size: int, default=256
            To fit memory, the data matrix is split into the chunks for inference. ``batch_size`` corresponds to
            the chunk size.
        three_dim: bool, default=False
            Whether the output should be two- or three-dimensional. By default, the embeddings of all channels are
            concatenated along the same axis, so the output is of shape (n_samples, n_channels * hidden_dim). When
            three_dim is set to True, the output is of shape (n_samples, n_channels, hidden_dim).
        to_numpy: bool, default=True
            Whether to convert the output to a numpy array.

        Returns
        -------
        z: array-like of shape (n_samples, n_channels * hidden_dim) or (n_samples, n_channels, hidden_dim)
            Embeddings.
        """
        concat = np.concatenate if to_numpy else torch.cat
        # apply network to each channel
        if three_dim:
            return concat([
                self._transform(x[:, [i], :], batch_size=batch_size, to_numpy=to_numpy)[
                    :, None, :]
                for i in range(x.shape[1])
            ], axis=1)
        else:
            return concat([
                self._transform(x[:, [i], :], batch_size=batch_size, to_numpy=to_numpy)
                for i in range(x.shape[1])
            ], axis=1)

    def _transform(self, x, batch_size=256, to_numpy=True):
        self.network.eval()
        dataloader = self._prepare_dataloader_for_inference(x, batch_size)
        outs = []
        for _, batch in enumerate(dataloader):
            x = batch[0].to(self.device)
            with torch.no_grad():
                out = self.network(x)
            outs.append(out)
        outs = torch.cat(outs)
        self.network.train()
        if to_numpy:
            return outs.cpu().numpy()
        else:
            return outs

    def predict_proba(self, x, batch_size=256, to_numpy=True):
        """
        Predicts the class probability matrix using self.fine_tuned_model.

        Parameters
        ----------
        x: array-like of shape (n_samples, n_channels, seq_len)
            The input samples. If data is univariate case, the shape should be (n_samples, 1, seq_len).
            ``seq_len`` should correspond to ``self.network.seq_len``.
        batch_size: int, default=256
            To fit memory, the data matrix is split into the chunks for inference. ``batch_size`` corresponds to
            the chunk size.
        to_numpy: bool, default=True
            Whether to convert the output to a numpy array.
        
        Returns
        -------
        probs: array_like of shape (n_samples, n_classes)
            Class probability matrix.
        """

        self.fine_tuned_model.eval()
        dataloader = self._prepare_dataloader_for_inference(x, batch_size)
        outs = []
        for _, batch in enumerate(dataloader):
            x = batch[0].to(self.device)
            with torch.no_grad():
                out = torch.softmax(self.fine_tuned_model(x), dim=-1)
            outs.append(out.cpu())
        outs = torch.cat(outs)
        if to_numpy:
            return outs.cpu().numpy()
        else:
            return outs

    def predict(self, x, batch_size=256, to_numpy=True):
        """
        Predicts the class labels using self.fine_tuned_model.

        Parameters
        ----------
        x: array-like of shape (n_samples, n_channels, seq_len)
            The input samples. If data is univariate case, the shape should be (n_samples, 1, seq_len).
            ``seq_len`` should correspond to ``self.network.seq_len``.
        batch_size: int, default=256
            To fit memory, the data matrix is split into the chunks for inference. ``batch_size`` corresponds to
            the chunk size.
        to_numpy: bool, default=True
            Whether to convert the output to a numpy array.
        
        Returns
        -------
        y: array_like of shape (n_samples,)
            Class labels.
        """
        probs = self.predict_proba(x, batch_size=batch_size, to_numpy=to_numpy)
        return probs.argmax(axis=1)

    def _prepare_dataloader_for_inference(self, x, batch_size):
        if isinstance(x, torch.Tensor):
            dataset = TensorDataset(x.type(torch.float))
        else:
            dataset = TensorDataset(torch.tensor(x, dtype=torch.float))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}({', '.join(f'{k}={v}' for k, v in vars(self).items() if k not in ['network'])})"

    def _get_fine_tuning_params(self, fine_tuning_type):
        tune_params_dict = {
            "full": [
                self.fine_tuned_model.parameters()
            ],
            "scratch": [
                self.fine_tuned_model.parameters()
            ],
            "head": [
                self.fine_tuned_model.head.parameters()
            ],
            "adapter_head": [
                [] if self.fine_tuned_model.adapter is None else self.fine_tuned_model.adapter.parameters(),
                self.fine_tuned_model.head.parameters()
            ]
        }
        params_list = list(chain(*tune_params_dict[fine_tuning_type]))
        return params_list

    def _set_train(self, fine_tuning_type):
        if fine_tuning_type in ["full", "scratch"]:
            self.fine_tuned_model.train()
        elif fine_tuning_type == "head":
            self.fine_tuned_model.head.train()
        elif fine_tuning_type == "adapter_head":
            self.fine_tuned_model.adapter.train()
            self.fine_tuned_model.head.train()
        else:
            raise KeyError("Unknown fine_tuning_type")
