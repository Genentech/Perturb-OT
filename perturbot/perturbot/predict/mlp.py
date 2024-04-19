from typing import Sequence, Dict, Union, Tuple
from numbers import Number
import sys
import time
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim, nn
import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from scvi.train._callbacks import LoudEarlyStopping
from scvi.train._progress import ProgressBar
from perturbot.utils import mdict_to_matrix


class MLP(L.LightningModule):
    def __init__(self, n_input, hidden_layers: Sequence[int], n_output: int):
        super().__init__()
        model = nn.Sequential()
        self.log_scale = nn.Parameter(torch.randn(n_output))
        prev_dim = n_input
        for i, n_dim in enumerate(hidden_layers):
            model.add_module(f"dense{i}", nn.Linear(prev_dim, n_dim))
            model.add_module(f"batchnorm{i}", nn.BatchNorm1d(n_dim))
            model.add_module(f"act{i}", nn.ReLU())
            prev_dim = n_dim
        model.add_module(f"dense_out", nn.Linear(prev_dim, n_output))
        self.model = model
        self.batch_losses = []

    def nll(self, x, x_hat):
        scale = torch.exp(self.log_scale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return -log_pxz.sum()

    def training_step(self, batch, batch_idx):
        x = batch["source"]
        y = batch["target"][:, 0, :]
        y_hat = self.model(x).squeeze()
        loss = nn.functional.mse_loss(y_hat, y)
        # loss = self.nll(y, y_hat)
        self.batch_losses.append(loss)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        x = val_batch["source"]
        y = val_batch["target"][:, 0, :].squeeze()
        y_hat = self.model(x).squeeze()
        # loss = self.nll(y, y_hat)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_train_epoch_end(self):
        loss = torch.stack(self.batch_losses).mean()
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.batch_losses.clear()


class LabeledDataset(Dataset):
    def __init__(
        self,
        X_dict: Dict[Union[int, float], np.ndarray],
        Y_dict: Dict[Union[int, float], np.ndarray],
    ):
        """
        For data (X, Y) and coupling (T) dictionaries (label -> np.ndarray),
        for each X, sample Y according to the coupling probability.
        Assume T has uniform marginal on X.
        If T is not provided, uniform matrix is used.
        """
        assert sorted(X_dict.keys()) == sorted(Y_dict.keys())

        self.X_dict = X_dict
        self.Y_dict = Y_dict
        self.labels = sorted(X_dict.keys())
        self.X = np.concatenate([X_dict[l] for l in self.labels])
        self.Y = np.concatenate([Y_dict[l] for l in self.labels])
        self.label = np.concatenate(
            [np.ones(X_dict[l].shape[0]) * l for l in self.labels]
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = {"source": self.X[idx, :], "target": self.Y[idx, :]}
        return sample


class LabeledCouplingDataset(Dataset):
    def __init__(
        self,
        X_dict: Dict[Union[int, float], np.ndarray],
        Y_dict: Dict[Union[int, float], np.ndarray],
        T_dict: Dict[Union[int, float], np.ndarray] = None,
    ):
        """
        For data (X, Y) and coupling (T) dictionaries (label -> np.ndarray),
        for each X, sample Y according to the coupling probability.
        Assume T has uniform marginal on X.
        If T is not provided, uniform matrix is used.
        """
        assert sorted(X_dict.keys()) == sorted(Y_dict.keys())

        self.labels = sorted(X_dict.keys())
        self.X = torch.from_numpy(np.concatenate([X_dict[l] for l in self.labels]))
        self.Y = torch.from_numpy(np.concatenate([Y_dict[l] for l in self.labels]))

        if T_dict is None:
            T_dict = {}
            for l in self.labels:
                T = np.ones((X_dict[l].shape[0], Y_dict[l].shape[0]))
                T_dict[l] = T / T.sum()

        else:
            assert self.labels == sorted(T_dict.keys())
        self.T = torch.from_numpy(
            mdict_to_matrix(
                T_dict,
                np.concatenate([np.ones(X_dict[l].shape[0]) * l for l in self.labels]),
                np.concatenate([np.ones(Y_dict[l].shape[0]) * l for l in self.labels]),
            )
        )
        self.T[self.T.sum(axis=-1) == 0, :] = 1e-8

    def _sample_Y_flattened(self, x_idx):
        try:
            Y_idx = torch.multinomial(self.T[x_idx, :], 1)
        except RuntimeError as e:
            print(x_idx)
            torch.set_printoptions(threshold=10_000)
            print(self.T[x_idx, :])
            raise e
        return self.Y[Y_idx, :]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = {"source": self.X[idx, :], "target": self._sample_Y_flattened(idx)}
        return sample


def train(
    model: L.LightningModule,
    dataset: Dataset,
    val_prop: float = 0.05,
    batch_size: int = 256,
    seed: int = 2024,
    max_epochs: int = 2000,
    checkpoint_dir: str = ".",
    loader_kwargs={},
    log_dir="./",
):
    if not torch.cuda.is_available():
        torch.set_num_threads(10)
    # csv_logger = CSVLogger(save_dir=log_dir, name="csv_file")
    torch_seed = torch.Generator().manual_seed(seed)
    # seed_everything(seed, workers=True)
    train_set_size = int(len(dataset) * (1 - val_prop))
    valid_set_size = len(dataset) - train_set_size
    train_set, valid_set = random_split(
        dataset, [train_set_size, valid_set_size], generator=torch_seed
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, **loader_kwargs)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, **loader_kwargs)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        callbacks=[
            ProgressBar(),
            LoudEarlyStopping(monitor="val_loss", mode="min", patience=45),
        ],  # , EarlyStopping(monitor="val_loss", mode="min")],
        default_root_dir=checkpoint_dir,
        # logger=[csv_logger],
        # deterministic=True,
    )
    trainer.fit(model.double(), train_loader, valid_loader)
    return model


def train_mlp(
    train_data: Tuple[Dict[Number, np.array], Dict[Number, np.array]],
    T_dict: Dict[Number, np.array],
) -> Tuple[nn.Module, Dict]:
    """Trains MLP predicting Y from X given the labeled sample-matching T_dict.

    Parameters
    ----------
    data :
        (source dataset, target dataset) where source and target datasets
        are the dictionaries mapping label to np.ndarray with matched labels.
    T_dict :
        Optimal Transport coupling between the samples per label

    Returns
    -------
    model :
        Trained predictor
    log :
        Training log
    """
    Xs_dict, Xt_dict = train_data
    dim_X = Xs_dict[list(Xs_dict.keys())[0]].shape[1]
    dim_Y = Xt_dict[list(Xt_dict.keys())[0]].shape[1]
    if isinstance(T_dict, np.ndarray):
        # T is all-to-all mapping, make it into a dictionary
        dataset = LabeledCouplingDataset(
            {0: np.concatenate([Xs_dict[l] for l in Xs_dict.keys()], axis=0)},
            {0: np.concatenate([Xt_dict[l] for l in Xs_dict.keys()], axis=0)},
            {0: T_dict},
        )
    else:
        dataset = LabeledCouplingDataset(Xs_dict, Xt_dict, T_dict)

    mlp = MLP(dim_X, [min(dim_X, 256), min(dim_X, 256)], dim_Y)
    start_time = time.time()
    model = train(mlp, dataset).model
    log = {"time": start_time - time.time()}
    return model, log


class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar
