from typing import Sequence, Dict, Union, Tuple
import collections.abc
import numpy as np

import torch
from torch import nn
from torch.nn.functional import one_hot
from torch.nn import CrossEntropyLoss
import lightning as L
from scvi.nn import FCLayers  # , one_hot
from scvi.module import Classifier
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


class AbstractVAE(L.LightningModule):
    # https://github.com/williamFalcon/pytorch-lightning-vae/blob/main/vae.py
    def __init__(
        self,
    ):
        super().__init__()

        self.save_hyperparameters()

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=1)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = kl - recon_loss
        elbo = elbo.mean()

        self.log_dict(
            {
                "elbo": elbo,
                "kl": kl.mean(),
                "recon_loss": recon_loss.mean(),
                "reconstruction": recon_loss.mean(),
                "kl": kl.mean(),
            }
        )

        return elbo


class MatchVAE(AbstractVAE):
    def __init__(
        self,
        n_input1,
        n_input2,
        n_hidden,
        n_latent,
        n_layers,
        n_labels,
        mod2_loss_scale=None,
        adv_loss_scale=1.0,
        weight_decay=1e-6,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.log_scale1 = nn.Parameter(torch.zeros(n_input1))
        self.log_scale2 = nn.Parameter(torch.zeros(n_input2))
        # encoder, decoder
        self.encoder1 = FCLayers(
            n_in=n_input1,
            n_out=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )
        self.decoder1 = FCLayers(
            n_in=n_latent,
            n_out=n_input1,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )
        self.encoder2 = FCLayers(
            n_in=n_input2,
            n_out=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )
        self.decoder2 = FCLayers(
            n_in=n_latent,
            n_out=n_input2,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

        # distribution parameters
        self.fc_mu1 = nn.Linear(n_latent, n_latent)
        self.fc_var1 = nn.Linear(n_latent, n_latent)
        self.fc_mu2 = nn.Linear(n_latent, n_latent)
        self.fc_var2 = nn.Linear(n_latent, n_latent)

        self.adversarial_classifier = Classifier(
            n_input=n_latent + n_labels,
            n_hidden=32,
            n_labels=n_labels,
            n_layers=2,
            logits=True,
        )
        if mod2_loss_scale is None:
            mod2_loss_scale = n_input1 / n_input2
        self.mod2_loss_scale = mod2_loss_scale
        self.adv_loss_scale = adv_loss_scale
        self.weight_decay = weight_decay
        self.n_labels = n_labels

    def encode(self, x1, x2):
        z1 = self.fc_mu1(self.encoder1(x1))
        z2 = self.fc_mu2(self.encoder2(x2))
        return z1, z2

    def configure_optimizers(self):
        """Configure optimizers for adversarial training."""
        # params1 = filter(lambda p: p.requires_grad, self.parameters())
        generators = [
            self.encoder1.parameters(),
            self.decoder1.parameters(),
            self.encoder2.parameters(),
            self.decoder2.parameters(),
            self.fc_mu1.parameters(),
            self.fc_var1.parameters(),
            self.fc_mu2.parameters(),
            self.fc_var2.parameters(),
        ]
        param1 = []
        for g in generators:
            param1 += list(g)
        param1 + [self.log_scale1, self.log_scale2]
        optimizer1 = torch.optim.Adam(
            param1,
            lr=1e-3,
            eps=0.01,
            weight_decay=self.weight_decay,
        )
        config1 = {"optimizer": optimizer1}

        params2 = filter(
            lambda p: p.requires_grad, self.adversarial_classifier.parameters()
        )
        optimizer2 = torch.optim.Adam(
            params2, lr=1e-3, eps=0.01, weight_decay=self.weight_decay
        )
        config2 = {"optimizer": optimizer2}

        # pytorch lightning requires this way to return
        opts = [config1.pop("optimizer"), config2["optimizer"]]
        if "lr_scheduler" in config1:
            scheds = [config1["lr_scheduler"]]
            return opts, scheds
        else:
            return opts

    def loss_adversarial_classifier(
        self, z, batch_index, label_index, predict_true_class=True
    ):
        """Loss for adversarial classifier."""
        n_classes = self.n_labels
        onehot_labels = one_hot(label_index, n_classes)
        logits = self.adversarial_classifier(torch.concat([z, onehot_labels], axis=-1))

        if predict_true_class:
            cls_target = one_hot(batch_index, n_classes).float()
        else:
            one_hot_batch = one_hot(batch_index, n_classes)
            # place zeroes where true label is
            cls_target = (~one_hot_batch.bool()).float()
            cls_target = cls_target / (n_classes - 1)

        ce_loss = CrossEntropyLoss()
        loss = ce_loss(logits, cls_target)
        return loss

    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()
        x1, x2, x1_label, x2_label = batch

        # encode x to get the mu and variance parameters
        z1 = self.encoder1(x1)
        mu1, log_var1 = self.fc_mu1(z1), self.fc_var1(z1)

        # sample z from q
        std1 = torch.exp(log_var1 / 2)
        q1 = torch.distributions.Normal(mu1, std1)
        z1 = q1.rsample()

        # decoded
        x_hat1 = self.decoder1(z1)

        # reconstruction loss
        recon_loss1 = self.gaussian_likelihood(x_hat1, self.log_scale1, x1)

        # kl
        kl1 = self.kl_divergence(z1, mu1, std1)

        # elbo
        elbo1 = kl1 - recon_loss1
        elbo1 = elbo1.sum() / (x1.shape[0] + x2.shape[0])

        # encode x to get the mu and variance parameters
        z2 = self.encoder1(x2)
        mu2, log_var2 = self.fc_mu1(z2), self.fc_var1(z2)

        # sample z from q
        std2 = torch.exp(log_var2 / 2)
        q2 = torch.distributions.Normal(mu2, std2)
        z2 = q2.rsample()

        # decoded
        x_hat2 = self.decoder1(z2)

        # reconstruction loss
        recon_loss2 = self.gaussian_likelihood(x_hat2, self.log_scale2, x2)

        # kl
        kl2 = self.kl_divergence(z2, mu2, std2)

        # elbo
        elbo2 = kl2 - recon_loss2
        elbo2 = elbo2.sum() / (x1.shape[0] + x2.shape[0])

        z = torch.concat([z1, z2], axis=0)
        modality = (
            torch.concat([torch.zeros(z1.shape[0]), torch.zeros(z2.shape[0])], axis=0)
            .long()
            .to(z.device)
        )
        label = torch.concat([x1_label, x2_label], axis=0)
        fool_loss = self.loss_adversarial_classifier(z, modality, label, False)

        loss = elbo1 + elbo2 * self.mod2_loss_scale + fool_loss * self.adv_loss_scale
        opt1.zero_grad()
        self.manual_backward(loss)
        opt1.step()

        adv_classifier_loss = (
            self.loss_adversarial_classifier(z.detach(), modality, label, True)
            * self.adv_loss_scale
        )
        opt2.zero_grad()
        self.manual_backward(adv_classifier_loss)
        opt2.step()

        self.log_dict(
            {
                "elbo1": elbo1,
                "kl1": kl1.mean(),
                "recon_loss1": recon_loss1.mean(),
                "kl1": kl1.mean(),
                "elbo2": elbo2,
                "kl2": kl2.mean(),
                "recon_loss2": recon_loss2.mean(),
                "kl2": kl2.mean(),
                "adv_loss": fool_loss,
                "train_loss": loss,
                "train_adv_classifier_loss": adv_classifier_loss,
                "batch_size": x1.shape[0] + x2.shape[0],
            }
        )
        # self.log(batch_size=x1.shape[0] + x2.shape[0])

    def validation_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()
        x1, x2, x1_label, x2_label = batch

        # encode x to get the mu and variance parameters
        z1 = self.encoder1(x1)
        mu1, log_var1 = self.fc_mu1(z1), self.fc_var1(z1)

        # sample z from q
        std1 = torch.exp(log_var1 / 2)
        q1 = torch.distributions.Normal(mu1, std1)
        z1 = q1.rsample()

        # decoded
        x_hat1 = self.decoder1(z1)

        # reconstruction loss
        recon_loss1 = self.gaussian_likelihood(x_hat1, self.log_scale1, x1)

        # kl
        kl1 = self.kl_divergence(z1, mu1, std1)

        # elbo
        elbo1 = kl1 - recon_loss1
        elbo1 = elbo1.sum() / (x1.shape[0] + x2.shape[0])

        # encode x to get the mu and variance parameters
        z2 = self.encoder1(x2)
        mu2, log_var2 = self.fc_mu1(z2), self.fc_var1(z2)

        # sample z from q
        std2 = torch.exp(log_var2 / 2)
        q2 = torch.distributions.Normal(mu2, std2)
        z2 = q2.rsample()

        # decoded
        x_hat2 = self.decoder1(z2)

        # reconstruction loss
        recon_loss2 = self.gaussian_likelihood(x_hat2, self.log_scale2, x2)

        # kl
        kl2 = self.kl_divergence(z2, mu2, std2)

        # elbo
        elbo2 = kl2 - recon_loss2
        elbo2 = elbo2.sum() / (x1.shape[0] + x2.shape[0])

        z = torch.concat([z1, z2], axis=0)
        modality = (
            torch.concat([torch.zeros(z1.shape[0]), torch.zeros(z2.shape[0])], axis=0)
            .long()
            .to(z.device)
        )
        label = torch.concat([x1_label, x2_label], axis=0)
        fool_loss = self.loss_adversarial_classifier(z, modality, label, False)

        loss = elbo1 + elbo2 * self.mod2_loss_scale + fool_loss * self.adv_loss_scale

        adv_classifier_loss = (
            self.loss_adversarial_classifier(z.detach(), modality, label, True)
            * self.adv_loss_scale
        )
        self.log_dict(
            {
                "val_elbo1": elbo1,
                "val_kl1": kl1.mean(),
                "val_recon_loss1": recon_loss1.mean(),
                "val_kl1": kl1.mean(),
                "val_elbo2": elbo2,
                "val_kl2": kl2.mean(),
                "val_recon_loss2": recon_loss2.mean(),
                "val_kl2": kl2.mean(),
                "val_adv_loss": fool_loss,
                "val_loss": loss,
                "val_train_adv_classifier_loss": adv_classifier_loss,
            }
        )
        # self.log(batch_size=x1.shape[0] + x2.shape[0])

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, Tuple):
            # move all tensors in your custom data structure to the device
            return tuple(item.to(device) for item in batch)
            batch.samples = batch.samples.to(device)
            batch.targets = batch.targets.to(device)
        elif dataloader_idx == 0:
            # skip device transfer for the first dataloader or anything you wish
            pass
        else:
            print(type(batch))
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch


class LabeledBimodalDataset(Dataset):
    def __init__(
        self,
        X_dict: Dict[Union[int, float], np.ndarray],
        Y_dict: Dict[Union[int, float], np.ndarray],
        **kwargs,
    ):
        """
        For data (X, Y) and coupling (T) dictionaries (label -> np.ndarray),
        for each X, sample Y according to the coupling probability.
        Assume T has uniform marginal on X.
        If T is not provided, uniform matrix is used.
        """
        super().__init__()
        assert sorted(X_dict.keys()) == sorted(Y_dict.keys())
        self.labels = sorted(X_dict.keys())
        self.labels_to_id = {i: l for i, l in enumerate(self.labels)}
        X = np.zeros((0, X_dict[self.labels[0]].shape[1]))
        Y = np.zeros((0, Y_dict[self.labels[0]].shape[1]))
        modality = np.zeros((0))
        labels_X = np.zeros((0))
        labels_Y = np.zeros((0))
        for l in self.labels:
            lid = self.labels_to_id[l]
            X = np.concatenate([X, X_dict[l]])
            Y = np.concatenate([Y, Y_dict[l]])
            modality = np.concatenate(
                [
                    modality,
                    np.zeros(X_dict[l].shape[0], dtype=int),
                    np.zeros(Y_dict[l].shape[0], dtype=int),
                ]
            )
            labels_X = np.concatenate(
                [labels_X, np.ones(X_dict[l].shape[0], dtype=int) * lid]
            )
            labels_Y = np.concatenate(
                [labels_Y, np.ones(X_dict[l].shape[0], dtype=int) * lid]
            )
        self.X = X
        self.Y = Y
        self.modality = modality
        self.labels_X = labels_X
        self.labels_Y = labels_Y
        self.data_idx = np.arange(0, self.X.shape[0] + self.Y.shape[0])
        np.random.shuffle(self.data_idx)
        self.idx_order = np.argsort(self.data_idx)
        # data_idx = 3 4 0 | 1 2
        # idx_order= 2 3 4 | 0 1
        # idx = 0, 1, 2
        # idx_order[idx] = [2, 3, 4]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X_idx = self.idx_order[idx][self.idx_order[idx] < self.X.shape[0]]
        Y_idx = (
            self.idx_order[idx][self.idx_order[idx] >= self.X.shape[0]]
            - self.X.shape[0]
        )
        X = self.X[X_idx, :]
        Y = self.Y[Y_idx, :]
        labels_X = self.labels_X[X_idx]
        labels_Y = self.labels_Y[Y_idx]
        return X, Y, labels_X, labels_Y


def collate_fn(data):
    """
    data: is a list of tuples with (example, label, length)
          where 'example' is a tensor of arbitrary shape
          and label/length are scalars
    """
    x, y, labels_x, labels_y = zip(*data)

    return (
        torch.from_numpy(np.vstack(x)),
        torch.from_numpy(np.vstack(y)),
        torch.from_numpy(np.concatenate(labels_x)).long(),
        torch.from_numpy(np.concatenate(labels_y)).long(),
    )
