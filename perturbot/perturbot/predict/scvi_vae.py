from typing import Dict, Union, Tuple
import numpy as np
import pandas as pd
import anndata as ad
import torch
import scvi
from tqdm import tqdm
import time


def make_adata(X_dict, Y_dict):
    adata_X = ad.AnnData(
        X=np.concatenate([v for k, v in X_dict.items()]),
        obs=pd.DataFrame(
            {
                "modality": "source",
                "labels": np.concatenate(
                    [np.repeat(k, v.shape[0]) for k, v in X_dict.items()]
                ),
            }
        ),
        var=pd.DataFrame(
            index=[f"PC{n+1}_X" for n in range(X_dict[list(X_dict.keys())[0]].shape[1])]
        ),
    )
    adata_Y = ad.AnnData(
        X=np.concatenate([v for k, v in Y_dict.items()]),
        obs=pd.DataFrame(
            {
                "modality": "target",
                "labels": np.concatenate(
                    [np.repeat(k, v.shape[0]) for k, v in Y_dict.items()]
                ),
            }
        ),
        var=pd.DataFrame(
            index=[f"PC{n+1}_Y" for n in range(Y_dict[list(Y_dict.keys())[0]].shape[1])]
        ),
    )
    sim_multi = ad.concat([adata_X, adata_Y], axis=1)[:0, :].copy()
    return sim_multi


def train_vae_model(
    data_dict,
    eps: Union[float, Tuple[float]] = None,
    use_label=True,
):
    """VAE with adversarial loss. Assumes normal likelihood for the observations."""
    if isinstance(eps, tuple):
        eps, latent_dim, learning_rate = eps
    else:
        latent_dim = 50
        learning_rate = 1e-4
    if not torch.cuda.is_available():
        scvi.settings.num_threads = 10
    else:
        torch.set_float32_matmul_precision("medium")
    X_dict = data_dict[0]
    Y_dict = data_dict[1]
    dim_X = X_dict[list(X_dict.keys())[0]].shape[1]
    dim_Y = Y_dict[list(Y_dict.keys())[0]].shape[1]
    assert X_dict.keys() == Y_dict.keys()
    adata_X = ad.AnnData(
        X=np.concatenate([v for k, v in X_dict.items()]),
        obs=pd.DataFrame(
            {
                "modality": "source",
                "labels": np.concatenate(
                    [np.repeat(k, v.shape[0]) for k, v in X_dict.items()]
                ),
            }
        ),
        var=pd.DataFrame(index=[f"PC{n+1}_X" for n in range(dim_X)]),
    )
    adata_Y = ad.AnnData(
        X=np.concatenate([v for k, v in Y_dict.items()]),
        obs=pd.DataFrame(
            {
                "modality": "target",
                "labels": np.concatenate(
                    [np.repeat(k, v.shape[0]) for k, v in Y_dict.items()]
                ),
            }
        ),
        var=pd.DataFrame(index=[f"PC{n+1}_Y" for n in range(dim_Y)]),
    )
    sim_multi = make_adata(X_dict, Y_dict)
    if not use_label:
        sim_multi.obs.labels = 0
    adata_mvi = scvi.data.organize_multiome_anndatas(sim_multi, adata_X, adata_Y)
    scvi.model.MATCHVI.setup_anndata(adata_mvi, batch_key="modality")
    model = scvi.model.MATCHVI(
        adata_mvi,
        n_genes=adata_X.n_vars,
        n_regions=adata_Y.n_vars,
        gene_likelihood="normal",
        accessibility_likelihood="normal",
        n_hidden=min(256, adata_X.n_vars),
        n_latent=latent_dim,
    )
    start_time = time.time()
    model.train(
        plan_kwargs={"match": True, "adversarial_classifier": True},
        scale_match_loss=0,
        scale_adversarial_loss=eps,
        max_epochs=2000,
        lr=learning_rate,
    )
    return model.module, {"time": time.time() - start_time, "history": model.history}


def infer_from_X(X, module, dY):
    fake_acc = np.empty((X.shape[0], dY))
    fake_acc[:] = np.nan
    X = torch.from_numpy(np.concatenate([X, fake_acc], axis=1)).float()
    print("infer from X:")
    print(X.shape)
    print(module.n_input_genes, module.n_input_regions)

    inference_output = module.inference(
        X,
        y=torch.zeros((X.shape[0], 1), requires_grad=False),
        batch_index=torch.zeros((X.shape[0], 1)),
        cont_covs=None,
        cat_covs=None,
        label=None,
        cell_idx=torch.range(0, X.shape[0] - 1).unsqueeze(-1),
    )
    return inference_output


def infer_from_Y(Y, module, dX):
    fake_gexp = np.empty((Y.shape[0], dX))
    fake_gexp[:] = np.nan
    X = torch.from_numpy(np.concatenate([fake_gexp, Y], axis=1)).float()
    print("infer from Y:")
    print(f"input:{X.shape}={fake_gexp.shape} + {Y.shape}")
    print(module.n_input_genes, module.n_input_regions)
    inference_output = module.inference(
        X,
        y=torch.zeros((X.shape[0], 1), requires_grad=False),
        batch_index=torch.zeros((X.shape[0], 1)),
        cont_covs=None,
        cat_covs=None,
        label=None,
        cell_idx=torch.range(0, X.shape[0] - 1).unsqueeze(-1),
    )
    return inference_output


def infer_from_Xs(X_dict, model, dY):
    res = {}
    for k, X in X_dict.items():
        res[k] = infer_from_X(X, model, dY)["qz_m"].detach().cpu().numpy()
    return res


def infer_from_Ys(Y_dict, model, dX):
    res = {}
    for k, Y in Y_dict.items():
        res[k] = infer_from_Y(Y, model, dX)["qz_m"].detach().cpu().numpy()
    return res


def predict_from_model(X, module, dY):
    # model = model.to("cpu")
    inference_output = infer_from_X(X, module, dY)
    generative_output = module.generative(
        inference_output["z"],
        inference_output["qz_m"],
        batch_index=torch.ones((X.shape[0], 1)),
        libsize_expr=inference_output["libsize_expr"],
        libsize_acc=inference_output["libsize_acc"],
        use_z_mean=True,
    )
    return generative_output["p"].detach().numpy()
