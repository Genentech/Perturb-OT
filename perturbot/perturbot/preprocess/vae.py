"""Preprocess raw data to learn the latent embedding and projections."""

import os
import anndata as ad
import scanpy as sc
import scvi
import mudata as md

SCVI_LATENT_KEY = "X_scVI"


def train_vae_rna(adata: ad.AnnData, save_dir="./"):
    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy()  # preserve counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # freeze the state in `.raw`
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=1200,
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        batch_key="cell_source",
    )
    model = scvi.model.SCVI(adata, n_latent=50)
    model.train()
    model_dir = os.path.join(save_dir, "scvi_model")
    model.save(model_dir, overwrite=True)

    latent = model.get_latent_representation()
    adata.obsm[SCVI_LATENT_KEY] = latent
    return adata, model


def train_vae_acc(adata: ad.AnnData, save_dir="./"):
    # compute the threshold: 5% of the cells
    min_cells = int(adata.shape[0] * 0.05)
    # in-place filtering of regions
    sc.pp.filter_genes(adata, min_cells=min_cells)
    scvi.model.PEAKVI.setup_anndata(adata)
    model = scvi.model.PEAKVI(adata, n_hidden=50)
    model.train()
    model_dir = os.path.join(save_dir.name, "peakvi_pbmc")
    model.save(model_dir, overwrite=True)
    latent = model.get_latent_representation()
    adata.obsm[SCVI_LATENT_KEY] = latent
    return adata, model


def train_vae_prot(adata: ad.AnnData, save_dir="./"):
    mdata = md.MuData({"rna": adata[:, :0].copy(), "protein": adata})
    scvi.model.TOTALVI.setup_mudata(
        mdata,
        rna_layer=None,
        protein_layer=None,
        modalities={
            "rna_layer": "rna",
            "protein_layer": "protein",
        },
    )
    model = scvi.model.TOTALVI(mdata, n_latent=50)
    model.train()
    adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()
    return adata, model
