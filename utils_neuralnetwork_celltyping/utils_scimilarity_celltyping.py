# credit to Kevin Fleisher
# curl -L -o ./model_v1.1.tar.gz https://zenodo.org/records/10685499/files/model_v1.1.tar.gz?download=1
# tar -xzvf ./model_v1.1.tar.gz

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy as sp
import torch
import pickle
from tqdm import tqdm
import gc
from typing import Tuple, Union, Dict
import json
import anndata
from scipy.sparse import csr_matrix

import os, sys
import os.path as osp
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

sys.path.append(osp.join(current_dir, 'checkpoints_celltyping_models/scimilarity/scimilarity/src'))
from scimilarity.utils import lognorm_counts, align_dataset
from scimilarity import CellAnnotation
from scimilarity import CellQuery, align_dataset, Interpreter


def cell_typing(counts, adata_var, knn=50):
    model_path = osp.join(current_dir, 'checkpoints_celltyping_models/scimilarity/model_v1.1')
    use_gpu = True if torch.cuda.is_available() else False
    ca = CellAnnotation(model_path=model_path, use_gpu=use_gpu)
    cq = CellQuery(model_path, use_gpu=use_gpu)

    if not adata_var.index.name == 'gene_names':
        adata_var['gene_ids'] = adata_var.index.values
        adata_var.index = adata_var['gene_names'].values
    adata_var.index = adata_var.index.astype(str)
    if adata_var.index.duplicated().any():
        counts_df = pd.DataFrame(counts, columns=adata_var.index)
        counts_df = counts_df.groupby(counts_df.columns, axis=1).mean()
        counts = counts_df.values
        # adata_var = adata_var.loc[counts_df.columns]
        adata_var = pd.DataFrame([], index=counts_df.columns)

    adata = anndata.AnnData(X=csr_matrix(counts), var=adata_var)

    print('number of overlapping genes', len(np.intersect1d(ca.gene_order, adata.var.index)), 'number of all genes', len(adata.var.index))
    adata = align_dataset(adata, ca.gene_order, gene_overlap_threshold=10)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.obsm["X_scimilarity"] = ca.get_embeddings(adata.X)

    predictions, nn_idxs, nn_dists, nn_stats = ca.get_predictions_knn(
        adata.obsm["X_scimilarity"], k=knn)

    celltypes = []
    probs = []
    for idx in tqdm(range(len(nn_stats))):
        pred = json.loads(nn_stats.iloc[idx].hits)
        celltype = []
        prob = []
        for k, v in pred.items():
            celltype.append(k)
            prob.append(v)
        prob = np.array(prob)
        prob /= prob.sum()
        idxs_probs_sorted = np.argsort(-prob)
        probs.append(prob[idxs_probs_sorted])
        celltypes.append([celltype[i] for i in idxs_probs_sorted])

    gc.collect()

    return celltypes, probs
