utils_scsimilarity_celltyping.py# credit to Kevin Fleisher
# curl -L -o ./model_v1.1.tar.gz https://zenodo.org/records/10685499/files/model_v1.1.tar.gz?download=1
# tar -xzvf ./model_v1.1.tar.gz

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy as sp
import torch

import os, sys
import pickle

from scimilarity.utils import lognorm_counts, align_dataset
from scimilarity import CellAnnotation
from scimilarity import CellQuery, align_dataset, Interpreter

from tqdm import tqdm
import gc
from typing import Tuple, Union, Dict
import json
import anndata
from scipy.sparse import csr_matrix


# def get_scimilarity_results(
#     # DATA NEEDS TO BE: 
#     # normalized per cell, scaled to 1e4 per cell, and log1p transformed
    
#     # should contain the var and obs metadata to locate dataset and gene names
#     adata: ad.AnnData,  
#     # true or predicted counts from GNN
#     count_matrix_key: str,
#     model_path: str,
# ) -> Dict:

#     '''
#     Function to use the SCimilarity model to output top cell type class labels 
#     and relevant metadata - including nearest neighbors and top 50 predicted cell
#     types per entry.
    
#     Requires input anndata object and a cell x gene matrix which will be supplied
#     as a separate argument called 'count_matrix'. The adata should have
#     gene metadata in the .var attribute.

#     Input Arguments:
#     ----------------

#     - adata: ad.AnnData
        
#         - AnnData object which may contain cell x gene matrix of gene counts.
#         NOTE: counts in cell x gene matrix needs to be normalized per cell,
#         scaled to 1e4, and log1p transformed

#         - Importantly, object should have the gene names in the index of the .var
#         attribute for SCimilarity

#     - count_matrix_key: str

#         - the string name to the key within the adata where the cell x gene matrix
#         of normalized, 1e4 scaled, log1p transformed counts - either 
#         true or predicted counts - exists.
#         - ex: 'X', "obsm['GNN_predictions']"
#         - adata.<count_matrix_key> should be of type
#             Union[sp.sparse._csr.csr_matrix, np.ndarray]

#     Output:
#     -------
    
#     - out_dict: Dict

#         - Dictionary with one key per unique dataset name, and values being
#         the predictions, nn_idxs, nn_dists, and nn_stats reported by
#         the ca.get_predictions_knn() function.
#         - The predictions in the output dictionary with key 
#         output_dict[<dataset_name>]['predictions'] will be per each cell as they 
#         appear in each sliced out sub adata object 
#         (
#         ex: the order of cells as they appear in 
#         adata.obs.loc[adata.obs['dataset_name']==<dataset_name>]
#         where <dataset_name> is the string denoting the specific dataset
#         and 'dataset_name' is the column name in the original adata
#         )
        
    
#     '''

#     print('Loading in the uncompressed SCimilarity model...')


model_path = '../models_cell_typing/scsimilarity/model_v1.1'
def cell_typing(counts, adata_var, knn=50):
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

    use_gpu = True if torch.cuda.is_available() else False
    ca = CellAnnotation(model_path=model_path, use_gpu=use_gpu)
    cq = CellQuery(model_path, use_gpu=use_gpu)
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
