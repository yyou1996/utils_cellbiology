import os
import os.path as osp
import sys
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
dir_root = osp.join(current_dir, 'checkpoints_celltyping_models/sctab/')
sys.path.append(dir_root)

from scipy.sparse import csc_matrix
from cellnet.utils.data_loading import streamline_count_matrix
from cellnet.utils.data_loading import dataloader_factory
import torch
import numpy as np
from collections import OrderedDict
import yaml
import pandas as pd
from tqdm import tqdm

# load checkpoint
if torch.cuda.is_available():
    ckpt = torch.load(
        osp.join(dir_root, 'scTab-checkpoints/scTab/run5/val_f1_macro_epoch=41_val_f1_macro=0.847.ckpt'),
    )
else:
    # map to cpu if there is not gpu available
    ckpt = torch.load(
        osp.join(dir_root, 'scTab-checkpoints/scTab/run5/val_f1_macro_epoch=41_val_f1_macro=0.847.ckpt'), 
        map_location=torch.device('cpu')
    )
# extract state_dict of tabnet model from checkpoint
# I can do this as well and just send you the updated checkpoint file - I think this would be the best solution
# I just put this here for completeness
tabnet_weights = OrderedDict()
for name, weight in ckpt['state_dict'].items():
    if 'classifier.' in name:
        tabnet_weights[name.replace('classifier.', '')] = weight
from cellnet.tabnet.tab_network import TabNet
# load in hparams file of model to get model architecture
with open(osp.join(dir_root, 'scTab-checkpoints/scTab/run5/hparams.yaml')) as f:
    model_params = yaml.full_load(f.read())
# initialzie model with hparams from hparams.yaml file
tabnet = TabNet(
    input_dim=model_params['gene_dim'],
    output_dim=model_params['type_dim'],
    n_d=model_params['n_d'],
    n_a=model_params['n_a'],
    n_steps=model_params['n_steps'],
    gamma=model_params['gamma'],
    n_independent=model_params['n_independent'],
    n_shared=model_params['n_shared'],
    epsilon=model_params['epsilon'],
    virtual_batch_size=model_params['virtual_batch_size'],
    momentum=model_params['momentum'],
    mask_type=model_params['mask_type'],
)
# load trained weights
tabnet.load_state_dict(tabnet_weights)
# set model to inference mode
tabnet.eval()

genes_from_model = pd.read_parquet(osp.join(dir_root, 'merlin_cxg_2023_05_15_sf-log1p_minimal/var.parquet'))
# print(genes_from_model)
cell_type_mapping = pd.read_parquet(osp.join(dir_root, 'merlin_cxg_2023_05_15_sf-log1p_minimal/categorical_lookup/cell_type.parquet'))


def cell_typing(counts, adata_var, topk=20):
    # subset gene space only to genes used by the model
    print('number of genes matched', adata_var.index.isin(genes_from_model.feature_id).sum(), 'out of', len(adata_var))
    counts = counts[:, adata_var.index.isin(genes_from_model.feature_id)]
    adata_var = adata_var[adata_var.index.isin(genes_from_model.feature_id)]
    # print(adata_var)
    # pass the count matrix in csc_matrix to make column slicing efficient
    x_streamlined = streamline_count_matrix(
        csc_matrix(counts), 
        adata_var.index,  # change this if gene names are stored in different column
        genes_from_model.feature_id
    )
    # Wrap dataset into pytorch data loader to use for batched inference
    loader = dataloader_factory(x_streamlined, batch_size=2048)
    def sf_log1p_norm(x):
        """Normalize each cell to have 10000 counts and apply log(x+1) transform."""
        counts = torch.sum(x, dim=1, keepdim=True)
        # avoid zero division error
        counts += counts == 0.
        scaling_factor = 10000. / counts
        return torch.log1p(scaling_factor * x)
    probs = []
    with torch.no_grad():
        for batch in tqdm(loader):
            # normalize data
            x_input = sf_log1p_norm(batch[0]['X'])
            logits, _ = tabnet(x_input)
            # preds.append(torch.argmax(logits, dim=1).numpy())
            probs.append(torch.softmax(logits, dim=1).numpy())
    # preds = np.hstack(preds)
    probs = np.vstack(probs)
    idxs_probs_sorted = np.argsort(-probs, axis=1)[:,:topk]
    probs_sorted = np.sort(probs, axis=1)[:, ::-1][:,:topk]

    probs = []
    celltypes = []
    celltype_labels = cell_type_mapping.label.tolist()
    for i in range(idxs_probs_sorted.shape[0]):
        probs.append(probs_sorted[i])
        celltypes.append([celltype_labels[j] for j in idxs_probs_sorted[i]])
    
    return celltypes, probs
