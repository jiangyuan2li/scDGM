import scDGM
import numpy as np
import scanpy as sc
import torch
import matplotlib.pyplot as plt

BATCH_SIZE=16

dataMgr = scDGM.DataManager(
    batch_size = BATCH_SIZE,
    split      = 0,
    data       = 'PBMC_10K'
)

gmvae = scDGM.GMVAE(
    n_input=dataMgr.getShape()[1], 
    n_hidden=[100,100], 
    latent_size=50, 
    n_clusters=9,
    kl_weight=1,
    warm_up_weight=1
)


scDGM.train(
    gmvae, 
    batch_size=BATCH_SIZE,
    train_loader=dataMgr.train_loader,
    valid_loader=None,      # Added validation loader (split parameter in datamgr must be > 0)
    warm_up_epoch=40, 
    num_epochs=105,
    weight_decay=1e-6,
    learning_rate=1e-5,
    seed=1,
    verbose=True,                           # Added verbose parameter
    patience=10                             # Added patience parameter
)