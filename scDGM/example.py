import scDGM
import numpy as np
import scanpy as sc
import torch
import matplotlib.pyplot as plt

params = {
    'n_hidden'    : [50, 250],			# start, stop
    'latent_size' : [75, 300],			# start, stop
    'kl_weight'   : [0, 1, 0.1],		# start, stop, step(step is ignored in random search)
    'epochs'      : [20, 125]			# start, stop
}

dataMgr = scDGM.DataManager(
    batch_size = 32,
    split      = 0.8,
    data_path  = 'C:\\Users\\kingr\\OneDrive\\Documents\\Code\\scDGM\\scDGM\\data\\cortex_scAnnData.h5ad'
)

score, opt_params = scDGM.gridSearch(
    dataMgr     = dataMgr, 
    test_params = params, 
    batch_size  = 32, 
    iterations  = 10, 
    scoring     = 'ARI'
)