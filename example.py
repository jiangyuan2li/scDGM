import scDGM
import numpy as np
import scanpy as sc
import torch
import matplotlib.pyplot as plt

BATCH_SIZE=16

dataMgr = scDGM.DataManager(
    batch_size = BATCH_SIZE,
    split      = 0,
    data       = 'PBMC'
)

gmvae = scDGM.GMVAE(
    n_input=dataMgr.getX().shape[1], 
    n_hidden=200, 
    latent_size=250, 
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

latent_z = gmvae.get_latent_z(torch.Tensor(dataMgr.getX()).cuda())  # Latent z
latent_y = gmvae.get_latent_y(torch.Tensor(dataMgr.getX()).cuda())  # Latent y
guesses = torch.argmax(latent_y.probs, dim=1)

score = scDGM.NMI(guesses.cpu().detach().numpy(), dataMgr.getRawData().obs['precise_clusters'])
print(score)

labels = np.unique(dataMgr.getRawData().obs['clusters'])

latent_z = np.argmax(latent_z,axis=1)
post_adata = sc.AnnData(X=dataMgr.getX().numpy())
post_adata.obsm["X_dca"] = latent_z.cpu().detach().numpy()
post_adata.obs['cell_type'] = [labels[guess] for guess in guesses]
sc.pp.neighbors(post_adata, use_rep="X_dca", n_neighbors=15)
sc.tl.umap(post_adata, min_dist=0.1)
fig, ax = plt.subplots(figsize=(7, 6))
sc.pl.umap(post_adata, color=["cell_type"], ax=ax)

torch.save(gmvae.state_dict(), 'C:\\Users\\kingr\\OneDrive\\Documents\\Code\\scDGM\\scDGM\\models\\LR_1e4_n_hidden_100_latent_size_100_kl_weight_0.1_warmup_weight_1_NMI_0.69.pt')