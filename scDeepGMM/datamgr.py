import torch
import numpy as np
import scanpy as sc
import tables
import numpy
import scipy
import pandas as pd
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#########################################################

class DataWithClusters():
    def __init__(self,
                 batch_size,
                 file_path = "./data/with_cluster/10X_PBMC_select_2100.h5"):
        
        self.raw_data = sc.read_h5ad(file_path)
        self.raw_data.X = self.raw_data.X.astype("float")
        self.raw_data.obs['precise_clusters'] = self.raw_data.uns['Y']
        
        sc.pp.normalize_per_cell(self.raw_data, counts_per_cell_after=1e4)
        sc.pp.log1p(self.raw_data)
        sc.pp.highly_variable_genes(self.raw_data)
        
        self.raw_data.obsm['X'] = self.raw_data.X[:,self.raw_data.var['highly_variable']].astype(np.float32)
        
        self.train_data = self.raw_data
        self.train_dataset = AnnH5ADDataset(self.raw_data)
        
        self.train_loader = DataLoader(
                    dataset=self.train_dataset,
                    batch_size=batch_size, 
                    shuffle=True,
                    drop_last = True
                )
    def getRawData(self, val=False):
        if val:
            assert(self._split == True)
            return self.val_data
        return self.train_data

    def getX(self, val=False):
        if val:
            assert(self._split == True)
            return torch.tensor(self.val_data.obsm['X'])
        return torch.tensor(self.train_data.obsm['X'])

#########################################################

class PBMC_8k_4k():
    def __init__(self, batch_size, data_path = "./data/10X/pbmc4k/filtered_gene_bc_matrices/GRCh38/",
                data_path2 = "./data/10X/pbmc8k/filtered_gene_bc_matrices/GRCh38/"):
        
        self.raw_data = sc.read_10x_mtx(data_path)
        self.raw_data2 = sc.read_10x_mtx(data_path2)
        self.raw_data = self.raw_data.concatenate(self.raw_data2)
        
        self.raw_data.X = self.raw_data.X.asformat("array")

        sc.pp.normalize_per_cell(self.raw_data, counts_per_cell_after=1e4)
        sc.pp.log1p(self.raw_data)
        sc.pp.highly_variable_genes(self.raw_data)
        
        self.raw_data.X = self.raw_data.X.astype("float32")
        self.raw_data.obsm['X'] = self.raw_data.X[:,self.raw_data.var['highly_variable']].astype(np.float32)
        
        # x_vec = self.raw_data.obs['batch'].values.astype("float32")
        # x_counter = 1 - x_vec
        
        # self.raw_data.obsm['X'] = np.column_stack((self.raw_data.obsm['X'], x_vec, x_counter))
        
        self.train_data = self.raw_data
        self.train_dataset = MergeDataset(self.raw_data)
        
        self.train_loader = DataLoader(
                    dataset=self.train_dataset,
                    batch_size=batch_size, 
                    shuffle=True,
                    drop_last = True
                )
        
    def getRawData(self, val=False):
        if val:
            assert(self._split == True)
            return self.val_data
        return self.train_data

    def getX(self, val=False):
        if val:
            assert(self._split == True)
            return torch.tensor(self.val_data.obsm['X'])
        return torch.tensor(self.train_data.obsm['X'])

#########################################################

class Cortex(object):
    def __init__(self, 
                 batch_size, 
                 split=0, 
                 data_path="./scDGM/data/cortex_scAnnData.h5ad"
            ):
        super(Cortex, self).__init__()
        assert(split <= 1)
        self._split = split
        self._batch_size = batch_size
        self.raw_data = sc.read_h5ad(data_path)

        if split > 0:
            print('Split has not been implemented')
            exit(0)
            self.raw_data = self.raw_data[np.random.permutation(len(self.raw_data))]
            self.train_data = self.raw_data[:int(split * len(self.raw_data))]
            self.val_data = self.raw_data[int(split * len(self.raw_data)):]
            self.train_dataset = AnnDataset(self.train_data)
            self.val_dataset = AnnDataset(self.val_data)
            self.train_loader = DataLoader(
                    dataset=self.train_dataset,
                    batch_size=batch_size, 
                    shuffle=True,
                    drop_last = True
                )
            self.valid_loader = DataLoader(
                    dataset=self.val_dataset,
                    batch_size=batch_size, 
                    shuffle=False,
                    drop_last = True
                )
        else:
            self.train_data = self.raw_data
            self.train_dataset = AnnH5ADDataset(self.train_data)
            self.train_loader = DataLoader(
                    dataset=self.train_dataset,
                    batch_size=batch_size, 
                    shuffle=True,
                    pin_memory=True,
                    drop_last = True
                )

    def getRawData(self, val=False):
        if val:
            assert(self._split == True)
            return self.val_data
        return self.train_data

    def getX(self, val=False):
        if val:
            assert(self._split == True)
            return torch.tensor(self.val_data.obsm['X'])
        return torch.tensor(self.train_data.obsm['X'])

#########################################################

class MergeLiuWu():
    def __init__(self,batch_size,file_path = './data/wu_liu_small.h5ad'):
        self.raw_data = sc.read_h5ad(file_path)
        self.raw_data.obsm['X'] = self.raw_data.X
        self.train_data = self.raw_data
        
        self.train_dataset = MergeDataset(self.train_data)
        self.train_loader = DataLoader(
        dataset=self.train_dataset,
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        drop_last = True)

    def getRawData(self, val=False):
        if val:
            assert(self._split == True)
            return self.val_data
        return self.train_data

    def getX(self, val=False):
        if val:
            assert(self._split == True)
            return torch.tensor(self.val_data.obsm['X'])
        return torch.tensor(self.train_data.obsm['X'])

#########################################################

class MergeDataset(Dataset):
    def __init__(self, data):
        self.raw_data = data
        self.raw_data = data
        local_mean = np.log(self.raw_data.obsm['X'].sum(axis=1)).mean()
        local_var = np.log(self.raw_data.obsm['X'].sum(axis=1)).var()
        self.raw_data.obs['local_mean'] = local_mean
        self.raw_data.obs['local_var'] = local_var
        self.X = self.raw_data.obsm['X']
        self.local_mean = self.raw_data.obs['local_mean']
        self.local_var = self.raw_data.obs['local_var']
        self.batch = self.raw_data.obs['batch'].astype('float')
        #self.precise_cluster = self.raw_data.obs['precise_clusters']
    def __getitem__(self, index):
        sample_x = self.X[index]
        sample_local_mean = self.local_mean[index]
        sample_local_var = self.local_var[index]
        sample_batch = self.batch[index]
        

        sample = {'x': sample_x, 'local_mean':sample_local_mean,'local_var':sample_local_var, 'batch': sample_batch}
        return sample

    def __len__(self):
        return len(self.X)

#########################################################

class SplatterData():
    def __init__(self, batch_size,
                 file_path = "./data/splatter_data/simulated_six_group"):
       
        # Reading data
        raw_data = pd.read_csv(os.path.join(file_path,"simulated_six_group.csv"))
        gene_info = pd.read_csv(os.path.join(file_path,"geneinfo_six_group.csv"))
        cell_info = pd.read_csv(os.path.join(file_path,"cellinfo_six_group.csv"))


        cell_info.index = cell_info.Cell.values
        gene_info.index = gene_info.Gene.values

        raw_data.index = cell_info.index
        raw_adata = sc.AnnData(raw_data,var=gene_info,obs=cell_info)

        # sc.pp.normalize_per_cell(raw_adata, counts_per_cell_after=1e4)
        # sc.pp.log1p(raw_adata)

        raw_adata.obsm['X'] = raw_adata.X
        
        local_mean = np.log(raw_adata.X.sum(axis=1)).mean()
        local_var = np.log(raw_adata.X.sum(axis=1)).var()

        raw_adata.obs['local_mean'] = local_mean
        raw_adata.obs['local_var'] = local_var
        raw_adata.obs['precise_clusters'] = raw_adata.obs['Group']
        
        self.raw_data = raw_adata
        self.train_data = self.raw_data
        self.train_dataset = AnnH5ADDataset(self.raw_data)
        self.train_loader = DataLoader(
                    dataset=self.train_dataset,
                    batch_size=batch_size, 
                    shuffle=True,
                    drop_last = True
                )
    def getRawData(self, val=False):
        if val:
            assert(self._split == True)
            return self.val_data
        return self.train_data

    def getX(self, val=False):
        if val:
            assert(self._split == True)
            return torch.tensor(self.val_data.obsm['X'])
        return torch.tensor(self.train_data.obsm['X'])

#########################################################

class AnnH5ADDataset(Dataset):
    def __init__(self, data):
        self.raw_data = data
        local_mean = np.log(self.raw_data.obsm['X'].sum(axis=1)).mean()
        local_var = np.log(self.raw_data.obsm['X'].sum(axis=1)).var()
        self.raw_data.obs['local_mean'] = local_mean
        self.raw_data.obs['local_var'] = local_var
        self.X = self.raw_data.obsm['X']
        self.local_mean = self.raw_data.obs['local_mean']
        self.local_var = self.raw_data.obs['local_var']
        self.precise_cluster = self.raw_data.obs['precise_clusters']
    def __getitem__(self, index):
        sample_x = self.X[index]
        sample_local_mean = self.local_mean[index]
        sample_local_var = self.local_var[index]
        precise_cluster = self.precise_cluster[index]
        sample = {'x': sample_x, 'local_mean':sample_local_mean,'local_var':sample_local_var, 'labels': precise_cluster}
        return sample

    def __len__(self):
        return len(self.X)