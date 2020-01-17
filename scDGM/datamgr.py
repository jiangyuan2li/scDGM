import torch
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DataManager:
    def __init__(self, 
                 batch_size, 
                 split=False, 
                 data_path="/data/cortex_scAnnData.h5ad"
            ):
        self._split = split
        self._batch_size = batch_size

        if split:
            print("Not implemented yet")
            exit(0)
        else:
            self.train_dataset = AnnDataset(data_path)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           drop_last = True
                                        )

    def getRawData(self, val=False):
        # if val:
        #       assert(self._split == True)
        #       return self.val_dataset.raw
        return self.train_dataset.raw_data

    def getX(self, val=False):
        return torch.tensor(self.train_dataset.X)

    def getLocalMean(self, val=False):
        return self.train_dataset.local_mean

    def getLocalVar(self, val=False):
        return self.train_dataset.local_var

class AnnDataset(Dataset):
    def __init__(self, data_path):
        self.raw_data = sc.read_h5ad(data_path)
        local_mean = np.log(self.raw_data.obsm['X'].sum(axis=1)).mean()
        local_var = np.log(self.raw_data.obsm['X'].sum(axis=1)).var()
        self.raw_data.obs['local_mean'] = local_mean
        self.raw_data.obs['local_var'] = local_var
        self.X = self.raw_data.obsm['X']
        self.local_mean = self.raw_data.obs['local_mean']
        self.local_var = self.raw_data.obs['local_var']
    def __getitem__(self, index):
        sample_x = self.X[index]
        sample_local_mean = self.local_mean[index]
        sample_local_var = self.local_var[index]
        sample = {'x': sample_x, 'local_mean':sample_local_mean,'local_var':sample_local_var}
        return sample

    def __len__(self):
        return len(self.X)
