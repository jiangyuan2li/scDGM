import torch
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DataManager:
    def __init__(self, 
                 batch_size, 
                 split=0, 
                 data_path="/data/cortex_scAnnData.h5ad"
            ):
        assert(split <= 1)
        self._split = split
        self._batch_size = batch_size
        self.raw_data = sc.read_h5ad(data_path)

        if split > 0:
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
            self.train_dataset = AnnDataset(self.train_data)
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

    def getLocalMean(self, val=False):
        if val:
            assert(self._split == True)
            return self.val_dataset.local_mean
        return self.train_dataset.local_mean

    def getLocalVar(self, val=False):
        if val:
            assert(self._split == True)
            return self.val_dataset.local_var
        return self.train_dataset.local_var

class AnnDataset(Dataset):
    def __init__(self, data):
        self.raw_data = data
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
