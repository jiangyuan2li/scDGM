import torch
import numpy as np
import scanpy as sc
import tables
import numpy
import scipy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def DataManager(batch_size, 
                 split=0, 
                 shuffle=True,
                 data='CortexData'):
        return eval(data)(batch_size,split,shuffle)

class PBMC_10K(object):
    def __init__(self,
           batch_size,
           split=0,
           shuffle=True,
           data_path='./scDGM/data/10x_pbmc_pp.sparse.h5'):
        super(PBMC_10K, self).__init__()
        self._split = split
        self._batch_size = batch_size
        with tables.open_file(data_path)as tables_file:
            self.data_dictionary = load(tables_file)
        self.train_data = self.data_dictionary
        self.train_dataset = AnnH5Dataset(self.train_data)
        self.train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=batch_size, 
                shuffle=shuffle,
                drop_last = True
            )

    def getX(self, val=False):
        for item in range(len(self.train_dataset)):
            yield self.train_dataset.__getitem__(item)

    def getShape(self):
        return self.train_dataset.x.shape

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
        sample = {'x': sample_x, 'local_mean':sample_local_mean,'local_var':sample_local_var, 'lables': precise_cluster}
        return sample

    def __len__(self):
        return len(self.X)

class AnnH5Dataset(Dataset):
    def __init__(self, data):
        super(AnnH5Dataset, self).__init__()
        self.x = data['values']
        self.labels = data['labels']
        self.indptr = torch.tensor(data['values'].indptr, dtype=torch.int64)
        self.indices = torch.tensor(data['values'].indices, dtype=torch.int64)
        self.data = torch.tensor(data['values'].data, dtype=torch.float32)

    def __getitem__(self, index):
        obs = torch.zeros((self.x.shape[1],), dtype=torch.float32)
        ind1, ind2 = self.indptr[index], self.indptr[index+1]
        obs[self.indices[ind1:ind2]] = self.data[ind1:ind2]
        return {'x' : obs, 'labels': self.labels[index]}

    def __len__(self):
        return self.x.shape[0]

def load(tables_file, group=None):

        if not group:
            group = tables_file.root

        data_dictionary = {}

        for node in tables_file.iter_nodes(group):
            node_title = node._v_title
            if node == group:
                pass
            elif isinstance(node, tables.Group):
                if node_title.endswith("set"):
                    data_dictionary[node_title] = load(
                        tables_file, group=node)
                elif node_title.endswith("values"):
                    data_dictionary[node_title] = _load_sparse_matrix(
                        tables_file, group=node)
                elif node_title == "split indices":
                    data_dictionary[node_title] = _load_split_indices(
                        tables_file, group=node)
                elif node_title == "feature mapping":
                    data_dictionary[node_title] = _load_feature_mapping(
                        tables_file, group=node)
                else:
                    raise NotImplementedError(
                        "Loading group `{}` not implemented.".format(
                            node_title)
                    )
            elif isinstance(node, tables.Array):
                data_dictionary[node_title] = _load_array_or_other_type(node)
            else:
                raise NotImplementedError(
                    "Loading node `{}` not implemented.".format(node_title)
                )

        return data_dictionary

def _load_sparse_matrix(tables_file, group):

    arrays = {}

    for array in tables_file.iter_nodes(group, "Array"):
        arrays[array.title] = array.read()

    sparse_matrix = scipy.sparse.csr_matrix(
        (arrays["data"], arrays["indices"], arrays["indptr"]),
        shape=arrays["shape"]
    )

    return sparse_matrix

def _load_array_or_other_type(node):

    value = node.read()

    if value.dtype.char == "S":
        decode = numpy.vectorize(lambda s: s.decode("UTF-8"))
        value = decode(value).astype("U")

    elif value.dtype == numpy.uint8:
        value = value.tostring().decode("UTF-8")

        if value == "None":
            value = None

    if node._v_name.endswith("_was_list"):
        value = value.tolist()

    return value