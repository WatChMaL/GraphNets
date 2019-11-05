# Data
import torch
from torch.utils.data.sampler import SubsetRandomSampler

from torch_geometric.data import Data, Dataset, DataLoader

import numpy as np
import h5py
import pickle


class WCH5Dataset(Dataset):
    """
    Dataset storing image-like data from Water Cherenkov detector
    memory-maps the detector data from hdf5 file
    The detector data must be uncompresses and unchunked
    labels are loaded into memory outright
    No other data is currently loaded 
    """

    # Override the default implementation
    def _download(self):
        pass
    
    def _process(self):
        pass
    
    
    def __init__(self, path, train_indices_file, validation_indices_file, test_indices_file, 
                 edge_index_pickle, nodes=15808,
                 transform=None, pre_transform=None, pre_filter=None, 
                 use_node_attr=False, use_edge_attr=False, cleaned=False):

        super(WCH5Dataset, self).__init__("", transform, pre_transform,
                                        pre_filter)
        
        f=h5py.File(path,'r')
        hdf5_event_data = f["event_data"]
        hdf5_labels=f["labels"]

        assert hdf5_event_data.shape[0] == hdf5_labels.shape[0]

        event_data_shape = hdf5_event_data.shape
        event_data_offset = hdf5_event_data.id.get_offset()
        event_data_dtype = hdf5_event_data.dtype

        #this creates a memory map - i.e. events are not loaded in memory here
        #only on get_item
        self.event_data = np.memmap(path, mode='r', shape=event_data_shape, 
                                    offset=event_data_offset, dtype=event_data_dtype)
        
        #this will fit easily in memory even for huge datasets
        self.labels = np.array(hdf5_labels)
        self.nodes = nodes
        self.load_edges(edge_index_pickle)
        
        self.transform=transform
        
        #the section below handles the subset
        #(for reduced dataset training tests)
        #as well as shuffling and train/test/validation splits
            
        self.train_indices = self.load_indicies(train_indices_file)
        self.val_indices = self.load_indicies(validation_indices_file)
        self.test_indices = self.load_indicies(test_indices_file)
    
    def load_indicies(self, indicies_file):
        with open(indicies_file, 'r') as f:
            lines = f.readlines()
        # indicies = [int(l.strip()) for l in lines if not l.isspace()]
        indicies = [int(l.strip()) for l in lines]
        return indicies
    
    def load_edges(self, edge_index_pickle):
        edge_index = torch.zeros([self.nodes, self.nodes], dtype=torch.int64)

        with open(edge_index_pickle, 'rb') as f:
            edges = pickle.load(f)

            for k,vs in edges.items():
                for v in vs:
                    edge_index[k,v] = 1

        self.edge_index=edge_index.to_sparse()._indices()
    
    def get(self, idx):
        x = torch.from_numpy(self.event_data[idx])
        y = torch.tensor([self.labels[idx]], dtype=torch.int64)

        return Data(x=x, y=y, edge_index=self.edge_index)

    def __len__(self):
        return self.labels.shape[0]
    
def get_loaders(path, train_indices_file, val_indices_file, test_indices_file, edges_dict_pickle, batch_size, workers):
    
    dataset = WCH5Dataset(path, train_indices_file, val_indices_file, test_indices_file, edges_dict_pickle)
                          
    train_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.train_indices))

    val_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.val_indices))

    test_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.test_indices))
    
    return train_loader, val_loader, test_loader