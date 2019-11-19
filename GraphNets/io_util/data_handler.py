from torch.utils.data.sampler import SubsetRandomSampler

from torch_geometric.data import DataLoader

import numpy as np

from io_util.dataset import WCH5Dataset

def load_indicies(indicies_file):
    with open(indicies_file, 'r') as f:
        lines = f.readlines()
    # indicies = [int(l.strip()) for l in lines if not l.isspace()]
    indicies = [int(l.strip()) for l in lines]
    return indicies


class WCH5Dataset_trainval(WCH5Dataset):
    """
    Dataset storing image-like data from Water Cherenkov detector
    memory-maps the detector data from hdf5 file
    The detector data must be uncompresses and unchunked
    labels are loaded into memory outright
    No other data is currently loaded
    """

    def __init__(self, path, train_indices_file, val_indices_file,
                 edge_index_pickle, nodes=15808,
                 transform=None, pre_transform=None, pre_filter=None,
                 use_node_attr=False, use_edge_attr=False, cleaned=False):

        super(WCH5Dataset_trainval, self).__init__( path,
                 edge_index_pickle, nodes=nodes,
                 transform=transform, pre_transform=pre_transform, pre_filter=pre_filter,
                 use_node_attr=use_node_attr, use_edge_attr=use_edge_attr, cleaned=cleaned)


        self.train_indices = load_indicies(train_indices_file)
        self.val_indices = load_indicies(val_indices_file)

def get_loaders(path, train_indices_file, val_indices_file, edges_dict_pickle, batch_size, workers):

    dataset = WCH5Dataset_trainval(path, train_indices_file, val_indices_file, edges_dict_pickle)

    train_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.train_indices))

    val_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.val_indices))

    return train_loader, val_loader, dataset
