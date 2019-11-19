## Reference
# Towards Sparse Hierarchical Graph Classifiers
# https://arxiv.org/pdf/1811.01287.pdf

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class Net(torch.nn.Module):
    def __init__(self, graph_w=64, lin_w1=32, lin_w2=8):
        super(Net, self).__init__()

        self.conv1 = GraphConv(2, graph_w, cached=False)
        self.pool1 = TopKPooling(graph_w, ratio=0.5)
        self.conv2 = GraphConv(graph_w, graph_w, cached=False)
        self.pool2 = TopKPooling(graph_w, ratio=0.8)
        self.conv3 = GraphConv(graph_w, graph_w, cached=False)
        self.pool3 = TopKPooling(graph_w, ratio=0.8)

        self.lin1 = torch.nn.Linear(graph_w, lin_w1)
        self.lin2 = torch.nn.Linear(lin_w1, lin_w2)
        self.lin3 = torch.nn.Linear(lin_w2, 3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
