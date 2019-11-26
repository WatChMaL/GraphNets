## Reference
# Based on
#   gcn_batch_topk
# With modifications
#   Extra gcnconv layers
#   Extra linear layers


import torch
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool #, TopKPooling

from custom_layers.topk_pool import TopKPooling
#from custom_layers.remove_isolated_nodes import remove_isolated_nodes

class Net(torch.nn.Module):
    def __init__(self, graph_ws=[128,128,128], lin_ws=[]):
        super(Net, self).__init__()

        self.topk = TopKPooling(2, min_score=0.1)
        self.topk.weight.requires_grad = False

        prev_w = 2
        self.convs = []
        self.bns = []
        for i, w in enumerate(graph_ws):
            setattr(self, "conv{}".format(i), GCNConv(prev_w, w, cached=False))
            setattr(self, "bn{}".format(i), BatchNorm1d(w))
            self.convs.append(getattr(self, "conv{}".format(i)))
            self.bns.append(getattr(self, "bn{}".format(i)))
            prev_w = w

        self.lins = []
        lin_ws.append(3) # This is match the output size
        for i, w in enumerate(lin_ws):
            setattr(self, "lin{}".format(i), Linear(prev_w, w))
            self.lins.append(getattr(self, "lin{}".format(i)))
            prev_w = w


    def forward(self, batch):
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch

        x, edge_index, _, batch_index, _, _ = self.topk(x, edge_index, None, batch_index)
        #print(self.topk.weight)
        #print("topk", batch_index.shape)

        ## Removes isolated nodes, but could result in a graph completely disappearing.
        ## Will not be using this
        #edge_index, _, mask = remove_isolated_nodes(edge_index, num_nodes=batch_index.shape[0])
        #x = x[mask]
        #batch_index = batch_index[mask]
        #print("remove isolate", batch_index.shape)

        for bn, conv in zip(self.bns, self.convs):
            x = F.relu(conv(x,edge_index))
            x = bn(x)

        x = global_max_pool(x, batch_index)

        for lin in self.lins:
            x = lin(x)

        return F.log_softmax(x, dim=1)
