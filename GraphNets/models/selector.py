
from models.gcn_kipf import Net as gcn_kipf
from models.gcn_relu import Net as gcn_relu
from models.gcn_batch_reg import Net as gcn_batch_reg
from models.gcn_batch_topk import Net as gcn_batch_topk
from models.gcn_batch_topk_2 import Net as gcn_batch_topk_2
from models.gcn_deeper import Net as gcn_deeper
from models.cheby_batch_topk import Net as cheby_batch_topk
from models.graph_topk import Net as graph_topk

def Model(name="gcn_kipf", **kwargs):
    if name == "gcn_kipf":
        return gcn_kipf(**kwargs)
    elif name == "gcn_relu":
        return gcn_relu(**kwargs)
    elif name == "gcn_batch_reg":
        return gcn_batch_reg(**kwargs)
    elif name == "gcn_batch_topk":
        return gcn_batch_topk(**kwargs)
    elif name == "gcn_batch_topk_2":
        return gcn_batch_topk_2(**kwargs)
    elif name == "gcn_deeper":
        return gcn_deeper(**kwargs)
    elif name == "cheby_batch_topk":
        return cheby_batch_topk(**kwargs)
    elif name == "graph_topk":
        return graph_topk(**kwargs)
    else:
        print("Model {} not found".format(name))
        return None

