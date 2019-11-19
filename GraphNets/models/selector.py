
from models.gcn_kipf import Net as gcn_kipf
from models.gcn_batch_reg import Net as gcn_batch_reg
from models.gcn_topk import Net as gcn_topk

def Model(name="gcn_kipf", **kwargs):
    if name == "gcn_kipf":
        return gcn_kipf(**kwargs)
    elif name == "gcn_batch_reg"
        return gcn_batch_reg(**kwargs)
    elif name == "gcn_topk"
        return gcn_topk(**kwargs)
    else:
        print("Model {} not found".format(name))
        return None

