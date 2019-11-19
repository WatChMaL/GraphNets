from models.gcn_kipf import Net as gcn_kipf

def Model(name="gcn_kipf", **kwargs):
    if name == "gcn_kipf":
        return gcn_kipf(**kwargs)
    else:
        print("Model {} not found".format(name))
        return None

