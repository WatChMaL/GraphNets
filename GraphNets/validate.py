import torch
import numpy as np

from models.gcn_kipf import Net
from training_utils.engine_graph import EngineGraph

from config.config_triumf import config

import os.path as osp

if __name__ == '__main__':
    # Initialization
    model = Net()
    engine = EngineGraph(model, config)
   
    # Load Model
    # engine.load_state(osp.join(engine.dirpath, config.model_name + "_best.pth"))
    engine.load_state("/home/jpeng/graphnets/dump/gcn20191109_091304/gcn_kipf_best.pth")

    # Validation
    engine.validate("test")
