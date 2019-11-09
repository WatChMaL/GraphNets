import torch
import numpy as np

from models.gcn_kipf import Net
from training_utils.engine_graph import EngineGraph

from config.config_cedar import config

import os.path as osp

if __name__ == '__main__':
    # Initialization
    model = Net()
    engine = EngineGraph(model, config)
    
    # Training
    engine.train()
    
    # Validation
    engine.validate("test")

    # Save network
    engine.save_state()

