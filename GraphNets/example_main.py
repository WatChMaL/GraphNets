import torch
import numpy as np

from models.gcn_kipf import Net
from training_utils.engine_graph import EngineGraph

from config.config_example import config

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
    print(model.conv1.weight)

    # Reset network
    model.conv1.weight = torch.nn.Parameter(torch.Tensor(np.random.rand(2,16)))
    print(model.conv1.weight)

    # Load network
    engine.load_state(osp.join(engine.dirpath, config.model_name + "_latest.pth"))
    print(model.conv1.weight)
    
    # Check load successful
    engine.validate("test")