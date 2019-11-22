import torch
import numpy as np

from models.selector import Model
from training_utils.engine_graph import EngineGraph

from config.config_cedar import config

import os.path as osp

if __name__ == '__main__':
    # Initialization
    model = Model(name=config.model_name, **config.model_kwargs)
    engine = EngineGraph(model, config)
    
    # Training
    engine.train()
    
    # Save network
    engine.save_state()
    
    # Validation
    # engine.load_state(osp.join(engine.dirpath, config.model_name + "_best.pth"))
    # engine.validate("test")


