# for commit 98f1b36b60e1be2891f34e09

from config.easy_dict import EasyDict

config = EasyDict()

## Model
config.model_name = "gcn_kipf"
config.model_kwargs = {}

## Data paths
config.data_path = "/app/test_data/split_h5/IWCDmPMT_4pi_fulltank_test_graphnet_trainval.h5"
config.indices_file = "/app/test_data/split_h5/IWCDmPMT_4pi_fulltank_test_graphnet_trainval_idxs.npz"
config.edge_index_pickle = "/app/GraphNets/metadata/edges_dict.pkl"

## Log location
config.dump_path = "/app/GraphNets/dump/gcn"

## Computer Parameters
config.num_data_workers = 0 # Sometime crashes if we do multiprocessing
config.device = 'gpu'
config.gpu_list = [0]

## Training parameters
config.batch_size = 32
config.lr=0.01
config.weight_decay=5e-4
config.epochs = 1

## Logging parameters for training
config.report_interval = 10 # 100
config.num_val_batches  = 32
config.valid_interval   = 100 # 10000

## Validating parameters
config.validate_batch_size = 32
config.validate_dump_interval = 256
