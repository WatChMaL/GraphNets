# for commit d4f2ea65dc1ba8fd09aa09405ea98e1

from config.easy_dict import EasyDict

config = EasyDict()

config.model_name = "gcn_kipf"

## Data paths
config.data_path = "/app/test_data/split_h5/IWCDmPMT_4pi_fulltank_test_graphnet_trainval.h5"
config.train_indices_file = "/app/test_data/split_h5/IWCDmPMT_4pi_fulltank_test_graphnet_splits/train.txt"
config.val_indices_file = "/app/test_data/split_h5/IWCDmPMT_4pi_fulltank_test_graphnet_splits/val.txt"
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
