# for commit 3277f51e257c94e2ce98545bfd5115b29

from config.easy_dict import EasyDict

config = EasyDict()

config.model_name = "gcn_kipf"
config.model_kwargs = {}

config.data_path = "/data/WatChMaL/data/IWCDmPMT_4pi_fulltank_test/split_h5/IWCDmPMT_4pi_fulltank_test_graphnet_trainval.h5"
config.indices_file = "/data/WatChMaL/data/IWCDmPMT_4pi_fulltank_test/split_h5/IWCDmPMT_4pi_fulltank_test_graphnet_trainval_idxs.npz"
config.edge_index_pickle = "/home/dylanlu/GraphNets/visualization/edges_dict.pkl"

config.dump_path = "/home/dylanlu/GraphNets/dump/gcn"

config.num_data_workers = 0 # Sometime crashes if we do multiprocessing
config.device = 'gpu'
config.gpu_list = [0]

config.optimizer = "Adam"
config.optimizer_kwargs = {"lr":0.01, "weight_decay":5e-4}

config.batch_size = 64
config.epochs = 10

config.report_interval = 50
config.num_val_batches  = 32
config.valid_interval   = 500

config.validate_batch_size = 32
config.validate_dump_interval = 256
