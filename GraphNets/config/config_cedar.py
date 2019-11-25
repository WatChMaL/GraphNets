# for commit e03099cbe54ab13928c1e

from config.easy_dict import EasyDict

config = EasyDict()

config.model_name = "gcn_kipf"
config.model_kwargs = {}

config.data_path = "/fast_scratch/IWCDmPMT_4pi_fulltank_9M_graphnet_trainval.h5"
config.indices_file = "/fast_scratch/IWCDmPMT_4pi_fulltank_9M_graphnet_trainval_idxs.npz"
config.edge_index_pickle = "/project_dir/visualization/edges_dict.pkl"

config.dump_path = "/project_dir/dump/gcn"

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

config.validate_batch_size = 64
config.validate_dump_interval = 2048
