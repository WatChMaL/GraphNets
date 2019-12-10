# for commit 300052df3430228ef5e8bc55e46845d95d5e57f0

from config.easy_dict import EasyDict

config = EasyDict()

config.model_name = "cheby_batch_topk"
config.model_kwargs = {"w1":128,"w2":128,"w3":128,'k':3}

config.data_path = "/data/WatChMaL/data/IWCDmPMT_4pi_fulltank_test/split_h5/IWCDmPMT_4pi_fulltank_test_graphnet_trainval.h5"
config.indices_file = "/data/WatChMaL/data/IWCDmPMT_4pi_fulltank_test/split_h5/IWCDmPMT_4pi_fulltank_test_graphnet_trainval_idxs.npz"
config.edge_index_pickle = "/home/jpeng/GraphNets/visualization/edges_dict.pkl"

config.dump_path = "/home/jpeng/GraphNets/dump/" + config.model_name

config.num_data_workers = 0 # Sometime crashes if we do multiprocessing
config.device = 'gpu'
config.gpu_list = [7]

config.optimizer = "SGD"
config.optimizer_kwargs = {"lr":0.01, "weight_decay":1e-3, "momentum":0.9, "nesterov":True}

config.scheduler_kwargs = {"mode":"min", "min_lr":1e-6, "patience":1, "verbose":True}
config.scheduler_step = 190

config.batch_size = 64
config.epochs = 30

config.report_interval = 50
config.num_val_batches  = 16
config.valid_interval   = 200

config.validate_batch_size = 64
config.validate_dump_interval = 256
