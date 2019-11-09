# for commit 32a94221b3fc60815683da9ca2948a40

from config.easy_dict import EasyDict

config = EasyDict()

config.model_name = "gcn_kipf"

config.data_path = "/project/rpp-tanaka-ab/wollip/data/IWCDmPMT_4pi_fulltank_9M_graphnet.h5"
config.train_indices_file = "/project/rpp-tanaka-ab/wollip/data/IWCDmPMT_4pi_fulltank_9M_splits/train.txt"
config.val_indices_file = "/project/rpp-tanaka-ab/wollip/data/IWCDmPMT_4pi_fulltank_9M_splits/val.txt"
config.test_indices_file = "/project/rpp-tanaka-ab/wollip/data/IWCDmPMT_4pi_fulltank_9M_splits/test.txt"
config.edge_index_pickle = "/project/rpp-tanaka-ab/wollip/GraphNets/visualization/edges_dict.pkl"

config.dump_path = "/project/rpp-tanaka-ab/wollip/GraphNets/dump/gcn"

config.num_data_workers = 0 # Sometime crashes if we do multiprocessing
config.device = 'gpu'
config.gpu_list = [0]

config.batch_size = 64
config.lr=0.01
config.weight_decay=5e-4

config.epochs = 10
config.report_interval = 50
config.num_val_batches  = 32
config.valid_interval   = 500
