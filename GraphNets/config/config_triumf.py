# for commit e54b672177b9a9afd8567bb57ce6 

from config.easy_dict import EasyDict

config = EasyDict()

config.model_name = "gcn_kipf"

config.data_path = "/home/jpeng/test_data/IWCDmPMT_4pi_fulltank_test_graphnet.h5"
config.train_indices_file = "/home/jpeng/test_data/IWCDmPMT_4pi_fulltank_test_splits/train.txt"
config.val_indices_file = "/home/jpeng/test_data/IWCDmPMT_4pi_fulltank_test_splits/val.txt"
config.test_indices_file = "/home/jpeng/test_data/IWCDmPMT_4pi_fulltank_test_splits/test.txt"
config.edge_index_pickle = "/home/jpeng/GraphNets/visualization/edges_dict.pkl"

config.dump_path = "/home/jpeng/GraphNets/dump/gcn"

config.num_data_workers = 0 # Sometime crashes if we do multiprocessing
config.device = 'gpu'
config.gpu_list = [0]

config.batch_size = 64
config.validate_batch_size = 64 
config.lr=0.01
config.weight_decay=5e-4

config.epochs = 10
config.report_interval = 50
config.num_val_batches  = 32
config.valid_interval   = 500
