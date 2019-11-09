# for commit af82ff8d6577b0e42840d80aee2

from config.easy_dict import EasyDict

config = EasyDict()

config.model_name = "gcn_kipf"

config.data_path = "../metadata/IWCDmPMT_4pi_full_tank_test.h5"
config.train_indices_file = "../metadata/train_indicies.txt"
config.val_indices_file = "../metadata/validation_indicies.txt"
config.test_indices_file = "../metadata/test_indicies.txt"
config.edge_index_pickle = "../metadata/edges_dict.pkl"

config.dump_path = "../dump/gcn"

config.num_data_workers = 0 # Sometime crashes if we do multiprocessing
config.device = 'gpu'
config.gpu_list = [0]

config.batch_size = 32
config.lr=0.01
config.weight_decay=5e-4

config.epochs = 10
config.report_interval = 50
config.num_val_batches  = 32
config.valid_interval   = 500
