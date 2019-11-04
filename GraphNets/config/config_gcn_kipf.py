# Port from https://github.com/tkarras/progressive_growing_of_gans/blob/master/config.py
class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

config = EasyDict()

config.model_name = "gcn_kipf"

config.data_path = "/app/IWCDmPMT_4pi_full_tank_test.h5"
config.train_indices_file = "train_indicies.txt"
config.val_indices_file = "validation_indicies.txt"
config.test_indices_file = "test_indicies.txt"
config.edge_index_pickle = "../../visualization/edges_dict.pkl"

config.dump_path = "dump"

config.num_data_workers = 0 # Sometime crashes if we do multiprocessing
config.device = 'gpu'
config.gpu_list = [0]

config.batch_size = 1024
config.lr=0.01
config.weight_decay=5e-4

config.epochs = 10
config.report_interval = 50
config.num_val_batches  = 32
config.valid_interval   = 500
