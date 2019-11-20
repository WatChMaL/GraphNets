#!/usr/bin/evn/ python

import h5py
import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Separate Train and Validation from Test data")
    parser.add_argument("h5_file",
                        type=str,
                        help="Path to h5_file,\
                        must contain 'event_data'")
    parser.add_argument('output_folder', type=str,
                        help="Path to output folder.")
    parser.add_argument('indices_file', type=str, help="Path to indices file")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Parse Arguments
    config = parse_args()

    # Load indices to split upon
    all_indices = np.load(config.indices_file)
    test_indices = all_indices['test_idxs']
    train_indices = all_indices['train_idxs']
    val_indices = all_indices['val_idxs']

    test_set = set(test_indices)
    train_set = set(train_indices)
    val_set = set(val_indices)

    test_length = len(test_indices)
    train_length = len(train_indices) + len(val_indices)

    # Generate names for the new files
    basename, extension = os.path.splitext(os.path.basename(config.h5_file))
    test_filename = basename + "_test" + extension
    train_filename = basename + "_trainval" + extension

    os.makedirs(config.output_folder, exist_ok=True)

    test_filepath = os.path.join(config.output_folder, test_filename)
    train_filepath = os.path.join(config.output_folder, train_filename)

    print("Writing testing data to {}".format(test_filepath))
    print("Writing training and validating data to {}".format(train_filepath))

    # Write new train and val indices for the new file
    splits_file = os.path.join(config.output_folder, basename + "_trainval_idxs")

    print("Writing new indices to {}".format(splits_file))

    new_train_indices = []
    new_val_indices = []

    # Read in original file
    print("Initializating data")
    with h5py.File(config.h5_file, 'r') as infile:
        keys = list(infile.keys())

        # Writing both file at the same time for sequential read
        with h5py.File(test_filepath, 'w') as testfile:
            with h5py.File(train_filepath, 'w') as trainfile:
                for key in keys:
                    if key == "root_files":
                        continue
                    print(key)
                    # Get info for original data
                    original_data = infile[key]
                    original_shape = original_data.shape
                    original_dtype = original_data.dtype

                    zero = np.zeros(original_shape[1:], dtype=original_dtype)

                    # Pre initialize test data to get offset
                    print("\tinitializing test data")
                    test_shape = (test_length,) + original_shape[1:]
                    test_data = testfile.create_dataset(key, shape=test_shape,
                                                        dtype=original_dtype)
                    test_data[:] = zero

                    # Pre initialize train data to get offset
                    print("\tinitializing train data")
                    train_shape = (train_length,) + original_shape[1:]
                    train_data = trainfile.create_dataset(key, shape=train_shape,
                                                        dtype=original_dtype)
                    train_data[:] = zero

    # Read in original file
    print("Begin mem copy")
    with h5py.File(config.h5_file, 'r') as infile:
        keys = list(infile.keys())

        # Writing both file at the same time for sequential read
        with h5py.File(test_filepath, 'r') as testfile:
            with h5py.File(train_filepath, 'r') as trainfile:
                for key in keys:
                    if key == "root_files":
                        continue
                    print(key)
                    # Get info for original data
                    original_data = infile[key]
                    original_shape = original_data.shape
                    original_dtype = original_data.dtype

                    # Get info for test data
                    test_data = testfile[key]
                    test_shape = test_data.shape

                    # Get info for train data
                    train_data = trainfile[key]
                    train_shape = train_data.shape

                    # Get offsets
                    original_offset = original_data.id.get_offset()
                    test_offset = test_data.id.get_offset()
                    train_offset = train_data.id.get_offset()

                    print(original_offset)
                    print(test_offset)
                    print(train_offset)

                    # Setup mem data
                    original_mem_data = np.memmap(config.h5_file, mode='r', shape=original_shape,
                                                    offset=original_offset, dtype=original_dtype)
                    test_mem_data = np.memmap(test_filepath, mode='readwrite', shape=test_shape,
                                                offset=test_offset, dtype=original_dtype)
                    train_mem_data = np.memmap(train_filepath, mode='readwrite', shape=train_shape,
                                                offset=train_offset, dtype=original_dtype)

                    # Copy
                    test_i = 0
                    train_i = 0
                    for i, data in enumerate(original_mem_data):
                        if i in test_set:
                            test_mem_data[test_i] = data
                            test_i += 1
                        else:
                            train_mem_data[train_i] = data
                            train_i += 1

        train_i = 0
        for i in range(infile[keys[0]].shape[0]):
            if i in train_set:
                new_train_indices.append(train_i)
                train_i += 1
            elif i in val_set:
                new_val_indices.append(train_i)
                train_i += 1

    new_train_indices = np.random.permutation(new_train_indices)
    new_val_indices = np.random.permutation(new_val_indices)

    np.savez(splits_file, train_idxs=new_train_indices, val_idxs=new_val_indices)
