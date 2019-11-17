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
    parser.add_argument('indices_folder', type=str, help="Path to indices folder")
    args = parser.parse_args()
    return args

def load_indices(indices_file):
    with open(indices_file, 'r') as f:
        lines = f.readlines()
    indices = [int(l.strip()) for l in lines]
    return indices

if __name__ == '__main__':
    # Parse Arguments
    config = parse_args()

    # Load indices to split upon
    test_indices = load_indices(os.path.join(config.indices_folder, "test.txt"))
    train_indices = load_indices(os.path.join(config.indices_folder, "train.txt"))
    val_indices = load_indices(os.path.join(config.indices_folder, "val.txt"))
    
    # Generate names for the new files
    basename, extension = os.path.splitext(os.path.basename(config.h5_file))
    test_filename = basename + "_test" + extension
    train_filename = basename + "_trainval" + extension

    os.makedirs(config.output_folder, exist_ok=True)

    test_filepath = os.path.join(config.output_folder, test_filename)
    train_filepath = os.path.join(config.output_folder, train_filename)
    
    # Read in original file
    with h5py.File(config.h5_file, 'r') as infile:
        keys = list(infile.keys())

        with h5py.File(test_filepath, 'w') as outfile:
            length = len(test_indices)
            for key in keys:
                original_shape = infile[key].shape
                original_dtype = infile[key].dtype
                new_shape = (length, ) + original_shape[1:]

                dataset = outfile.create_dataset(key, shape=new_shape, dtype=original_dtype)

                sorted_indices = sorted(test_indices)
                i = 0
                while i < len(sorted_indices):
                    end = max(len(sorted_indices), i+128)
                    dataset[i:end] = infile[key][sorted_indices[i:end]]
                    i = end

        with h5py.File(train_filepath, 'w') as outfile:
            length = len(train_indices) + len(val_indices)
            for key in keys:
                original_shape = infile[key].shape
                original_dtype = infile[key].dtype
                new_shape = (length, ) + original_shape[1:]

                dataset = outfile.create_dataset(key, shape=new_shape, dtype=original_dtype)

                sorted_indices = sorted(train_indices)
                i = 0
                while i < len(sorted_indices):
                    end = max(len(sorted_indices), i+128)
                    dataset[i:end] = infile[key][sorted_indices[i:end]]
                    i = end


                sorted_indices = sorted(val_indices)
                i = 0
                while i < len(sorted_indices):
                    end = max(len(sorted_indices), i+128)
                    dataset[i+len(train_indices):end+len(train_indices)] = infile[key][sorted_indices[i:end]]
                    i = end   

    # Write new train and val indices for the new file
    splits_dir = os.path.join(config.output_folder, basename + "_splits")
    os.makedirs(splits_dir, exist_ok=True)

    print("Writing new indices to {}".format(splits_dir))
    
    with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
        f.writelines(["{}\n".format(i) for i in range(len(train_indices))])

    with open(os.path.join(splits_dir, 'val.txt'), 'w') as f:
        f.writelines(["{}\n".format(i) for i in range(len(train_indices), len(train_indices) + len(val_indices))])
