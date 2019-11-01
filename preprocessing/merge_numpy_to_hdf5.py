#!/usr/bin/env python
# coding: utf-8

import os
import h5py
import argparse
import numpy as np

EVENT_SHAPE = (15808, 2)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merges numpy arrays; outputs hdf5 file")
    parser.add_argument("input_file_list",
                        type=str, nargs=1,
                        help="Path to input text file,\
                        each file on a different line.")
    parser.add_argument('output_file', type=str, nargs=1,
                        help="Path to output file.")  
    args = parser.parse_args()
    return args

def count_events(files):
    # Because we want to remove events with 0 hits, 
    # we need to count the events beforehand (to create the h5 file).
    # This function counts and indexes the events with more than 0 hits.
    # Files need to be iterated in the same order to use the indexes.
    num_events = 0
    nonzero_file_events = []
    for file_index, f in enumerate(files):
        data = np.load(f, allow_pickle=True)
        nonzero_file_events.append([])
        hits = data['digi_hit_pmt']
        for i in range(len(hits)):
            if len(hits[i]) != 0:
                nonzero_file_events[file_index].append(i)
                num_events += 1
    return (num_events, nonzero_file_events)


if __name__ == '__main__':

    # -- Parse arguments
    config = parse_args()

    # Read in the input file list
    with open(config.input_file_list[0]) as f:
        files = f.readlines()

    # Remove whitespace 
    files = [x.strip() for x in files] 

    # Check that files were provided
    if len(files) == 0:
        raise ValueError("No files provided!!")
    print("Merging "+str(len(files))+" files")

    # Start merging (define data type)
    num_nonzero_events, nonzero_event_indexes = count_events(files)
    print(num_nonzero_events) 
    dtype_events = np.dtype(np.float32)
    dtype_labels = np.dtype(np.int32)
    dtype_energies = np.dtype(np.float32)
    dtype_positions = np.dtype(np.float32)
    dtype_IDX = np.dtype(np.int32)
    dtype_PATHS = h5py.special_dtype(vlen=str)
    dtype_angles = np.dtype(np.float32)
    
    # Set up h5_file with correct datatype
    h5_file = h5py.File(config.output_file[0], 'w')
    
    dset_event_data = h5_file.create_dataset("event_data",
                                       shape=(num_nonzero_events,)+EVENT_SHAPE,
                                       dtype=dtype_events)
    dset_labels = h5_file.create_dataset("labels",
                                   shape=(num_nonzero_events,),
                                   dtype=dtype_labels)
    dset_energies = h5_file.create_dataset("energies",
                                     shape=(num_nonzero_events, 1),
                                     dtype=dtype_energies)
    dset_positions = h5_file.create_dataset("positions",
                                      shape=(num_nonzero_events, 1, 3),
                                      dtype=dtype_positions)
    dset_IDX = h5_file.create_dataset("event_ids",
                                shape=(num_nonzero_events,),
                                dtype=dtype_IDX)
    dset_PATHS = h5_file.create_dataset("root_files",
                                  shape=(num_nonzero_events,),
                                  dtype=dtype_PATHS)
    dset_angles = h5_file.create_dataset("angles",
                                 shape=(num_nonzero_events, 2),
                                 dtype=dtype_angles)


    # 22 -> gamma, 11 -> electron, 13 -> muon
    # corresponds to labelling used in CNN with only barrel
    #IWCDmPMT_4pi_full_tank_gamma_E0to1000MeV_unif-pos-R371-y521cm_4pi-dir_3000evts_329.npz has an event with pid 11 though....
    #pid_to_label = {22:0, 11:1, 13:2}

    offset = 0
    offset_next = 0
    for file_index, filename in enumerate(files):
        data = np.load(filename, allow_pickle=True)
        nonzero_events_in_file = len(nonzero_event_indexes[file_index])
        x_data = np.zeros((nonzero_events_in_file,)+EVENT_SHAPE, 
                          dtype=dtype_events)
        digi_hit_pmt = data['digi_hit_pmt']
        digi_hit_charge = data['digi_hit_charge']
        digi_hit_time = data['digi_hit_time']
        digi_hit_trigger = data['digi_hit_trigger']
        trigger_time = data['trigger_time']
        delay = 0
        for i in range(len(digi_hit_pmt)):
            first_trigger = np.argmin(trigger_time[i])
            good_hits = np.where(digi_hit_trigger[i]==first_trigger)
            hit_pmts = digi_hit_pmt[i][good_hits]
            if len(hit_pmts) == 0:
                delay += 1
                continue
            charge = digi_hit_charge[i][good_hits]
            time = digi_hit_time[i][good_hits]
            x_data[i-delay, hit_pmts, 0] = charge
            x_data[i-delay, hit_pmts, 1] = time

        event_id = data['event_id']
        root_file = data['root_file']
        pid = data['pid']
        position = data['position']
        direction = data['direction']
        energy = data['energy'] 

        offset_next += nonzero_events_in_file 

        file_indices = nonzero_event_indexes[file_index]

        dset_IDX[offset:offset_next] = event_id[file_indices]
        dset_PATHS[offset:offset_next] = root_file[file_indices]
        dset_energies[offset:offset_next,:] = energy[file_indices].reshape(-1,1)
        dset_positions[offset:offset_next,:,:] = position[file_indices].reshape(-1,1,3)

        labels = np.full(pid.shape[0], -1)
        labels[pid==22] = 0
        labels[pid==11] = 1
        labels[pid==13] = 2
        dset_labels[offset:offset_next] = labels[file_indices]

        direction = direction[file_indices]
        polar = np.arccos(direction[:,1])
        azimuth = np.arctan2(direction[:,2], direction[:,0])
        dset_angles[offset:offset_next,:] = np.hstack((polar.reshape(-1,1),azimuth.reshape(-1,1)))
        dset_event_data[offset:offset_next,:] = x_data

        offset = offset_next
        print("Finished file: {}".format(filename))


    print("Saving")
    h5_file.close()
    print("Finished")
