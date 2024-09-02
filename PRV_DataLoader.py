import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import warnings
from collections import Counter


PRV_PPG_DIRECTORY = '/dcs/large/u2212061/PRV_features/Window_21_final/'
PRV_STAGE_DIRECTORY = '/dcs/large/u2212061/PRV_stage/Window_21_final/'


def train_val_test_split(training_ratio=0.6, validation_ratio=0.2, testing_ratio=0.2, base_path=PRV_PPG_DIRECTORY):
    ppg_name_list = []
    for ppg_name in sorted(os.listdir(base_path)):
        ppg_name_list.append(ppg_name)
    
    total_files = len(ppg_name_list)
    num_train = int(total_files * training_ratio)
    num_val = int(total_files * validation_ratio)
    num_test = int(total_files * testing_ratio)

    # Training set
    training_list = ppg_name_list[:num_train]
    # Validation set
    validation_list = ppg_name_list[num_train:num_train + num_val]
    # Testing set
    testing_list = ppg_name_list[num_train + num_val:num_train + num_val + num_test]

    return training_list, validation_list, testing_list


def class_counts(filename_list):
    all_stage_array = []

    for filename in filename_list:
        stage_data = np.loadtxt(os.path.join(PRV_STAGE_DIRECTORY, filename), skiprows=1, dtype=np.int64).flatten()
        all_stage_array.append(stage_data)
    
    all_stage_list = np.concatenate(all_stage_array)

    counter = Counter(all_stage_list[all_stage_list != -1])

    class_counts = sorted(counter.items())

    return class_counts


class PRV_DATASET(Dataset):
    def __init__(self, ppg_file_list, feature_length=1180, num_features=8):
        self.ppg_file_list = ppg_file_list
        self.feature_length = feature_length
        self.num_features = num_features

    def __len__(self):
        return len(self.ppg_file_list)

    def __getitem__(self, idx):
        ppg_sample = self.ppg_file_list[idx]
        ppg_sample_input = np.loadtxt(os.path.join(PRV_PPG_DIRECTORY, ppg_sample), skiprows=1)
        
        if ppg_sample_input.shape != (self.feature_length, self.num_features):
            raise ValueError(f"Expected shape ({self.feature_length}, {self.num_features}), but got {ppg_sample_input.shape}")
        
        stage_sample = ppg_sample 
        stage_sample_input = np.loadtxt(os.path.join(PRV_STAGE_DIRECTORY, stage_sample), skiprows=1)
    
        if stage_sample_input.shape[0] != self.feature_length:
            raise ValueError(f"Expected label length {self.feature_length}, but got {stage_sample_input.shape[0]}")
        
        ppg_sample_input = torch.tensor(ppg_sample_input, dtype=torch.float32)
        stage_sample_input = torch.tensor(stage_sample_input, dtype=torch.long)
        
        return ppg_sample_input, stage_sample_input


prv_ppg_training_list, prv_ppg_validation_list, prv_ppg_testing_list = train_val_test_split()

prv_training_dataset = PRV_DATASET(prv_ppg_training_list)
prv_validation_dataset = PRV_DATASET(prv_ppg_validation_list)
prv_testing_dataset = PRV_DATASET(prv_ppg_testing_list)

prv_training_dataset_length = len(prv_ppg_training_list)
prv_validation_dataset_length = len(prv_ppg_validation_list)
prv_testing_dataset_length = len(prv_ppg_testing_list)

prv_training_dataset_dataloader = DataLoader(dataset=prv_training_dataset, batch_size=8, shuffle=True, drop_last=True)
prv_validation_dataset_dataloader = DataLoader(dataset=prv_validation_dataset, batch_size=8, shuffle=False, drop_last=False)
prv_testing_dataset_dataloader = DataLoader(dataset=prv_testing_dataset, batch_size=8, shuffle=False, drop_last=False)

prv_training_class_counts = class_counts(prv_ppg_training_list)
prv_validation_class_counts = class_counts(prv_ppg_validation_list)
prv_testing_class_counts = class_counts(prv_ppg_testing_list)
