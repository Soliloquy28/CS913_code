import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader, Dataset
from collections import Counter

MESA_PPG_PATH = '/dcs/large/u2212061/ppg_34_zero/'
MESA_STAGE_PATH = '/dcs/large/u2212061/stage_30_minus/'
# MESA_PPG_PATH = '/dcs/large/u2212061/0347/'
# MESA_STAGE_PATH = '/dcs/large/u2212061/0347_labels/'



def train_val_test_split(training_ratio=0.6, validation_ratio=0.2, testing_ratio=0.2, base_path=MESA_PPG_PATH):
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
        stage_data = np.loadtxt(os.path.join(MESA_STAGE_PATH, filename), dtype=np.int64).flatten()
        all_stage_array.append(stage_data)
    
    all_stage_list = np.concatenate(all_stage_array)

    counter = Counter(all_stage_list[all_stage_list != -1])

    class_counts = sorted(counter.items())

    return class_counts


class PPG_DATASET(Dataset):
    def __init__(self, ppg_file_list):
        self.ppg_file_list = ppg_file_list

    def __len__(self):
        return len(self.ppg_file_list)
    
    def __getitem__(self, idx):
        ppg_sample = self.ppg_file_list[idx]
        ppg_sample_input = np.loadtxt(os.path.join(MESA_PPG_PATH, ppg_sample))
        stage_sample = ppg_sample
        stage_sample_input = np.loadtxt(os.path.join(MESA_STAGE_PATH, stage_sample))

        # Pytorch tensor format
        ppg_sample_input = torch.tensor(ppg_sample_input, dtype=torch.float32).unsqueeze(1)
        stage_sample_input = torch.tensor(stage_sample_input, dtype=torch.long)

        ppg_sample_input = ppg_sample_input.permute(1, 0)

        return ppg_sample_input, stage_sample_input


ppg_training_list, ppg_validation_list, ppg_testing_list = train_val_test_split()

training_dataset = PPG_DATASET(ppg_training_list)
validation_dataset = PPG_DATASET(ppg_validation_list)
testing_dataset = PPG_DATASET(ppg_testing_list)

training_dataset_length = len(ppg_training_list)
validation_dataset_length = len(ppg_validation_list)
testing_dataset_length = len(ppg_testing_list)

training_dataset_dataloader = DataLoader(dataset=training_dataset, batch_size=8, shuffle=True, drop_last=True)
validation_dataset_dataloader = DataLoader(dataset=validation_dataset, batch_size=8, shuffle=False, drop_last=False)
testing_dataset_dataloader = DataLoader(dataset=testing_dataset, batch_size=8, shuffle=False, drop_last=False)

training_list, validation_list, testing_list = train_val_test_split()
training_class_counts = class_counts(training_list)
validation_class_counts = class_counts(validation_list)
testing_class_counts = class_counts(testing_list)
print(training_class_counts)
print(validation_class_counts)
print(testing_class_counts)


print('Finished')



