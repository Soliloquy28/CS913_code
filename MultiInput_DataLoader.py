import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import numpy as np
import os
import torch
from collections import Counter


def train_val_test_split(base_path, training_ratio=0.6, validation_ratio=0.2, testing_ratio=0.2):
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


def class_counts(filename_list, stage_path):
    all_stage_array = []

    for filename in filename_list:
        try:
            stage_data = np.loadtxt(os.path.join(stage_path, filename), dtype=np.int64).flatten()
        except:
            stage_data = np.loadtxt(os.path.join(stage_path, filename), skiprows=1, dtype=np.int64).flatten()
        all_stage_array.append(stage_data)
    
    all_stage_list = np.concatenate(all_stage_array)

    counter = Counter(all_stage_list[all_stage_list != -1])

    class_counts = sorted(counter.items())

    return class_counts


class MultiInputSleepDataset(Dataset):
    def __init__(self, ppg_file_list, ppg_path, prv_path, ppg_stage_path, prv_stage_path):
        self.ppg_file_list = ppg_file_list
        self.ppg_path = ppg_path
        self.prv_path = prv_path
        self.ppg_stage_path = ppg_stage_path
        self.prv_stage_path = prv_stage_path

    def __len__(self):
        return len(self.ppg_file_list)
    
    def __getitem__(self, idx):
        filename = self.ppg_file_list[idx]
        
        ppg_signal = np.loadtxt(os.path.join(self.ppg_path, filename))
        ppg_signal = torch.tensor(ppg_signal, dtype=torch.float32).unsqueeze(0)
        
        prv_features = np.loadtxt(os.path.join(self.prv_path, filename), skiprows=1)
        prv_features = torch.tensor(prv_features, dtype=torch.float32)
        
        ppg_stage_labels = np.loadtxt(os.path.join(self.ppg_stage_path, filename), dtype=np.int64)
        ppg_stage_labels = torch.tensor(ppg_stage_labels, dtype=torch.long)
        
        prv_stage_labels = np.loadtxt(os.path.join(self.prv_stage_path, filename), skiprows=1, dtype=np.int64)
        prv_stage_labels = torch.tensor(prv_stage_labels, dtype=torch.long)
        
        return ppg_signal, prv_features, ppg_stage_labels, prv_stage_labels

def create_multi_input_dataloader(ppg_file_list, ppg_path, prv_path, ppg_stage_path, prv_stage_path, batch_size=8, shuffle=True):
    dataset = MultiInputSleepDataset(ppg_file_list, ppg_path, prv_path, ppg_stage_path, prv_stage_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


raw_path = '/dcs/large/u2212061/ppg_34_zero/'
prv_path = '/dcs/large/u2212061/PRV_features/Window_21_final/'
raw_stage_path = '/dcs/large/u2212061/stage_30_minus/'
prv_stage_path = '/dcs/large/u2212061/PRV_stage/Window_21_final/'

training_list, validation_list, testing_list = train_val_test_split(raw_path)

multi_train_loader = create_multi_input_dataloader(training_list, raw_path, prv_path, raw_stage_path, prv_stage_path)
multi_val_loader = create_multi_input_dataloader(validation_list, raw_path, prv_path, raw_stage_path, prv_stage_path, shuffle=False)
multi_test_loader = create_multi_input_dataloader(testing_list, raw_path, prv_path, raw_stage_path, prv_stage_path, shuffle=False)

raw_training_class_counts = class_counts(training_list, raw_stage_path)
raw_validation_class_counts = class_counts(validation_list, raw_stage_path)
raw_testing_class_counts = class_counts(testing_list, raw_stage_path)
# print(raw_training_class_counts)
# print(raw_validation_class_counts)
# print(raw_testing_class_counts)

prv_training_class_counts = class_counts(training_list, prv_stage_path)
prv_validation_class_counts = class_counts(validation_list, prv_stage_path)
prv_testing_class_counts = class_counts(testing_list, prv_stage_path)
# print(prv_training_class_counts)
# print(prv_validation_class_counts)
# print(prv_testing_class_counts)