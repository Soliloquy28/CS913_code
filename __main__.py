import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# from Database import training_dataset_dataloader, validation_dataset_dataloader, testing_dataset_dataloader
from SleepPPGNet import SleepPPGNet
from SleepPPGNetInception import SleepPPGNetInception
from SleepPPGNetInception2 import SleepPPGNetInception2
from SleepPPGNetParallel import SleepPPGNetParallel
from SleepPPGNetCascaded import SleepPPGNetCascaded
from CNN import CNN
from LSTM import LSTM
from BiLSTM import LSTMBi
from TCN import TCN
from PRV_Model import CNNLSTM
from Raw_Training import training_part
from Raw_Testing import testing_part
from PRV_Training import prv_training_part
from PRV_Testing import prv_testing_part
from collections import Counter
import matplotlib.pyplot as plt
import gc
from MultiInput_Model import MultiInputSleepModel
from MultiInput_Training import multi_training_part
from MultiInput_Testing import multi_testing_part


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# CNN
# model_cnn = CNN().to(device)

# print('CNN Training')
# training_part(
#     model=model_cnn, 
#     learning_rate=0.001, 
#     num_epochs=50,
#     model_name='CNN'
# )

# print('CNN Testing')
# testing_part(
#     model=model_cnn, 
#     learning_rate=0.001, 
#     model_name='CNN'
# )


# SleepPPG-Net
# model_sleepppgnet = SleepPPGNet().to(device)

# print('SleepPPG-Net Training')
# training_part(
#     model=model_sleepppgnet, 
#     learning_rate=0.0005, 
#     num_epochs=150,
#     model_name='SleepPPG-Net'
# )

# print('SleepPPG-Net Testing')
# testing_part(
#     model=model_sleepppgnet, 
#     learning_rate=0.0005, 
#     model_name='SleepPPG-Net'
# )


# SleepPPG-Net-Inception
# model_sleepppgnetinception = SleepPPGNetInception().to(device)

# print('SleepPPG-Net-Inception Training')
# training_part(
#     model=model_sleepppgnetinception, 
#     learning_rate=0.0005, 
#     num_epochs=150,
#     model_name='SleepPPG-Net-Inception'
# )

# print('SleepPPG-Net-Inception Testing')
# testing_part(
#     model=model_sleepppgnetinception, 
#     learning_rate=0.0005, 
#     model_name='SleepPPG-Net-Inception'
# )

# SleepPPG-Net-Inception-2
# model_sleepppgnetinception2 = SleepPPGNetInception2().to(device)

# print('SleepPPG-Net-Inception-2 Training')
# training_part(
#     model=model_sleepppgnetinception2, 
#     learning_rate=0.0005, 
#     num_epochs=150,
#     model_name='SleepPPG-Net-Inception-2'
# )

# print('SleepPPG-Net-Inception-2 Testing')
# testing_part(
#     model=model_sleepppgnetinception2, 
#     learning_rate=0.0005, 
#     model_name='SleepPPG-Net-Inception-2'
# )

# # SleepPPG-Net-Parallel
# model_sleepppgnetparallel = SleepPPGNetParallel().to(device)

# print('SleepPPG-Net-Parallel Training')
# training_part(
#     model=model_sleepppgnetparallel, 
#     learning_rate=0.0005, 
#     num_epochs=150,
#     model_name='SleepPPG-Net-Parallel'
# )

# print('SleepPPG-Net-Parallel Testing')
# testing_part(
#     model=model_sleepppgnetparallel, 
#     learning_rate=0.0005, 
#     model_name='SleepPPG-Net-Parallel'
# )

# # SleepPPG-Net-Parallel
# model_sleepppgnetcascaded = SleepPPGNetCascaded().to(device)

# print('SleepPPG-Net-Cascaded Training')
# training_part(
#     model=model_sleepppgnetcascaded, 
#     learning_rate=0.0005, 
#     num_epochs=150,
#     model_name='SleepPPG-Net-Cascaded'
# )

# print('SleepPPG-Net-Cascaded Testing')
# testing_part(
#     model=model_sleepppgnetcascaded, 
#     learning_rate=0.0005, 
#     model_name='SleepPPG-Net-Cascaded'
# )


# # LSTM
# model_lstm = LSTM().to(device)

# print('LSTM Training')
# training_part(
#     model=model_lstm, 
#     learning_rate=0.1, 
#     num_epochs=150,
#     model_name='LSTM'
# )

# print('LSTM Testing')
# testing_part(
#     model=model_lstm, 
#     learning_rate=0.1, 
#     model_name='LSTM'
# )


# Bi-LSTM
# model_lstmbi = LSTMBi().to(device)

# print('Bi-LSTM Training')
# training_part(
#     model=model_lstmbi, 
#     learning_rate=0.1, 
#     num_epochs=150,
#     model_name='Bi-LSTM'
# )

# print('Bi-LSTM Testing')
# testing_part(
#     model=model_lstmbi, 
#     learning_rate=0.1, 
#     model_name='Bi-LSTM'
# )


# # TCN
# input_size = 128
# output_size = 128
# kernel_size = 7
# dropout = 0.2

# model_tcn = TCN(input_size, output_size, kernel_size, dropout).to(device)

# print('TCN Training')
# training_part(
#     model=model_tcn, 
#     learning_rate=0.1, 
#     num_epochs=50,
#     model_name='TCN'
# )

# print('TCN Testing')
# testing_part(
#     model=model_tcn, 
#     learning_rate=0.001, 
#     model_name='TCN'
# )


# CNN-LSTM
# model_cnn_lstm = CNNLSTM().to(device)

# print('PRV CNN-LSTM Training')
# prv_training_part(
#     model=model_cnn_lstm, 
#     learning_rate=0.001, 
#     num_epochs=150,
#     model_name='CNN-LSTM'
# )

# print('PRV CNN Testing')
# prv_testing_part(
#     model=model_cnn_lstm, 
#     learning_rate=0.001, 
#     model_name='CNN-LSTM'
# )


# # MultiInput
# model_multi = MultiInputSleepModel().to(device)

# print('Multi-Input Model Training')
# multi_training_part(
#     model = model_multi,
#     num_epochs = 150, 
#     model_name = 'Multi-Input'
# )

# print('Multi-Input Model Testing')
# multi_testing_part(
#     model = model_multi, 
#     model_name = 'Multi-Input'
# )