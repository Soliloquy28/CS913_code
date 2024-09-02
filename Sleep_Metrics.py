import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, f1_score
from datetime import datetime
from SleepPPGNet import SleepPPGNet
from SleepPPGNetInception import SleepPPGNetInception
from PRV_Model import CNNLSTM
from MultiInput_Model import MultiInputSleepModel
from Raw_DataLoader import ppg_testing_list
from PRV_DataLoader import prv_ppg_testing_list
from tqdm import tqdm
import os
from scipy import stats


def preprocess_sleepppgnet(raw_file_path):
    raw_sample_input = np.loadtxt(raw_file_path)
    raw_sample_input = torch.tensor(raw_sample_input, dtype=torch.float32).unsqueeze(1)
    raw_sample_input = raw_sample_input.permute(1, 0)
    return raw_sample_input.unsqueeze(0)


def preprocess_cnnlstm(prv_file_path):
    prv_sample_input = np.loadtxt(prv_file_path, skiprows=1)
    prv_sample_input = torch.tensor(prv_sample_input, dtype=torch.float32)
    return prv_sample_input.unsqueeze(0)


def preprocess_multiinput(raw_file_path, prv_file_path):
    raw_sample_input = np.loadtxt(raw_file_path)
    raw_sample_input = torch.tensor(raw_sample_input, dtype=torch.float32).unsqueeze(1)
    raw_sample_input = raw_sample_input.permute(1, 0)
    
    prv_sample_input = np.loadtxt(prv_file_path, skiprows=1)
    prv_sample_input = torch.tensor(prv_sample_input, dtype=torch.float32)
    
    return raw_sample_input.unsqueeze(0), prv_sample_input.unsqueeze(0)


def multi_predict_sleep_stages(model, raw_data, prv_data, device):
    model.eval()
    with torch.no_grad():
        raw_data = raw_data.to(device)
        prv_data = prv_data.to(device)
        raw_output, prv_output = model(raw_data, prv_data)
        average_output = (raw_output + prv_output) / 2
        predicted = torch.argmax(average_output, dim=1)
    
    return predicted.squeeze().cpu().numpy()


def predict_sleep_stages(model, data, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        outputs = model(data)
        predicted = outputs.argmax(1)
    return predicted.squeeze().cpu().numpy()


def calculate_sleep_metrics(labels):
    labels = np.array(labels)
    
    tst = np.sum(labels != 0)
    light_sleep = np.sum(labels == 1)
    deep_sleep = np.sum(labels == 2)
    rem_sleep = np.sum(labels == 3)
    wake_time = np.sum(labels == 0)
    
    se = (tst / len(labels)) * 100
    fr_light = (light_sleep / tst) * 100 if tst > 0 else 0
    fr_deep = (deep_sleep / tst) * 100 if tst > 0 else 0
    fr_rem = (rem_sleep / tst) * 100 if tst > 0 else 0
    
    transitions = np.sum(np.diff(labels) < 0)
    
    return {
        'TST': tst,
        'SE': se,
        'FR_Light': fr_light,
        'FR_Deep': fr_deep,
        'FR_REM': fr_rem,
        'Transitions': transitions
    }


def calculate_comparison_metrics(predicted_list, true_list):
    predicted_array = np.array(predicted_list)
    true_array = np.array(true_list)
    
    mse = np.mean((predicted_array - true_array) ** 2)
    
    ss_total = np.sum((true_array - np.mean(true_array)) ** 2)
    ss_residual = np.sum((true_array - predicted_array) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    pearson_corr, _ = stats.pearsonr(true_array, predicted_array)
    
    return mse, r_squared, pearson_corr


def main(raw_folder_path, prv_folder_path, raw_true_labels_folder_path, prv_true_labels_folder_path, model_name, learning_rate):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if model_name == 'SleepPPG-Net':
        model = SleepPPGNet()
    elif model_name ==  'SleepPPG-Net-Inception':
        model = SleepPPGNetInception()
    elif model_name == 'CNN-LSTM':
        model = CNNLSTM()
    elif model_name == 'Multi-Input':
        model = MultiInputSleepModel()
    
    if model_name == 'Multi-Input':
        checkpoint = torch.load(f'Multi_{model_name}_best_model.pth') 
    elif model_name == 'CNN-LSTM':
        checkpoint = torch.load(f'PRV_{model_name}_best_model_lr{learning_rate}.pth') 
    elif model_name == 'SleepPPG-Net' or 'SleepPPG-Net-Inception':
        checkpoint = torch.load(f'{model_name}_best_model_lr{learning_rate}.pth') 
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    predicted_tst_list = []
    predicted_se_list = []
    predicted_light_fraction_list = []
    predicted_deep_fraction_list = []
    predicted_rem_fraction_list = []
    predicted_transitions_list = []
    
    true_tst_list = []
    true_se_list = []
    true_light_fraction_list = []
    true_deep_fraction_list = []
    true_rem_fraction_list = []
    true_transitions_list = []

    # if model_name == 'SleepPPG-Net' or 'SleepPPG-Net-Inception':
    #     for raw_file in tqdm(ppg_testing_list, desc='Processing files'):
    #         raw_file_path = os.path.join(raw_folder_path, raw_file)
    #         data = preprocess_sleepppgnet(raw_file_path)
    #         predicted_labels = predict_sleep_stages(model, data, device)
    #         true_file_path = os.path.join(raw_true_labels_folder_path, raw_file)
    #         true_labels = np.loadtxt(true_file_path)
    #        
    #         min_length = min(len(predicted_labels), len(true_labels))
    #         predicted_labels = predicted_labels[:min_length]
    #         true_labels = true_labels[:min_length]

    #         predicted_metrics = calculate_sleep_metrics(predicted_labels)
    #         # Store the metrics
    #         predicted_tst_list.append(predicted_metrics['TST'])
    #         predicted_se_list.append(predicted_metrics['SE'])
    #         predicted_light_fraction_list.append(predicted_metrics['FR_Light'])
    #         predicted_deep_fraction_list.append(predicted_metrics['FR_Deep'])
    #         predicted_rem_fraction_list.append(predicted_metrics['FR_REM'])
    #         predicted_transitions_list.append(predicted_metrics['Transitions'])

    #         true_metrics = calculate_sleep_metrics(true_labels)
    #         # Store the metrics
    #         true_tst_list.append(true_metrics['TST'])
    #         true_se_list.append(true_metrics['SE'])
    #         true_light_fraction_list.append(true_metrics['FR_Light'])
    #         true_deep_fraction_list.append(true_metrics['FR_Deep'])
    #         true_rem_fraction_list.append(true_metrics['FR_REM'])
    #         true_transitions_list.append(true_metrics['Transitions'])

    
    # if model_name == 'CNN-LSTM':
    #     for prv_file in tqdm(prv_ppg_testing_list, desc='Processing files'):
    #         prv_file_path = os.path.join(prv_folder_path, prv_file)
    #         data = preprocess_cnnlstm(prv_file_path)
    #         predicted_labels = predict_sleep_stages(model, data, device)
    #         true_file_path = os.path.join(prv_true_labels_folder_path, prv_file)
    #         true_labels = np.loadtxt(true_file_path, skiprows=1)
    #         
    #         min_length = min(len(predicted_labels), len(true_labels))
    #         predicted_labels = predicted_labels[:min_length]
    #         true_labels = true_labels[:min_length]

    #         predicted_metrics = calculate_sleep_metrics(predicted_labels)
    #         # Store the metrics
    #         predicted_tst_list.append(predicted_metrics['TST'])
    #         predicted_se_list.append(predicted_metrics['SE'])
    #         predicted_light_fraction_list.append(predicted_metrics['FR_Light'])
    #         predicted_deep_fraction_list.append(predicted_metrics['FR_Deep'])
    #         predicted_rem_fraction_list.append(predicted_metrics['FR_REM'])
    #         predicted_transitions_list.append(predicted_metrics['Transitions'])

    #         true_metrics = calculate_sleep_metrics(true_labels)
    #         # Store the metrics
    #         true_tst_list.append(true_metrics['TST'])
    #         true_se_list.append(true_metrics['SE'])
    #         true_light_fraction_list.append(true_metrics['FR_Light'])
    #         true_deep_fraction_list.append(true_metrics['FR_Deep'])
    #         true_rem_fraction_list.append(true_metrics['FR_REM'])
    #         true_transitions_list.append(true_metrics['Transitions'])
    

    if model_name == 'Multi-Input':
        for raw_file, prv_file in tqdm(zip(ppg_testing_list, prv_ppg_testing_list), desc='Processing files'):
            raw_file_path = os.path.join(raw_folder_path, raw_file)
            prv_file_path = os.path.join(prv_folder_path, prv_file)
            raw_data, prv_data = preprocess_multiinput(raw_file_path, prv_file_path)
            predicted_labels = multi_predict_sleep_stages(model, raw_data, prv_data, device)
            true_file_path = os.path.join(prv_true_labels_folder_path, prv_file)
            true_labels = np.loadtxt(true_file_path, skiprows=1)
            min_length = min(len(predicted_labels), len(true_labels))
            predicted_labels = predicted_labels[:min_length]
            true_labels = true_labels[:min_length]

            predicted_metrics = calculate_sleep_metrics(predicted_labels)
            # Store the metrics
            predicted_tst_list.append(predicted_metrics['TST'])
            predicted_se_list.append(predicted_metrics['SE'])
            predicted_light_fraction_list.append(predicted_metrics['FR_Light'])
            predicted_deep_fraction_list.append(predicted_metrics['FR_Deep'])
            predicted_rem_fraction_list.append(predicted_metrics['FR_REM'])
            predicted_transitions_list.append(predicted_metrics['Transitions'])

            true_metrics = calculate_sleep_metrics(true_labels)
            # Store the metrics
            true_tst_list.append(true_metrics['TST'])
            true_se_list.append(true_metrics['SE'])
            true_light_fraction_list.append(true_metrics['FR_Light'])
            true_deep_fraction_list.append(true_metrics['FR_Deep'])
            true_rem_fraction_list.append(true_metrics['FR_REM'])
            true_transitions_list.append(true_metrics['Transitions'])

    
    # MSE, R-squared, Pearson
    tst_mse, tst_r2, tst_r = calculate_comparison_metrics(predicted_tst_list, true_tst_list)
    se_mse, se_r2, se_r = calculate_comparison_metrics(predicted_se_list, true_se_list)
    light_fraction_mse, light_fraction_r2, light_fraction_r = calculate_comparison_metrics(predicted_light_fraction_list, true_light_fraction_list)
    deep_fraction_mse, deep_fraction_r2, deep_fraction_r = calculate_comparison_metrics(predicted_deep_fraction_list, true_deep_fraction_list)
    rem_fraction_mse, rem_fraction_r2, rem_fraction_r = calculate_comparison_metrics(predicted_rem_fraction_list, true_rem_fraction_list)
    transitions_mse, transitions_r2, transitions_r = calculate_comparison_metrics(predicted_transitions_list, true_transitions_list)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(timestamp)
    print(model_name)
    print("TST - MSE:", tst_mse, "Pearson correlation coefficient:", tst_r, "R-squared:", tst_r2)
    print("SE - MSE:", se_mse, "Pearson correlation coefficient:", se_r, "R-squared:", se_r2)
    print("Light Fraction - MSE:", light_fraction_mse, "Pearson correlation coefficient:", light_fraction_r, "R-squared:", light_fraction_r2)
    print("Deep Fraction - MSE:", deep_fraction_mse, "Pearson correlation coefficient:", deep_fraction_r, "R-squared:", deep_fraction_r2)
    print("REM Fraction - MSE:", rem_fraction_mse, "Pearson correlation coefficient:", rem_fraction_r, "R-squared:", rem_fraction_r2)
    print("Transitions - MSE:", transitions_mse,"Pearson correlation coefficient:", transitions_r, "R-squared:", transitions_r2)
    
    with open(f'{model_name}_sleep_metrics_{timestamp}.txt', 'w') as f:
        f.write(f'Timestamp: {timestamp}\n')
        f.write(f"\nResults for {model_name}:\n")
        f.write("=" * 50 + "\n")
        f.write(f"TST - MSE: {tst_mse:.4f}, Pearson correlation coefficient: {tst_r:.4f}, R-squared: {tst_r2:.4f}\n")
        f.write(f"SE - MSE: {se_mse:.4f}, Pearson correlation coefficient: {se_r:.4f}, R-squared: {se_r2:.4f}\n")
        f.write(f"Light Fraction - MSE: {light_fraction_mse:.4f}, Pearson correlation coefficient: {light_fraction_r:.4f}, R-squared: {light_fraction_r2:.4f}\n")
        f.write(f"Deep Fraction - MSE: {deep_fraction_mse:.4f}, Pearson correlation coefficient: {deep_fraction_r:.4f}, R-squared: {deep_fraction_r2:.4f}\n")
        f.write(f"REM Fraction - MSE: {rem_fraction_mse:.4f}, Pearson correlation coefficient: {rem_fraction_r:.4f}, R-squared: {rem_fraction_r2:.4f}\n")
        f.write(f"Transitions - MSE: {transitions_mse:.4f}, Pearson correlation coefficient: {transitions_r:.4f}, R-squared: {transitions_r2:.4f}\n")
        f.write("\n") 

    print('Finished processing and writing results.')

if __name__ == "__main__":
    raw_folder_path = '/dcs/large/u2212061/ppg_34_zero_testing/'
    prv_folder_path = '/dcs/large/u2212061/PRV_features/Window_21_final_testing/'
    raw_true_labels_folder_path = '/dcs/large/u2212061/stage_30_minus_testing/'
    prv_true_labels_folder_path = '/dcs/large/u2212061/PRV_stage/Window_21_final_testing/'
    model_name = 'Multi-Input'
    learning_rate = '0.001'
    main(raw_folder_path, prv_folder_path, raw_true_labels_folder_path, prv_true_labels_folder_path, model_name, learning_rate)