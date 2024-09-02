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


def plot_hypnogram(true_labels, predicted_labels, timestamp, model_name, cohen, f1):

    plt.figure(figsize=(20, 5))
    plt.rcParams.update({'font.size': 18})
    
    total_samples = len(true_labels)
    hours = np.linspace(0, 10, total_samples)  # 10 hours
    
    plt.plot(hours, true_labels, label='True', linewidth=1)
    
    diff_indices = np.where(true_labels != predicted_labels)[0]
    diff_hours = hours[diff_indices]
    diff_predictions = predicted_labels[diff_indices]
    
    # Wrong points
    plt.scatter(diff_hours, diff_predictions, color='red', s=20, alpha=0.7, label='Predicted (errors)')
    plt.yticks([0, 1, 2, 3], ['Wake', 'Light', 'Deep', 'REM'])
    plt.xlabel('Time (hours)')
    plt.ylabel('Sleep Stage')
    plt.title(f'{model_name} Scored Hypnogram (k={cohen.round(2)} , f1={f1.round(2)})')
    # plt.title(f'Ensemble Model Scored Hypnogram (k={cohen.round(2)} , f1={f1.round(2)})')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, 11, 1))
    plt.ylim(-0.5, 3.5)
    plt.savefig(f'Single_{model_name}_hypnogram_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def load_and_preprocess_file(raw_file_path, prv_file_path, model_name):
    if model_name == 'SleepPPG-Net-Inception':
        raw_sample_input = np.loadtxt(raw_file_path)
        raw_sample_input = torch.tensor(raw_sample_input, dtype=torch.float32).unsqueeze(1)
        raw_sample_input = raw_sample_input.permute(1, 0)
    elif model_name == 'CNN-LSTM':
        prv_sample_input = np.loadtxt(prv_file_path, skiprows=1)
        prv_sample_input = torch.tensor(prv_sample_input, dtype=torch.float32)
        prv_sample_input = prv_sample_input
    elif model_name == 'Multi-Input':
        raw_sample_input = np.loadtxt(raw_file_path)
        raw_sample_input = torch.tensor(raw_sample_input, dtype=torch.float32).unsqueeze(1)
        raw_sample_input = raw_sample_input.permute(1, 0)
        prv_sample_input = np.loadtxt(prv_file_path, skiprows=1)
        prv_sample_input = torch.tensor(prv_sample_input, dtype=torch.float32)
        prv_sample_input = prv_sample_input
    
    # return raw_sample_input.unsqueeze(0), prv_sample_input.unsqueeze(0)
    return raw_sample_input.unsqueeze(0)

def multi_predict_sleep_stages(model, raw_data, prv_data, device):
    model.eval()
    with torch.no_grad():
        raw_data = raw_data.to(device)
        prv_data = prv_data.to(device)
        raw_output, prv_output = model(raw_data, prv_data)

        # ppg_max_prob, _ = torch.max(raw_output, dim=1)
        # prv_max_prob, _ = torch.max(prv_output, dim=1)

        # # Maximum
        # use_ppg = ppg_max_prob > prv_max_prob
        # final_output = torch.where(use_ppg.unsqueeze(1), raw_output, prv_output)

        # predicted = torch.argmax(final_output, dim=1)
        # Average fusion
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


def evaluate_predictions(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    report = classification_report(true_labels, predicted_labels)
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')
    accuracy = np.mean(true_labels == predicted_labels)
    
    return cm, cm_percent, report, kappa, weighted_f1, accuracy

def plot_confusion_matrix(cm, cm_percent, model_name, accuracy, timestamp):
    labels = ['Wake', 'Light', 'Deep', 'REM']
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.2f}%)'

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_percent, annot=annot, fmt='', cmap='Blues', annot_kws={'size': 16}, xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"Single_{model_name}: Confusion Matrix (acc={accuracy:.4f})", fontsize=18)
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('True Labels', fontsize=16)
    plt.savefig(f'Single_{model_name}_confusion_matrix_{timestamp}.png')
    plt.close()

def main(raw_file_path, prv_file_path, true_labels_path, model_name, learning_rate):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if model_name == 'SleepPPG-Net-Inception':
        model = SleepPPGNetInception()
    elif model_name == 'CNN-LSTM':
        model = CNNLSTM()
    elif model_name == 'Multi-Input':
        model = MultiInputSleepModel()
    
    if model_name == 'Multi-Input':
        checkpoint = torch.load(f'Multi_{model_name}_best_model.pth') 
    elif model_name == 'CNN-LSTM':
        checkpoint = torch.load(f'PRV_{model_name}_best_model_lr{learning_rate}.pth') 
    elif model_name == 'SleepPPG-Net-Inception':
        checkpoint = torch.load(f'{model_name}_best_model_lr{learning_rate}.pth') 
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    if model_name == 'Multi-Input':
        raw_data, prv_data = load_and_preprocess_file(raw_file_path, prv_file_path, model_name)
        predicted_labels = multi_predict_sleep_stages(model, raw_data, prv_data, device)
    elif model_name == 'CNN-LSTM' or 'SleepPPG-Net-Inception':
        data = load_and_preprocess_file(raw_file_path, prv_file_path, model_name)
        predicted_labels = predict_sleep_stages(model, data, device)

    true_labels = np.loadtxt(true_labels_path, skiprows=1)
    # Cropping and Alignment
    min_length = min(len(predicted_labels), len(true_labels))
    predicted_labels = predicted_labels[:min_length]
    true_labels = true_labels[:min_length]

    cm, cm_percent, report, kappa, weighted_f1, accuracy = evaluate_predictions(true_labels, predicted_labels)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # plot_confusion_matrix(cm, cm_percent, model_name, accuracy, timestamp)

    plot_hypnogram(true_labels, predicted_labels, timestamp, model_name, kappa, weighted_f1)
    print('Finished processing and writing results.')

if __name__ == "__main__":
    raw_file_path = '/dcs/large/u2212061/ppg_34_zero/mesa-sleep-0001.txt'
    prv_file_path = '/dcs/large/u2212061/PRV_features/Window_21_final/mesa-sleep-0001.txt'
    true_labels_path = '/dcs/large/u2212061/stage_30_minus/mesa-sleep-0001.txt'
    # true_labels_path = '/dcs/large/u2212061/PRV_stage/Window_21_final/mesa-sleep-0001.txt'
    model_name = 'SleepPPG-Net-Inception'
    learning_rate = '0.0005'
    main(raw_file_path, prv_file_path, true_labels_path, model_name, learning_rate)