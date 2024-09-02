import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from Raw_DataLoader import testing_dataset_dataloader
from SleepPPGNet import SleepPPGNet
# from Training import validation
import numpy as np
import os
from Raw_DataLoader import MESA_PPG_PATH, MESA_STAGE_PATH
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
# from Training import learning_rate
import gc
from collections import Counter
from sklearn.metrics import f1_score, precision_recall_fscore_support
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def validation(dataloader, model, device, criterion):
    model.eval()   
    running_loss = 0.0
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    all_predicted_labels = []
    all_true_labels = []
    
    with torch.no_grad():  

        for inputs, labels in tqdm(dataloader):

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # loss = criterion(outputs, labels)

            outputs = outputs.permute(0, 2, 1) 
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1)).mean()

            mask = labels != -1
            valid_outputs = outputs[mask]
            valid_labels = labels[mask]

            predicted = valid_outputs.argmax(1)  
            correct += predicted.eq(valid_labels).sum().item()    
            total += valid_labels.size(0)     
            running_loss += loss.item() * valid_labels.size(0)
        
            predicted_list = predicted[:].tolist()
            predicted_labels.extend(predicted_list)
            true_labels.extend(valid_labels[:].tolist())
        
        gc.collect()
        torch.cuda.empty_cache()

    count = Counter(predicted_labels)
    print('Total samples:', total)
    print('Correct predictions:', correct)
    # print('Predicted labels:', predicted_labels[:1000])
    print(count)

    epoch_loss = running_loss / total if total > 0 else 0
    epoch_accuracy = correct / total if total > 0 else 0

    weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')

    return epoch_loss, epoch_accuracy, predicted_labels, true_labels, weighted_f1


def testing_part(model, learning_rate, model_name):
    print(f'Learning rate: {learning_rate}.')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    torch.set_grad_enabled(True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    checkpoint = torch.load(f'{model_name}_best_model_lr{learning_rate}.pth')  
    # checkpoint = torch.load('LSTM_best_model_lr0.0025_lr5e-05.pth') 
    
    # checkpoint = torch.load(f'best_model_lr{learning_rate}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_accuracy, predicted_labels, true_labels, weighted_f1 = validation(testing_dataset_dataloader, model, device, criterion)
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    print(f'Predicted labels: {len(predicted_labels)}')
    print(predicted_labels)
    print(f'True labels: {len(true_labels)}')
    print(true_labels)

    cm = confusion_matrix(true_labels, predicted_labels)

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.2f}%)'
            # annot[i, j] = f'{cm_percent[i, j]:.2f}%'

    labels = ['Wake', 'Light', 'Deep', 'REM']

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_percent, annot=annot, fmt='', cmap='Blues', annot_kws={'size': 16}, xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"{model_name}: Confusion Matrix (acc={format(test_accuracy, '.4f')})", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('True Labels', fontsize=16)
    plt.savefig(f'{model_name}_confusion_matrix_lr{learning_rate}_{timestamp}.png')
    plt.close()

    report = classification_report(true_labels, predicted_labels)
    print('Classification Report:')
    print(report)

    kappa = cohen_kappa_score(true_labels, predicted_labels)
    print(f'Cohen\'s Kappa: {kappa}.')
    print(f'Weighted F1-score: {weighted_f1}.')
    print(f'Accuracy: {str(test_accuracy)}')

    with open(f'{model_name}_classification_report_lr{learning_rate}_{timestamp}.txt', 'w') as f:
        f.write(f'Timestamp: {timestamp}\n')
        f.write(report)
        f.write(f'\nCohen\'s Kappa: {str(kappa)}\n')
        f.write(f'Weighted F1-score: {weighted_f1}.')
        f.write(f'\nAccuracy: {str(test_accuracy)}\n')

    print('Finished writing results into txt file.')

