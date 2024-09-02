import torch
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, classification_report, confusion_matrix
import numpy as np
import torch.nn as nn
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from MultiInput_DataLoader import multi_test_loader
from datetime import datetime
from collections import Counter


def multi_testing_epoch(dataloader, model, device):
    model.eval()  
    running_loss = 0.0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad(): 
        for raw_signal, prv_features, raw_stage_labels, prv_stage_labels in tqdm(dataloader):
            raw_signal, prv_features = raw_signal.to(device), prv_features.to(device)
            raw_stage_labels, prv_stage_labels = raw_stage_labels.to(device), prv_stage_labels.to(device)

            raw_output, prv_output = model(raw_signal, prv_features)

            # # mean_output
            # max_output_expanded = max_output.unsqueeze(1).expand(-1, 4, -1)
            # predicted = torch.argmax(max_output_expanded, dim=1)

            ppg_max_prob, _ = torch.max(raw_output, dim=1)
            prv_max_prob, _ = torch.max(prv_output, dim=1)

            # use_ppg = ppg_max_prob > prv_max_prob
            # final_output = torch.where(use_ppg.unsqueeze(1), raw_output, prv_output)
            # predicted = torch.argmax(final_output, dim=1)

            # average_output = 0.8 * raw_output + 0.2 * prv_output
            average_output = (raw_output + prv_output) / 2

            predicted = torch.argmax(average_output, dim=1)

            mask = prv_stage_labels != -1
            valid_predictions = predicted[mask]
            valid_labels = prv_stage_labels[mask]

            all_predictions.extend(valid_predictions.cpu().numpy())
            all_labels.extend(valid_labels.cpu().numpy())
                
        gc.collect()
        torch.cuda.empty_cache()

    accuracy = accuracy_score(all_labels, all_predictions)
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_predictions)

    return accuracy, weighted_f1, kappa, all_predictions, all_labels


def multi_testing_part(model, model_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    torch.set_grad_enabled(True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint = torch.load(f'Multi_{model_name}_best_model.pth') 
    # checkpoint = torch.load('LSTM_best_model_lr0.0025_lr5e-05.pth')
    
    # checkpoint = torch.load(f'best_model_lr{learning_rate}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_accuracy, weighted_f1, kappa, predicted_labels, true_labels = multi_testing_epoch(multi_test_loader, model, device)
    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

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
    plt.title(f"Multi_{model_name}: Confusion Matrix (acc={format(test_accuracy, '.4f')})", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('True Labels', fontsize=16)
    plt.savefig(f'Multi_{model_name}_confusion_matrix_{timestamp}.png')
    plt.close()

    class_report = classification_report(true_labels, predicted_labels)
    print('Classification Report:')
    print(class_report)

    kappa = cohen_kappa_score(true_labels, predicted_labels)
    weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f'Cohen\'s Kappa: {kappa}.')
    print(f'Weighted F1-score: {weighted_f1}.')
    print(f'Accuracy: {str(test_accuracy)}')

    with open(f'Multi_{model_name}_classification_report_{timestamp}.txt', 'w') as f:
        f.write(f'Timestamp: {timestamp}\n')
        f.write(class_report)
        f.write(f'\nCohen\'s Kappa: {str(kappa)}\n')
        f.write(f'Weighted F1-score: {weighted_f1}.')
        f.write(f'\nAccuracy: {str(test_accuracy)}')

    print('Finished writing results into txt file.')
