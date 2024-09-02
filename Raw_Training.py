import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Raw_DataLoader import training_dataset_dataloader, validation_dataset_dataloader, testing_dataset_dataloader, training_class_counts
from SleepPPGNet import SleepPPGNet
from collections import Counter
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import f1_score, precision_recall_fscore_support
import json
from datetime import datetime


def train_epoch(dataloader, model, device, optimizer, criterion):
    model.train()  
    running_loss = 0.0  
    total = 0

    for inputs, labels in tqdm(dataloader):  
        inputs, labels = inputs.to(device), labels.to(device)     

        optimizer.zero_grad()   
        outputs = model(inputs)     
        
        outputs = outputs.permute(0, 2, 1)
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1)).mean()
        
        loss.backward()     
        optimizer.step()    

        mask = labels != -1
        valid_labels = labels[mask]

        total += valid_labels.size(0)
        running_loss += loss.item() * valid_labels.size(0)
    
    gc.collect()
    torch.cuda.empty_cache()

    epoch_loss = running_loss / total if total > 0 else 0
    
    return epoch_loss


def validation(dataloader, model, device, criterion):

    model.eval()   
    running_loss = 0.0
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    
    with torch.no_grad():  

        for inputs, labels in tqdm(dataloader):

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

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
    print("Total samples:", total)
    print("Correct predictions:", correct)
    # print("Predicted labels:", predicted_labels[:1000])
    print(count)

    epoch_loss = running_loss / total if total > 0 else 0
    epoch_accuracy = correct / total if total > 0 else 0

    weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')

    return epoch_loss, epoch_accuracy, predicted_labels, true_labels, weighted_f1


def training_part(model, learning_rate, num_epochs, model_name):
    print(f'Learning rate: {learning_rate}.')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Class 0: 42.12%
    # Class 1: 46.93%
    # Class 2: 6.81%
    # Class 3: 11.88%
    # class_weights = torch.tensor([0.0860, 0.0772, 0.5319, 0.3049]).to(device)  # 0: 946235, 1: 1042059, 2: 148641, 3: 263894
    class_weights = torch.tensor([1 / training_class_counts[0][1], 1 / training_class_counts[1][1], 1 / training_class_counts[2][1], 1 / training_class_counts[3][1]]).to(device) 
    train_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1, reduction="none")
    val_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    torch.set_grad_enabled(True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_validation_loss = float('inf') 
    best_validation_f1 = 0
    best_model_path = f'{model_name}_best_model_lr{learning_rate}.pth'
    # last_model_path = f'{model_name}_last_model_lr{learning_rate}.pth'

    patience = 10
    trigger_times = 0
        
    training_loss_list = []
    validation_loss_list = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1} / {num_epochs}')

        train_epoch_loss = train_epoch(training_dataset_dataloader, model, device, optimizer, train_criterion)
        training_loss_list.append(train_epoch_loss)
        print(f'Training Loss: {train_epoch_loss}.')

        validation_epoch_loss, validation_accuracy, _, _, validation_weighted_f1 = validation(validation_dataset_dataloader, model, device, val_criterion)
        validation_loss_list.append(validation_epoch_loss)
        print(f'Validation Loss: {validation_epoch_loss}.')
        print(f'Weighted F1-score: {validation_weighted_f1}')

        if validation_weighted_f1 > best_validation_f1:
            best_validation_f1 = validation_weighted_f1
        
        if validation_epoch_loss < best_validation_loss:
            best_validation_loss = validation_epoch_loss
            trigger_times = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_validation_loss': best_validation_loss,
                'best_validation_f1': best_validation_f1,
                'train_loss': train_epoch_loss,
                'val_loss': validation_epoch_loss,
                'val_accuracy': validation_accuracy,
                'training_loss_list': training_loss_list,
                'validation_loss_list': validation_loss_list
            }
            torch.save(checkpoint, best_model_path)
            print('--------------------------------------Saved best model-------------------------------------------')
        else:
            # print('-------------------------------------------------------------------------------------------------')
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                break

        torch.cuda.empty_cache()
    

    print("Training completed.")

    plt.figure(figsize=(6, 5))
    plt.plot(range(1, len(training_loss_list) + 1), training_loss_list, label='Training Loss')
    plt.plot(range(1, len(validation_loss_list) + 1), validation_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name}: Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_loss_trend_lr{learning_rate}_{timestamp}.png')
    print(f"Loss trend graph saved as '{model_name}_loss_trend_lr{learning_rate}_{timestamp}.png'")

    return train_epoch, validation




