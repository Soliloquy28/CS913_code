import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PRV_DataLoader import prv_training_dataset_dataloader, prv_validation_dataset_dataloader, prv_testing_dataset_dataloader, prv_training_class_counts
from collections import Counter
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import f1_score, precision_recall_fscore_support
import json
from datetime import datetime


def prv_train_epoch(dataloader, model, device, optimizer, criterion):
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


def prv_validation(dataloader, model, device, criterion):
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
    print("Total samples:", total)
    print("Correct predictions:", correct)
    # print("Predicted labels:", predicted_labels[:1000])
    print(count)

    epoch_loss = running_loss / total if total > 0 else 0
    epoch_accuracy = correct / total if total > 0 else 0

    weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')

    # precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels, average=None)

    return epoch_loss, epoch_accuracy, predicted_labels, true_labels, weighted_f1
    

def prv_training_part(model, learning_rate, num_epochs, model_name):

    print(f'Learning rate: {learning_rate}.')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    class_weights = torch.tensor([1 / prv_training_class_counts[0][1], 1 / prv_training_class_counts[1][1], 1 / prv_training_class_counts[2][1], 1 / prv_training_class_counts[3][1]]).to(device) 
    train_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1, reduction="none")
    val_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    torch.set_grad_enabled(True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_validation_loss = float('inf')
    best_validation_f1 = 0
    best_model_path = f'PRV_{model_name}_best_model_lr{learning_rate}.pth'
    # last_model_path = f'{model_name}_last_model_lr{learning_rate}.pth'

    patience = 10
    trigger_times = 0
        
    training_loss_list = []
    validation_loss_list = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1} / {num_epochs}')

        train_epoch_loss = prv_train_epoch(prv_training_dataset_dataloader, model, device, optimizer, train_criterion)
        training_loss_list.append(train_epoch_loss)
        print(f'Training Loss: {train_epoch_loss}.')

        validation_epoch_loss, validation_accuracy, _, _, validation_weighted_f1 = prv_validation(prv_validation_dataset_dataloader, model, device, val_criterion)
        validation_loss_list.append(validation_epoch_loss)
        print(f'Validation Loss: {validation_epoch_loss}.')
        print(f'Weighted F1-score: {validation_weighted_f1}')

        if validation_weighted_f1 > best_validation_f1:
            best_validation_f1 = validation_weighted_f1
        
        # Early stop
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
    
    print("PRV Training completed.")

    plt.figure(figsize=(6, 5))
    plt.plot(range(1, len(training_loss_list) + 1), training_loss_list, label='Training Loss')
    plt.plot(range(1, len(validation_loss_list) + 1), validation_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name}: Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'PRV_{model_name}_loss_trend_lr{learning_rate}_{timestamp}.png')
    print(f"Loss trend graph saved as 'PRV_{model_name}_loss_trend_lr{learning_rate}_{timestamp}.png'")

    return prv_train_epoch, prv_validation



# # Training batch: 1 (1414)/
# # Testing batch: 5 (129)/

# # 3:
# # Counter({1: 484779, 0: 262623})
# # Validation Loss: 1.1750418487868421, Validation Accuracy: 0.567558020984691.
# # Saved best model

# # 4:




