import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from MultiInput_Model import MultiInputSleepModel
from MultiInput_DataLoader import raw_training_class_counts, prv_training_class_counts, multi_train_loader, multi_val_loader
from tqdm import tqdm
import gc
from datetime import datetime


def multi_train_epoch(dataloader, model, device, optimizers, criterions):
    model.train()
    running_loss = 0.0
    total = 0

    for raw_signal, prv_features, raw_stage_labels, prv_stage_labels in tqdm(dataloader):
        # print("TTraw_signal shape:", raw_signal.shape)
        # print("TTprv_features shape:", prv_features.shape)
        # print("TTraw_stage_labels shape:", raw_stage_labels.shape)
        # print("TTprv_stage_labels shape:", prv_stage_labels.shape)
        raw_signal, prv_features = raw_signal.to(device), prv_features.to(device)
        raw_stage_labels, prv_stage_labels = raw_stage_labels.to(device), prv_stage_labels.to(device)

        raw_optimizer, prv_optimizer = optimizers
        raw_optimizer.zero_grad()
        prv_optimizer.zero_grad()

        raw_output, prv_output = model(raw_signal, prv_features)

        raw_criterion, prv_criterion = criterions
        raw_loss = raw_criterion(raw_output, prv_stage_labels)
        prv_loss = prv_criterion(prv_output, prv_stage_labels)

        raw_loss = raw_loss.mean() if raw_loss.dim() > 0 else raw_loss
        prv_loss = prv_loss.mean() if prv_loss.dim() > 0 else prv_loss

        total_loss = (raw_loss + prv_loss) / 2

        total_loss.backward()
        raw_optimizer.step()
        prv_optimizer.step()

        mask = prv_stage_labels != -1
        valid_labels = prv_stage_labels[mask]

        total += valid_labels.size(0)
        running_loss += total_loss.item() * valid_labels.size(0)

    gc.collect()
    torch.cuda.empty_cache()

    epoch_loss = running_loss / total

    return epoch_loss


def multi_validation(dataloader, model, device, criterion):
    model.eval()
    running_loss = 0.0
    total = 0

    with torch.no_grad(): 
        for raw_signal, prv_features, raw_stage_labels, prv_stage_labels in tqdm(dataloader):
            # print("raw_signal shape:", raw_signal.shape)
            # print("prv_features shape:", prv_features.shape)
            raw_signal, prv_features = raw_signal.to(device), prv_features.to(device)
            raw_stage_labels, prv_stage_labels = raw_stage_labels.to(device), prv_stage_labels.to(device)

            raw_output, prv_output = model(raw_signal, prv_features)

            raw_loss = criterion(raw_output, prv_stage_labels)
            prv_loss = criterion(prv_output, prv_stage_labels)

            raw_loss = raw_loss.mean() if raw_loss.dim() > 0 else raw_loss
            prv_loss = prv_loss.mean() if prv_loss.dim() > 0 else prv_loss

            total_loss = (raw_loss + prv_loss) / 2

            mask = prv_stage_labels != -1
            valid_labels = prv_stage_labels[mask]

            total += valid_labels.size(0)
            running_loss += total_loss.item() * valid_labels.size(0)

    epoch_loss = running_loss / total
    # accuracy = accuracy_score(all_labels, all_predictions)

    return epoch_loss


def multi_training_part(model, num_epochs, model_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    raw_optimizer = optim.Adam(model.raw_branch.parameters(), lr=0.0005)
    prv_optimizer = optim.Adam(model.prv_branch.parameters(), lr=0.001)

    optimizers = (raw_optimizer, prv_optimizer)

    raw_class_weights = torch.tensor([1 / raw_training_class_counts[0][1], 1 / raw_training_class_counts[1][1], 1 / raw_training_class_counts[2][1], 1 / raw_training_class_counts[3][1]]).to(device) 
    raw_train_criterion = nn.CrossEntropyLoss(weight=raw_class_weights, ignore_index=-1, reduction="none")

    prv_class_weights = torch.tensor([1 / prv_training_class_counts[0][1], 1 / prv_training_class_counts[1][1], 1 / prv_training_class_counts[2][1], 1 / prv_training_class_counts[3][1]]).to(device) 
    prv_train_criterion = nn.CrossEntropyLoss(weight=prv_class_weights, ignore_index=-1, reduction="none")

    criterions = (raw_train_criterion, prv_train_criterion)

    val_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    torch.set_grad_enabled(True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_validation_loss = float('inf') 
    best_model_path = f'Multi_{model_name}_best_model.pth'

    patience = 10
    trigger_times = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1} / {num_epochs}')
        train_epoch_loss = multi_train_epoch(multi_train_loader, model, device, optimizers, criterions)
        print(f'Training Loss: {train_epoch_loss}.')

        validation_epoch_loss = multi_validation(multi_val_loader, model, device, val_criterion)
        print(f'Validation Loss: {validation_epoch_loss}.')
        
        if validation_epoch_loss < best_validation_loss:
            best_validation_loss = validation_epoch_loss
            trigger_times = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'raw_optimizer_state_dict': raw_optimizer.state_dict(),
                'prv_optimizer_state_dict': prv_optimizer.state_dict(),
                'best_validation_loss': best_validation_loss,
                'train_loss': train_epoch_loss,
                'val_loss': validation_epoch_loss
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
