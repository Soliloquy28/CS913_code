import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PRV_Model import CNNLSTM
from sklearn.metrics import accuracy_score, make_scorer
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from PRV_DataLoader import prv_training_class_counts, prv_training_dataset_dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

X_list = []
Y_list = []

for batch_features, batch_labels in prv_training_dataset_dataloader:
    X_list.append(batch_features.numpy())
    Y_list.append(batch_labels.numpy())

X = np.concatenate(X_list, axis=0)
Y = np.concatenate(Y_list, axis=0)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

class_weights = torch.tensor([
    1 / prv_training_class_counts[0][1],
    1 / prv_training_class_counts[1][1],
    1 / prv_training_class_counts[2][1],
    1 / prv_training_class_counts[3][1]
])

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.weight is not None:
            self.weight = self.weight.to(input.device)
        
        input = input.view(-1, input.size(-1))
        target = target.view(-1)
        
        return nn.functional.cross_entropy(
            input, target, weight=self.weight, ignore_index=self.ignore_index
        )

def initialize_criterion():
    return WeightedCrossEntropyLoss(weight=class_weights, ignore_index=-1)

def sequence_accuracy(y_true, y_pred):
    # print("Debug information:")
    # print(f"y_true shape: {y_true.shape}")
    # print(f"y_pred shape: {y_pred.shape}")
    # print(f"y_pred size: {y_pred.size}")
    
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 4: 
            y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
        else:
            pass
    
    if y_pred.ndim == 3:
        if y_pred.shape[1] == 1:
            y_pred = np.repeat(y_pred, y_true.shape[1], axis=1)
    
    if y_pred.shape[-1] > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mask = y_true_flat != -1
    valid_y_true = y_true_flat[mask]
    valid_y_pred = y_pred_flat[mask]
    
    return accuracy_score(valid_y_true, valid_y_pred)

accuracy_scorer = make_scorer(sequence_accuracy)

net = NeuralNetClassifier(
    module=CNNLSTM,
    criterion=initialize_criterion,
    optimizer=optim.Adam,
    batch_size=8,
    max_epochs=20,
    device=device,
    train_split=None,
    iterator_train__shuffle=False,
    predict_nonlinearity=None,
)

params = {
    'module__input_channels': [8],
    'module__hidden_size': [32, 64, 128],
    'module__num_layers': [1, 2, 3],
    'module__dropout_rate': [0.1, 0.3, 0.5],
    'module__kernel_size': [3, 5, 7],
    'lr': [0.0001, 0.001, 0.01]
}

gs = GridSearchCV(net, params, cv=3, n_jobs=1, scoring=accuracy_scorer, verbose=2)

gs.fit(X, Y)

print("Best parameters:", gs.best_params_)
print("Best cross-validation accuracy score:", gs.best_score_)


    




