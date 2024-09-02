import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score, classification_report
from tqdm import tqdm
import warnings
import torch.nn.functional as F

warnings.filterwarnings('always')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=4, sequence_length=1228800):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=63, stride=128, padding=31),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        self.feature_length = sequence_length // (128 * 2 * 2 * 2)
        
        self.classifier = nn.Sequential(
            nn.Conv1d(256, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  

