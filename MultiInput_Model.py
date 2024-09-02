import torch
import torch.nn as nn
import torch.nn.functional as F
from SleepPPGNetInception import SleepPPGNetInception
from PRV_Model import CNNLSTM


class MultiInputSleepModel(nn.Module):
    def __init__(self):
        super(MultiInputSleepModel, self).__init__()
        self.raw_branch = SleepPPGNetInception()
        self.prv_branch = CNNLSTM()
        
    
    def forward(self, ppg_signal, prv_features):
        ppg_output = self.raw_branch(ppg_signal)  # [batch_size, num_classes, 1200]
        prv_output = self.prv_branch(prv_features)  # [batch_size, num_classes, 1180]
        
        # Truncating first 10 and last 10
        ppg_output = ppg_output[:, :, 10:-10]  # [batch_size, num_classes, 1180]
        
        return ppg_output, prv_output