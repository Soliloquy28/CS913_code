import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_features=1024, hidden_dim=128, num_layers=2, num_classes=4):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_features,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=False 
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, channels, length = x.shape

        x = x.view(batch_size, channels, 1200, -1)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, -1, 1200)
      
        x = x.transpose(1, 2) 

        x, _ = self.lstm(x) 
        x = self.fc(x)
        x = x.transpose(1, 2)
        x = F.softmax(x, dim=-1)
        return x