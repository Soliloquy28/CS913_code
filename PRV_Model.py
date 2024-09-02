import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    def __init__(self, input_channels=8, hidden_size=32, num_layers=2, num_classes=4, dropout_rate=0.5, kernel_size=5):
        super(CNNLSTM, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_channels)
        x = x.permute(0, 2, 1) 
        
        cnn_out = self.cnn(x) 
        lstm_in = cnn_out.permute(0, 2, 1) 
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.dropout(lstm_out)
        
        out = self.fc(lstm_out)
        out = out.permute(0, 2, 1)
        
        return F.softmax(out, dim=1)
