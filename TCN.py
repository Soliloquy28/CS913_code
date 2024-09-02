import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations

"""
TCN code from: https://github.com/locuslab/TCN
@article{BaiTCN2018,
	author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
	title     = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
	journal   = {arXiv:1803.01271},
	year      = {2018},
}
"""


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = parametrizations.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = parametrizations.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()


    def init_weights(self):
        
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels=128, kernel_size=7, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = 6
        # num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels
            out_channels = num_channels
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.dense = nn.Linear(1024, 128)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.final_conv = nn.Conv1d(128, 4, 1)


    def forward(self, x):
        # Window reshape into 1200x128
        batch_size, channels, length = x.shape
        x = x.view(batch_size, channels, 1200, -1)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, -1, 1200)
        # print('After reshaping:', x.shape)  

        x = x.transpose(1, 2)  
        x = self.dense(x)
        x = x.transpose(1, 2)
        # print('After dense layer:', x.shape)

        x = self.tcn(x)
        # print('After TCN shape:', x.shape)
        
        x = self.final_conv(x)
        # print('After final conv:', x.shape)
        
        x = F.softmax(x, dim=1)
        
        return x