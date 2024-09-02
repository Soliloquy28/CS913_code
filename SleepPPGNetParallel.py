import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

"""
InceptionTime code from: https://github.com/TheMrGhostman/InceptionTime-Pytorch
TCN code from: https://github.com/locuslab/TCN
@article{BaiTCN2018,
	author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
	title     = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
	journal   = {arXiv:1803.01271},
	year      = {2018},
}
"""

class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None
        
    def forward(self, x):
        residual = x

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))

        x = self.pool(x)

        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        return x + residual[:, :, :x.shape[2]]
        # return x + residual


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()
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
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super(InceptionModule, self).__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1)
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, out_channels, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        outs = [conv(bottleneck) for conv in self.convs]
        outs.append(self.conv_pool(self.max_pool(x)))
        return torch.cat(outs, dim=1)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.inception1 = InceptionModule(in_channels, out_channels//4)
        self.inception2 = InceptionModule(out_channels, out_channels//4)
        self.inception3 = InceptionModule(out_channels, out_channels//4)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.inception1(x)
        out = self.inception2(out)
        out = self.inception3(out)
        out = out + self.shortcut(x)
        out = self.bn(out)
        return self.relu(out)


class ParallelTCNInception(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(ParallelTCNInception, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        self.inception = InceptionBlock(num_inputs, num_channels[-1])
        self.fusion = nn.Conv1d(num_channels[-1] * 2, num_channels[-1], kernel_size=1)

    def forward(self, x):
        tcn_out = self.tcn(x)
        inception_out = self.inception(x)
        combined = torch.cat([tcn_out, inception_out], dim=1)
        out = self.fusion(combined)
        return out


class SleepPPGNetParallel(nn.Module):
    def __init__(self):
        super(SleepPPGNetParallel, self).__init__()
        self.resconv_blocks = nn.Sequential(
            ResConvBlock(1, 16),
            ResConvBlock(16, 16),
            ResConvBlock(16, 32),
            ResConvBlock(32, 32),
            ResConvBlock(32, 64),
            ResConvBlock(64, 64),
            ResConvBlock(64, 128),
            ResConvBlock(128, 256),
        )

        self.dense = nn.Linear(1024, 128)
        self.parallel_block = ParallelTCNInception(
            num_inputs=128,
            num_channels=[128, 128, 128, 128, 128, 128], 
            kernel_size=7,
            dropout=0.2
        )

        self.final_conv = nn.Conv1d(128, 4, 1)

    def forward(self, x):
        x = self.resconv_blocks(x)
        
        batch_size, channels, length = x.shape
        x = x.view(batch_size, channels, 1200, -1)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, -1, 1200)
        
        x = x.transpose(1, 2)
        x = self.dense(x)
        x = x.transpose(1, 2)
        x = self.parallel_block(x)
        x = self.final_conv(x)
        x = F.softmax(x, dim=1)
        
        return x