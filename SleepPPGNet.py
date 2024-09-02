import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

"""
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

        # Residual connection: Residual connections work by adding the output of a previous layer to the input of a later layer. 
        # This means that the later layer does not have to learn the entire function that maps the input to the output, but only the residual or difference between them.
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


class SleepPPGNet(nn.Module):
    def __init__(self):
        super(SleepPPGNet, self).__init__()

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
        # self.dense = nn.Conv1d(1024, 128, kernel_size=1)

        self.tcnblock1 = TemporalConvNet(num_inputs=128, num_channels=128, kernel_size=7, dropout=0.2)
        self.tcnblock2 = TemporalConvNet(num_inputs=128, num_channels=128, kernel_size=7, dropout=0.2)

        self.final_conv = nn.Conv1d(128, 4, 1)


    # Input shape: torch.Size([8, 1, 1228800])
    # After ResConv blocks: torch.Size([8, 256, 4800])
    # After reshaping: torch.Size([8, 1024, 1200])
    # After dense layer: torch.Size([8, 128, 1200])
    # After TCN blocks: torch.Size([8, 128, 1200])
    # After final conv: torch.Size([8, 4, 1200])

    def forward(self, x):

        # print("Input shape:", x.shape)

        x = self.resconv_blocks(x)
        # print("After ResConv blocks:", x.shape)

        # Window reshape into 1200x128
        batch_size, channels, length = x.shape
        x = x.view(batch_size, channels, 1200, -1)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, -1, 1200)
        # print("After reshaping:", x.shape)  

        x = x.transpose(1, 2)  
        x = self.dense(x)
        x = x.transpose(1, 2)  
        # print("After dense layer:", x.shape)

        # x = x.permute(0, 2, 1)

        x = self.tcnblock1(x)
        x = self.tcnblock2(x)
        # print("After TCN blocks:", x.shape)

        x = self.final_conv(x)
        # print("After final conv:", x.shape)

        x = F.softmax(x, dim=1)

        return x

        # x.argmax(1) # after softmax

