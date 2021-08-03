import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.init as init
from torch.autograd import Variable
from utils.tgcn import ConvTemporalGraphical
from utils.graph import Graph
import torch.nn.functional as F

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class Match_Net(nn.Module):
    def __init__(self, args, flag=0, in_channels=1, edge_importance_weighting=True):
        super(Match_Net, self).__init__()
        self.flag = flag
        # load graph
        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=True)
        self.register_buffer('A', A)
        self.args = args
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 128, kernel_size, 2),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 256, kernel_size, 2),
            st_gcn(256, 256, kernel_size, 1),
            st_gcn(256, 256, kernel_size, 1),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, 16, kernel_size=1)

        self.init = None
        if args.train_audio:
            hidden_dim = 100
            device = args.device
            batch_sz = args.batch_size
            # Create the trainable initial state
            h_init = init.constant_(torch.empty(2, batch_sz, hidden_dim, device=device), 0.0)
            c_init = init.constant_(torch.empty(2, batch_sz, hidden_dim, device=device), 0.0)
            h_init = Variable(h_init, requires_grad=True)
            c_init = Variable(c_init, requires_grad=True)
            self.init = (h_init, c_init)

        self.lstm = nn.LSTM(24, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = torch.nn.Linear(hidden_dim, 16)
        
        self.initialize()


    def initialize(self):
        # Initialize LSTM Weights and Biases
        for layer in self.lstm._all_weights:
            for param_name in layer:
                if 'weight' in param_name:
                    weight = getattr(self.lstm, param_name)
                    init.xavier_normal_(weight.data)
                else:
                    bias = getattr(self.lstm, param_name)
                    init.uniform_(bias.data, 0.25, 0.5)

        # Initialize FC
        init.xavier_normal_(self.fc.weight.data)
        init.constant_(self.fc.bias.data, 0)

    def forward(self, inputs):
        if self.args.audio_file:
            output, (h_n, c_n) = self.lstm(inputs, self.init)
            output = output.reshape((output.shape[0], -1))  # flatten before FC
            dped_output = self.dropout(output)
            dped_output = dped_output[:, -100:]
            audio_output = self.fc(dped_output)
            return audio_output
        x_audio = inputs[:, :self.args.len_seg * 24]
        batch = x_audio.shape[0]
        x_audio = x_audio.reshape((batch, self.args.len_seg, 24))
        output, (h_n, c_n) = self.lstm(x_audio, self.init)
        output = output.reshape((output.shape[0], -1))  # flatten before FC
        dped_output = self.dropout(output)
        dped_output = dped_output[:, -100:]
        audio_output = self.fc(dped_output)

        x = inputs[:, self.args.len_seg * 24:]
        batch = x.shape[0]
        x = x.reshape((batch, 1, self.args.len_seg, 3, 23))
        x = x.transpose(3, 4)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        x = self.fcn(x)
        keyps_output = x.view(x.size(0), -1)

        predictions = (audio_output - keyps_output) ** 2
        predictions = torch.sum(predictions, dim=1)

        return predictions, audio_output, keyps_output

class st_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
