import torch
import torch.nn as nn
from MLP import MLPClassifier
import lightning.pytorch as pl


class ConvLayer1d(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, kernel, pooling):
        super(ConvLayer1d, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                padding="same",
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=pooling)
        )

    def forward(self, x):
        return self.layers(x)


class FCNEncoder(nn.Module):
    def __init__(self, in_channels, num_features, hidden_layer, dropout, kernel, pooling):
        super(FCNEncoder, self).__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.hidden_layer = hidden_layer
        self.input_layer = ConvLayer1d(in_channels, hidden_layer[0], dropout, kernel, pooling)
        self.output_layer = nn.Sequential(
            ConvLayer1d(hidden_layer[-1], num_features, dropout, kernel, pooling),
            nn.AdaptiveAvgPool1d(1),
        )
        self.layers = nn.Sequential(self.input_layer)
        for i in range(len(hidden_layer) - 1):
            self.layers.append(
                ConvLayer1d(hidden_layer[i], hidden_layer[i + 1], dropout, kernel, pooling)
            )
        self.layers.append(self.output_layer)

    def forward(self, x):
        y = self.layers(x)
        N = y.shape[0]
        return y.view(N, -1)

    def __str__(self):
        return "FCNEncoder(\n\tin_channels={},\n\tout_channels={},\n\thidden_layer={}\n)\n".format(
            self.in_channels, self.num_features, str(self.hidden_layer)
        )


class FCNClassifier(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_features,
                 hidden_convlayer,
                 hidden_fclayer,
                 dropout=0.5,
                 convdropout=0.3,
                 kernel=3,
                 pooling=2,
                 ):
        super(FCNClassifier, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.hidden_convlayer = hidden_convlayer
        self.hidden_fclayer = hidden_fclayer
        self.dropout = dropout
        self.convdropout = convdropout
        self.kernel = kernel
        self.pooling = pooling
        self.feature_encoder = FCNEncoder(
            in_channels=in_channels,
            num_features=num_features,
            hidden_layer=hidden_convlayer,
            dropout=convdropout,
            kernel=kernel,
            pooling=pooling,
        )
        self.classifier = MLPClassifier(
            in_channels=num_features,
            out_channels=out_channels,
            hidden_layer=hidden_fclayer,
            dropout=dropout,
        )

    def forward(self, x):
        x = x.to(torch.float32)
        return self.classifier(self.feature_encoder(x))

    def __str__(self):
        return "FCNClassifier(" \
               "\n\tin_channels={}," \
               "\n\tout_channels={}," \
               "\n\tnum_features={}," \
               "\n\thidden_convlayer={}," \
               "\n\thidden_fclayer={}," \
               "\n\tfcdropout={}," \
               "\n\tconvdropout," \
               "\n\tkernel={}," \
               "\n\tpooling={})".format(
                    self.in_channels,
                    self.out_channels,
                    self.num_features,
                    self.hidden_convlayer,
                    self.hidden_fclayer,
                    self.dropout,
                    self.convdropout,
                    self.kernel,
                    self.pooling
                )

    def __repr__(self):
        return self.__str__()