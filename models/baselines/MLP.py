import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layer, dropout):
        super(MLPClassifier, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_layer = hidden_layer
        input_layer = nn.Linear(in_channels, hidden_layer[0])
        output_layer = nn.Linear(hidden_layer[-1], out_channels)
        self.model = nn.Sequential(input_layer)
        for i in range(len(hidden_layer) - 1):
            self.model.append(nn.Sequential(nn.Linear(hidden_layer[i], hidden_layer[i + 1]),
                                            nn.BatchNorm1d(hidden_layer[i + 1]),
                                            nn.ReLU(),
                                            nn.Dropout(dropout)
                                            ))
        self.model.append(output_layer)

    def forward(self, x):
        x = x.to(torch.float32)
        return self.model(x)

    def __str__(self):
        return "MLPClassifier(\n\tin_channels={},\n\tout_channels={},\n\thidden_layer={}\n)\n".format(
            self.in_channels, self.out_channels, str(self.hidden_layer)
        )

    def __repr__(self):
        return self.__str__()
