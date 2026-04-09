import torch.nn as nn
import torch.nn.functional as F


class FFNet(nn.Module):
    def __init__(self, in_features, out_features, device, layers=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        self.emb_layer = nn.Linear(self.in_features, layers[0])
        self.out_layer = nn.Linear(layers[-1], self.out_features)
        self.layers = nn.ModuleList()
        for l_in, l_out in zip(layers[:-1], layers[1:]):
            self.layers.append(nn.Linear(l_in, l_out))

        self.to(self.device)

    def forward(self, x):
        x = self.emb_layer(x)
        for l_i in self.layers:
            x = x + F.leaky_relu(l_i(x))
        x = self.out_layer(x)
        return x

    def last_layer(self):
        return self.layers[-1]
