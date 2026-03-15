import torch.nn as nn
import torch.nn.functional as F


class AttentionNet(nn.Module):

    def __init__(self, in_features, hidden_size, out_features, n_heads, device, emb_layers=1, mha_layers=1,
                 emb_activation='tanh', att_activation='relu'):
        super().__init__()

        self._input_size = in_features
        self._hidden_size = hidden_size
        self._output_size = out_features
        self._n_heads = n_heads
        self._device = device
        self._emb_activation = emb_activation
        self._emb_activation_cls = nn.LeakyReLU if self._emb_activation == 'relu' else nn.Tanh
        self._att_activation = att_activation
        self._att_activation_fn = F.relu if self._att_activation == 'relu' else F.tanh

        self._layer_norm = nn.LayerNorm(self._input_size)

        e_layers = [nn.Linear(self._input_size, self._hidden_size), nn.LayerNorm(self._hidden_size),
                    self._emb_activation_cls()]
        for _ in range(emb_layers - 1):
            e_layers += [nn.Linear(self._hidden_size, self._hidden_size), nn.LayerNorm(self._hidden_size),
                         self._emb_activation_cls()]
        self._embedder = nn.Sequential(*e_layers)

        self._emb1 = nn.Linear(self._hidden_size, self._hidden_size)
        self._emb2 = nn.Linear(self._hidden_size, self._hidden_size)
        self._emb3 = nn.Linear(self._hidden_size, self._hidden_size)

        self._mha = nn.ModuleList(
            [nn.MultiheadAttention(self._hidden_size, self._n_heads, batch_first=True) for _ in range(mha_layers)])
        self._l_norm = nn.ModuleList([nn.LayerNorm(self._hidden_size) for _ in range(mha_layers)])

        self._lin = nn.Linear(self._hidden_size, self._output_size)

        self.to(self._device)

    def forward(self, inputs):
        ins = self._layer_norm(inputs)
        emb = self._embedder(ins)
        for mha_i, ln_i in zip(self._mha, self._l_norm):
            emb = emb + self._att_activation_fn(
                ln_i(mha_i(emb + self._emb1(emb), emb + self._emb2(emb), emb + self._emb3(emb), need_weights=False)[0]))

        return self._lin(emb)

    def last_layer(self):
        return self._lin
