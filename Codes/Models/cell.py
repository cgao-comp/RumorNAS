import torch
import torch.nn as nn
import torch.nn.functional as F
from Libs.Models.human_designed import GeniePathLayer, GeoLayer
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import (GATConv, GCNConv, GINConv, JumpingKnowledge,
                                SAGEConv)
from torch_geometric.nn.pool import (global_add_pool, global_max_pool,
                                     global_mean_pool)

# Node aggregator
NA_OPS = {
    'sage': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sage'),
    'sage_sum': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sum'),
    'sage_max': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'max'),
    'gcn': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gcn'),
    'gat': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat'),
    'gin': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gin'),
    'gat_sym': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat_sym'),
    'gat_linear': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'linear'),
    'gat_cos': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'cos'),
    'gat_generalized_linear': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'generalized_linear'),
    'geniepath': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'geniepath'),
}

# Layer aggregator
LA_OPS = {
    'l_max': lambda hidden_size, num_layers: LaAggregator('max', hidden_size, num_layers),
    'l_concat': lambda hidden_size, num_layers: LaAggregator('cat', hidden_size, num_layers),
    'l_lstm': lambda hidden_size, num_layers: LaAggregator('lstm', hidden_size, num_layers),
    'l_sum': lambda hidden_size, num_layers: LaAggregator('sum', hidden_size, num_layers),
    'l_att': lambda hidden_size, num_layers: LaAggregator('att', hidden_size, num_layers),
    'l_mean': lambda hidden_size, num_layers: LaAggregator('mean', hidden_size, num_layers)
}

# Skip-connection
SC_OPS = {
    'none': lambda: Zero(),
    'skip': lambda: Identity(),
}

# Pooling
POOL_OPS = {
    'p_max': lambda: PoolingAggregator('max'),
    'p_mean': lambda: PoolingAggregator('mean'),
    'p_add': lambda: PoolingAggregator('add'),
}

# Sequence aggregator
SEQ_OPS = {
    't_rnn': lambda hidden_size, num_layers: SeqAggregator('rnn', hidden_size, num_layers),
    't_gru': lambda hidden_size, num_layers: SeqAggregator('gru', hidden_size, num_layers),
    't_lstm': lambda hidden_size, num_layers: SeqAggregator('lstm', hidden_size, num_layers),
    't_bilstm': lambda hidden_size, num_layers: SeqAggregator('bilstm', hidden_size, num_layers),
    't_att': lambda hidden_size, num_layers: SeqAggregator('att', hidden_size, num_layers),
    't_transformer': lambda hidden_size, num_layers: SeqAggregator('transformer', hidden_size, num_layers),
    't_cnn': lambda hidden_size, num_layers: SeqAggregator('cnn', hidden_size, num_layers),
}


class NaAggregator(nn.Module):

    def __init__(self, in_dim, out_dim, aggregator):
        super(NaAggregator, self).__init__()
        # aggregator, K = agg_str.split('_')
        if aggregator == 'sage':
            self._op = SAGEConv(in_dim, out_dim, normalize=True)
        if aggregator == 'gcn':
            self._op = GCNConv(in_dim, out_dim)
        if aggregator == 'gat':
            heads = 8
            out_dim //= heads
            self._op = GATConv(in_dim, out_dim, heads=heads, dropout=0.5)
        if aggregator == 'gin':
            nn1 = Sequential(Linear(in_dim, out_dim), ReLU(),
                             Linear(out_dim, out_dim))
            self._op = GINConv(nn1)
        if aggregator in ['gat_sym', 'cos', 'linear', 'generalized_linear']:
            heads = 8
            out_dim //= heads
            self._op = GeoLayer(
                in_dim, out_dim, heads=heads, att_type=aggregator, dropout=0.5
            )
        if aggregator in ['sum', 'max']:
            self._op = GeoLayer(in_dim, out_dim, att_type='const',
                                agg_type=aggregator, dropout=0.5)
        if aggregator in ['geniepath']:
            self._op = GeniePathLayer(in_dim, out_dim)

    def forward(self, x, edge_index):
        return self._op(x, edge_index)


class LaAggregator(nn.Module):

    def __init__(self, mode, hidden_size, num_layers=3):
        super(LaAggregator, self).__init__()
        self.mode = mode
        if mode in ['lstm', 'cat', 'max']:
            self.jump = JumpingKnowledge(
                mode, channels=hidden_size, num_layers=num_layers)
        elif mode == 'att':
            self.att = Linear(hidden_size, 1)

        if mode == 'cat':
            self.lin = Linear(hidden_size * num_layers, hidden_size)
        else:
            self.lin = Linear(hidden_size, hidden_size)

    def forward(self, xs):
        if self.mode in ['lstm', 'cat', 'max']:
            output = self.jump(xs)
        elif self.mode == 'sum':
            output = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            output = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'att':
            input = torch.stack(xs, dim=-1).transpose(1, 2)
            weight = self.att(input)
            weight = F.softmax(weight, dim=1)
            output = torch.mul(input, weight).transpose(1, 2).sum(dim=-1)

        return self.lin(F.relu(output))


class PoolingAggregator(nn.Module):

    def __init__(self, mode):
        super(PoolingAggregator, self).__init__()
        self.mode = mode
        if mode == 'add':
            self.readout = global_add_pool
        elif mode == 'mean':
            self.readout = global_mean_pool
        elif mode == 'max':
            self.readout = global_max_pool

    def forward(self, h, batch):
        return self.readout(h, batch)


class SeqAggregator(nn.Module):

    def __init__(self, mode, hidden_size, num_layers=3):
        super(SeqAggregator, self).__init__()
        self.mode = mode
        if mode == 'lstm':
            self.seq_enc = nn.LSTM(
                hidden_size * 2, hidden_size, num_layers, batch_first=True)
        elif mode == 'rnn':
            self.seq_enc = nn.RNN(
                hidden_size * 2, hidden_size, num_layers, batch_first=True)
        elif mode == 'gru':
            self.seq_enc = nn.GRU(
                hidden_size * 2, hidden_size, num_layers, batch_first=True)
        elif mode == 'bilstm':
            self.seq_enc = nn.LSTM(
                hidden_size * 2, hidden_size, num_layers, bidirectional=True, batch_first=True)
        elif mode == 'att':
            self.seq_enc = Linear(hidden_size * 2, 1)
        # elif mode == 'transformer':
        #     heads = 1
        #     self.seq_enc = nn.ModuleList([
        #         TransformerEncoder(TransformerEncoderLayer(hidden_size, heads), num_layers),
        #         nn.Linear(hidden_size*heads, hidden_size),
        #     ])
        elif mode == 'cnn':
            self.seq_enc = nn.Conv1d(hidden_size * 2, hidden_size, 1)

        self.lin = Linear(hidden_size, hidden_size)
        self.lin_cat = Linear(hidden_size * 2, hidden_size)

    def forward(self, xs):
        h = xs
        if self.mode in ['rnn', 'gru', 'lstm', ]:
            input = torch.stack(h, dim=0)
            h, _ = self.seq_enc(input)
            h = h.mean(dim=0).squeeze()
            h = self.lin(F.relu(h))
        elif self.mode == 'bilstm':
            input = torch.stack(h, dim=0)
            h, _ = self.seq_enc(input)
            h = h.mean(dim=0).squeeze()
            h = self.lin_cat(F.relu(h))
        elif self.mode == 'att':
            input = torch.stack(h, dim=-1).transpose(1, 2)
            weight = self.seq_enc(input)
            weight = F.softmax(weight, dim=1)
            h = torch.mul(input, weight).transpose(1, 2).sum(dim=-1)
            h = self.lin_cat(F.relu(h))
        # elif self.mode == 'transformer':
        #     input = torch.stack(h, dim=-1).transpose(0, 2)
        #     for layer in self.seq_enc:
        #         h = layer(input).transpose(0, 2)
        #     h = h.mean(dim=-1).squeeze()
        #     h = self.lin_cat(F.relu(h))
        elif self.mode == 'cnn':
            input = torch.stack(h, dim=0).transpose(1, 2)
            h = self.seq_enc(input).transpose(1, 2)
            h = h.mean(dim=0).squeeze()
            h = self.lin(F.relu(h))
        return h


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)
