import torch
import torch.nn as nn
import torch.nn.functional as F
from Libs.Models.ops import LaOp, NaOp, PoolOp, ScOp, SeqOp
from torch_geometric.nn import LayerNorm


class UniBlock4Tune(nn.Module):

    def __init__(self, mode, genotype, criterion, in_dim, out_dim, hidden_size, num_layers=3, in_dropout=0.5, out_dropout=0.5, act='relu', is_mlp=False, args=None):
        super(UniBlock4Tune, self).__init__()
        assert mode in ['TD', 'BU']
        self.mode = mode
        self.genotype = genotype
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.in_dropout = in_dropout
        self.out_dropout = out_dropout
        self._criterion = criterion
        ops = genotype[self.mode].split('||')
        self.args = args

        # node aggregator op
        self.lin1 = nn.Linear(in_dim, hidden_size)

        self.gnn_layers = nn.ModuleList(
            [NaOp(ops[i], hidden_size, hidden_size, act, with_linear=args.with_linear) for i in range(num_layers)])

        # skip op
        if self.args.fix_last:
            if self.num_layers > 1:
                self.sc_layers = nn.ModuleList(
                    [ScOp(ops[i+num_layers]) for i in range(num_layers - 1)])
            else:
                self.sc_layers = nn.ModuleList([ScOp(ops[num_layers])])
        else:
            # no output conditions.
            skip_op = ops[num_layers:2 * num_layers]
            if skip_op == ['none'] * num_layers:
                skip_op[-1] = 'skip'
                print('skip_op:', skip_op)
            self.sc_layers = nn.ModuleList(
                [ScOp(skip_op[i]) for i in range(num_layers)])

        # layer norm
        self.lns = torch.nn.ModuleList()
        if self.args.with_layernorm:
            for _ in range(num_layers):
                self.lns.append(LayerNorm(hidden_size, affine=True))

        # layer aggregator op
        self.layer6 = LaOp(ops[-2], hidden_size, 'linear', num_layers)

        self.readout = PoolOp(ops[-1])

    def forward(self, data):
        x, batch = data.x, data.batch
        if self.mode == 'TD':
            edge_index = data.edge_index
        elif self.mode == 'BU':
            edge_index = data.BU_edge_index
        # generate weights by softmax
        h = self.lin1(x)
        h = F.dropout(h, p=self.in_dropout, training=self.training)
        js = []
        for i in range(self.num_layers):
            h = self.gnn_layers[i](h, edge_index)
            if self.args.with_layernorm:
                # layer_norm = nn.LayerNorm(normalized_shape=x.size(), elementwise_affine=False)
                # x = layer_norm(x)
                h = self.lns[i](h)
            h = F.dropout(h, p=self.in_dropout, training=self.training)
            if i == self.num_layers - 1 and self.args.fix_last:
                js.append(h)
            else:
                js.append(self.sc_layers[i](h))
        h = self.layer6(js)
        h = F.dropout(h, p=self.out_dropout, training=self.training)

        h = self.readout(h, batch)
        return h


class BiGNN4Tune(nn.Module):
    def __init__(self, genotype, criterion, in_dim, out_dim, hidden_size,
                 num_layers=3, in_dropout=0.5, out_dropout=0.5, act='relu', is_mlp=False, args=None):
        super(BiGNN4Tune, self).__init__()
        self.TD_Block = UniBlock4Tune('TD', genotype, criterion, in_dim, out_dim, hidden_size,
                                      num_layers, in_dropout, out_dropout, act, is_mlp, args)
        self.BU_Block = UniBlock4Tune('BU', genotype, criterion, in_dim, out_dim, hidden_size,
                                      num_layers, in_dropout, out_dropout, act, is_mlp, args)

    def forward(self, data):
        TD_h = self.TD_Block(data)
        BU_h = self.BU_Block(data)
        h = torch.cat((TD_h, BU_h), 1)
        return h


class Network4Tune(nn.Module):
    def __init__(self, genotype, criterion, in_dim, out_dim, hidden_size,
                 num_layers=3, in_dropout=0.5, out_dropout=0.5, act='relu', is_mlp=False, args=None):
        super(Network4Tune, self).__init__()
        self.genotype = genotype
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.in_dropout = in_dropout
        self.out_dropout = out_dropout
        self._criterion = criterion
        ops = genotype['SEQ']
        self.args = args
        self.GNN = BiGNN4Tune(genotype, criterion, in_dim, out_dim, hidden_size,
                              num_layers, in_dropout, out_dropout, act, is_mlp, args)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim))
        self.seqop = SeqOp(ops, hidden_size, 'leaky_relu', num_layers)

    def forward(self, snapshots):
        h = [self.GNN(s) for s in snapshots]
        h = self.seqop(h)
        h = self.classifier(h)
        return F.log_softmax(h, dim=1)
