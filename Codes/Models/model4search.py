import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from Libs.Models.genotype import (LA_PRIMITIVES, NA_PRIMITIVES,
                                  POOL_PRIMITIVES, SC_PRIMITIVES, SEQ_PRIMITIVES)
from Libs.Models.mixed_ops import (LaMixedOp, NaMixedOp, PoolingMixedOp,
                                   ScMixedOp, SeqMixedOp)
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UniBlock(nn.Module):

    def __init__(self, mode, criterion, in_dim, out_dim, hidden_size, num_layers=3, dropout=0.5, epsilon=0.0, with_conv_linear=False, args=None):
        super(UniBlock, self).__init__()
        assert mode in ['TD', 'BU']
        self.mode = mode
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._criterion = criterion
        self.dropout = dropout
        self.epsilon = epsilon
        # self.explore_num = 0
        self.with_linear = with_conv_linear
        self.args = args

        # node aggregator op
        self.lin1 = nn.Linear(in_dim, hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(
                NaMixedOp(hidden_size, hidden_size, self.with_linear))
        # self.layer1 = NaMixedOp(hidden_size, hidden_size,self.with_linear)
        # self.layer2 = NaMixedOp(hidden_size, hidden_size,self.with_linear)
        # self.layer3 = NaMixedOp(hidden_size, hidden_size,self.with_linear)

        # skip op
        self.scops = nn.ModuleList()
        for _ in range(self.num_layers-1):
            self.scops.append(ScMixedOp())
        if not self.args.fix_last:
            self.scops.append(ScMixedOp())
        # self.layer4 = ScMixedOp()
        # self.layer5 = ScMixedOp()
        # if not self.args.fix_last:
        #     self.layer6 = ScMixedOp()
        # layer aggregator op
        self.laop = LaMixedOp(hidden_size, num_layers)
        # pooling layer
        self.poolop = PoolingMixedOp()

        self._initialize_alphas()

    def _initialize_alphas(self):
        self._initialize_randomly()
        # self._initialize_He()

        self._arch_parameters = [
            self.na_alphas,
            self.sc_alphas,
            self.la_alphas,
            self.pool_alphas,
        ]

    def _initialize_randomly(self):

        num_na_ops = len(NA_PRIMITIVES)
        num_sc_ops = len(SC_PRIMITIVES)
        num_la_ops = len(LA_PRIMITIVES)
        num_pool_ops = len(POOL_PRIMITIVES)

        self.na_alphas = Variable(
            1e-3*torch.randn(self.num_layers, num_na_ops).cuda(), requires_grad=True)
        if self.args.fix_last:
            self.sc_alphas = Variable(
                1e-3*torch.randn(self.num_layers-1, num_sc_ops).cuda(), requires_grad=True)
        else:
            self.sc_alphas = Variable(
                1e-3*torch.randn(self.num_layers, num_sc_ops).cuda(), requires_grad=True)
        self.la_alphas = Variable(
            1e-3*torch.randn(1, num_la_ops).cuda(), requires_grad=True)
        self.pool_alphas = Variable(
            1e-3*torch.randn(1, num_pool_ops).cuda(), requires_grad=True)

    def _initialize_He(self):

        num_na_ops = len(NA_PRIMITIVES)
        num_sc_ops = len(SC_PRIMITIVES)
        num_la_ops = len(LA_PRIMITIVES)
        num_pool_ops = len(POOL_PRIMITIVES)

        self.na_alphas = Variable(torch.nn.init.kaiming_normal_(
            torch.empty(self.num_layers, num_na_ops).cuda()), requires_grad=True)
        if self.args.fix_last:
            self.sc_alphas = Variable(torch.nn.init.kaiming_normal_(
                torch.empty(self.num_layers-1, num_sc_ops).cuda()), requires_grad=True)
        else:
            self.sc_alphas = Variable(torch.nn.init.kaiming_normal_(
                torch.empty(self.num_layers, num_sc_ops).cuda()), requires_grad=True)

        self.la_alphas = Variable(torch.nn.init.kaiming_normal_(
            torch.empty(1, num_la_ops).cuda()), requires_grad=True)
        self.pool_alphas = Variable(torch.nn.init.kaiming_normal_(
            torch.empty(1, num_pool_ops).cuda()), requires_grad=True)

    def forward(self, data):
        print('Is NA OPS required grad: ',
              self._arch_parameters[0].requires_grad)
        x, batch = data.x, data.batch
        if self.mode == 'TD':
            edge_index = data.edge_index
        elif self.mode == 'BU':
            edge_index = data.BU_edge_index
        # prob = float(np.random.choice(range(1,11), 1) / 10.0)
        self.na_weights = F.softmax(self.na_alphas, dim=-1)
        self.sc_weights = F.softmax(self.sc_alphas, dim=-1)
        self.la_weights = F.softmax(self.la_alphas, dim=-1)
        self.pool_weights = F.softmax(self.pool_alphas, dim=-1)
        # generate weights by softmax
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        jk = []
        for i in range(self.num_layers):
            x = self.layers[i](x, self.na_weights[0], edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.args.fix_last and i == self.num_layers-1:
                jk += [x]
            else:
                jk += [self.scops[i](x, self.sc_weights[i])]

        merge_feature = self.laop(jk, self.la_weights[0])
        merge_feature = F.dropout(
            merge_feature, p=self.dropout, training=self.training)
        # merge_feature = scatter_mean(merge_feature, batch, dim=0)
        readout = self.poolop(merge_feature, batch, self.pool_weights[0])
        return readout


class BiGNN(nn.Module):
    def __init__(self, criterion, in_dim, out_dim, hidden_size, num_layers=3, dropout=0.5, epsilon=0.0, with_conv_linear=False, args=None):
        super(BiGNN, self).__init__()
        self.TD_Block = UniBlock('TD', criterion, in_dim, out_dim, hidden_size,
                                 num_layers, dropout, epsilon, with_conv_linear, args)
        self.BU_Block = UniBlock('BU', criterion, in_dim, out_dim, hidden_size,
                                 num_layers, dropout, epsilon, with_conv_linear, args)

    def forward(self, data):
        TD_h = self.TD_Block(data)
        BU_h = self.BU_Block(data)
        h = torch.cat((TD_h, BU_h), 1)
        return h


class Network(nn.Module):
    def __init__(self, criterion, in_dim, out_dim, hidden_size, num_layers=3, dropout=0.5, epsilon=0.0, with_conv_linear=False, args=None):
        super(Network, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._criterion = criterion
        self.dropout = dropout
        self.epsilon = epsilon
        # self.explore_num = 0
        self.with_linear = with_conv_linear
        self.args = args
        self.GNN = BiGNN(criterion, in_dim, out_dim, hidden_size,
                         num_layers, dropout, epsilon, with_conv_linear, args)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim))
        self.seqop = SeqMixedOp(hidden_size, num_layers)
        self._initialize_alphas()

    def forward(self, snapshots):
        self.seq_weights = F.softmax(self.seq_alphas, dim=-1)
        h = [self.GNN(snapshot) for snapshot in snapshots]
        h = self.seqop(h, self.seq_weights[0])
        h = self.classifier(h)
        return F.log_softmax(h, dim=1)

    def _loss(self, snapshots, batch, is_valid=True):
        input = self(snapshots).cuda()
        target = batch[0].y.cuda()
        return self._criterion(input, target)

    def _initialize_alphas(self):

        num_seq_ops = len(SEQ_PRIMITIVES)

        self.seq_alphas = Variable(
            1e-3*torch.randn(1, num_seq_ops).cuda(), requires_grad=True)

        self._arch_parameters = [
            self.GNN.TD_Block.na_alphas,
            self.GNN.TD_Block.sc_alphas,
            self.GNN.TD_Block.la_alphas,
            self.GNN.TD_Block.pool_alphas,

            self.GNN.BU_Block.na_alphas,
            self.GNN.BU_Block.sc_alphas,
            self.GNN.BU_Block.la_alphas,
            self.GNN.BU_Block.pool_alphas,

            self.seq_alphas,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(
                td_na_weights,
                td_sc_weights,
                td_la_weights,
                td_pool_weights,
                bu_na_weights,
                bu_sc_weights,
                bu_la_weights,
                bu_pool_weights,
                seq_weights):
            gene = []

            td_na_indices = torch.argmax(td_na_weights, dim=-1)
            for k in td_na_indices:
                gene.append(NA_PRIMITIVES[k])
            td_sc_indices = torch.argmax(td_sc_weights, dim=-1)
            for k in td_sc_indices:
                gene.append(SC_PRIMITIVES[k])
            td_la_indices = torch.argmax(td_la_weights, dim=-1)
            for k in td_la_indices:
                gene.append(LA_PRIMITIVES[k])
            td_pool_indices = torch.argmax(td_pool_weights, dim=-1)
            for k in td_pool_indices:
                gene.append(POOL_PRIMITIVES[k])

            bu_na_indices = torch.argmax(bu_na_weights, dim=-1)
            for k in bu_na_indices:
                gene.append(NA_PRIMITIVES[k])
            bu_sc_indices = torch.argmax(bu_sc_weights, dim=-1)
            for k in bu_sc_indices:
                gene.append(SC_PRIMITIVES[k])
            bu_la_indices = torch.argmax(bu_la_weights, dim=-1)
            for k in bu_la_indices:
                gene.append(LA_PRIMITIVES[k])
            bu_pool_indices = torch.argmax(bu_pool_weights, dim=-1)
            for k in bu_pool_indices:
                gene.append(POOL_PRIMITIVES[k])

            seq_indices = torch.argmax(seq_weights, dim=-1)
            for k in seq_indices:
                gene.append(SEQ_PRIMITIVES[k])
            return '||'.join(gene)

        gene = _parse(
            F.softmax(self.GNN.TD_Block.na_alphas, dim=-1).data.cpu(),
            F.softmax(self.GNN.TD_Block.sc_alphas, dim=-1).data.cpu(),
            F.softmax(self.GNN.TD_Block.la_alphas, dim=-1).data.cpu(),
            F.softmax(self.GNN.TD_Block.pool_alphas, dim=-1).data.cpu(),

            F.softmax(self.GNN.BU_Block.na_alphas, dim=-1).data.cpu(),
            F.softmax(self.GNN.BU_Block.sc_alphas, dim=-1).data.cpu(),
            F.softmax(self.GNN.BU_Block.la_alphas, dim=-1).data.cpu(),
            F.softmax(self.GNN.BU_Block.pool_alphas, dim=-1).data.cpu(),

            F.softmax(self.seq_alphas, dim=-1).data.cpu(),
        )

        return gene

    def sample_arch(self):
        gene = []
        for _ in range(2):
            for _ in range(3):
                op = np.random.choice(NA_PRIMITIVES, 1)[0]
                gene.append(op)
            for _ in range(2):
                op = np.random.choice(SC_PRIMITIVES, 1)[0]
                gene.append(op)
            op = np.random.choice(LA_PRIMITIVES, 1)[0]
            op = np.random.choice(POOL_PRIMITIVES, 1)[0]
        op = np.random.choice(SEQ_PRIMITIVES, 1)[0]
        gene.append(op)
        return '||'.join(gene)


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.args = args
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, data, eta, network_optimizer):
        loss = self.model._loss(data, is_valid=False)  # train loss
        theta = _concat(self.model.parameters()).data  # w
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.model.parameters()).mul_(self.network_momentum)
        except Exception:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(
            loss, self.model.parameters())).data + self.network_weight_decay * theta
        # gradient, L2 norm
        unrolled_model = self._construct_model_from_theta(
            theta.sub(moment + dtheta, alpha=eta))  # one-step update, get w' for Eq.7 in the paper
        return unrolled_model

    def step(self, data, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(data, eta, network_optimizer)
        else:
            self._backward_step(data, is_valid=True)
        self.optimizer.step()

    def _backward_step(self, loader, is_valid=True):
        for data in loader:
            data = data.to(device)
            loss = self.model._loss(data, is_valid)
            loss.backward()

    def _backward_step_unrolled(self, data, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(
            data, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(
            data, is_valid=True)  # validation loss

        unrolled_loss.backward()  # one-step update for w?
        # L_vali w.r.t alpha
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        # gradient, L_train w.r.t w, double check the model construction
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, data)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        # update alpha, which is the ultimate goal of this func, also the goal of the second-order darts
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.to(device)

    def _hessian_vector_product(self, vector, data, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)  # R * d(L_val/w', i.e., get w^+
        loss = self.model._loss(data, is_valid=False)  # train loss
        grads_p = torch.autograd.grad(
            loss, self.model.arch_parameters())  # d(L_train)/d_alpha, w^+

        for p, v in zip(self.model.parameters(), vector):
            # get w^-, need to subtract 2 * R since it has add R
            p.data.sub_(2*R, v)
        loss = self.model._loss(data, is_valid=False)  # train loss
        grads_n = torch.autograd.grad(
            loss, self.model.arch_parameters())  # d(L_train)/d_alpha, w^-

        # reset to the orignial w, always using the self.model, i.e., the original model
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
