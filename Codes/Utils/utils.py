# -*- coding: utf-8 -*-
# Copyright 2023 Haowei Xu
# Licensed under the MIT License

import json

__author__ = 'Haowei Xu'
__email__ = 'hwxu@mail.nwpu.edu.cn'

"""This module provides utility functions and classes for training and evaluating neural network models.

Attributes:
    MyDumper (class): Custom YAML dumper for better formatting.
    EVLocalAvg (class): Helper class for eigenvalues local average tracking.
    AvgrageMeter (class): Class to compute and store the average and current value.
"""


import os
import os.path as osp
import shutil
import subprocess

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from torch import cat
from torch.autograd import Variable


class MyDumper(yaml.Dumper):
    """Custom Dumper class for better YAML formatting."""

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


class EVLocalAvg(object):
    """Eigenvalues local average tracker.

    Attributes:
        window (int): Window size for local average computation.
        ev_freq (int): Eigenvalue computation frequency.
        epochs (int): Total number of epochs.
        ev (list): List of eigenvalues.
        ev_local_avg (list): List of local averages of eigenvalues.
        genotypes (dict): Dictionary of genotypes.
        la_epochs (dict): Dictionary of local average epochs.
        la_start_idx (int): Start index for local average window.
        la_end_idx (int): End index for local average window.
        stop_search (bool): Flag to indicate whether to stop search.
        stop_epoch (int): Epoch at which to stop.
        stop_genotype (namedtuple): Genotype at stopping point.
    """

    def __init__(self, window=5, ev_freq=2, total_epochs=50):
        """ Keep track of the eigenvalues local average.
        Args:
            window (int): number of elements used to compute local average.
                Default: 5
            ev_freq (int): frequency used to compute eigenvalues. Default:
                every 2 epochs
            total_epochs (int): total number of epochs that DARTS runs.
                Default: 50
        """
        self.window = window
        self.ev_freq = ev_freq
        self.epochs = total_epochs

        self.stop_search = False
        self.stop_epoch = total_epochs - 1
        self.stop_genotype = None

        self.ev = []
        self.ev_local_avg = []
        self.genotypes = {}
        self.la_epochs = {}

        # start and end index of the local average window
        self.la_start_idx = 0
        self.la_end_idx = self.window

    def reset(self):
        self.ev = []
        self.ev_local_avg = []
        self.genotypes = {}
        self.la_epochs = {}

    def update(self, epoch, ev, genotype):
        """ Method to update the local average list.
        Args:
            epoch (int): current epoch
            ev (float): current dominant eigenvalue
            genotype (namedtuple): current genotype
        """
        self.ev.append(ev)
        self.genotypes.update({epoch: genotype})
        # set the stop_genotype to the current genotype in case the early stop
        # procedure decides not to early stop
        self.stop_genotype = genotype

        # since the local average computation starts after the dominant
        # eigenvalue in the first epoch is already computed we have to wait
        # at least until we have 3 eigenvalues in the list.
        if (len(self.ev) >= int(np.ceil(self.window/2))) and (epoch < self.epochs - 1):
            # start sliding the window as soon as the number of eigenvalues in
            # the list becomes equal to the window size
            if len(self.ev) < self.window:
                self.ev_local_avg.append(np.mean(self.ev))
            else:
                assert len(
                    self.ev[self.la_start_idx: self.la_end_idx]) == self.window
                self.ev_local_avg.append(
                    np.mean(self.ev[self.la_start_idx: self.la_end_idx]))
                self.la_start_idx += 1
                self.la_end_idx += 1

            # keep track of the offset between the current epoch and the epoch
            # corresponding to the local average. NOTE: in the end the size of
            # self.ev and self.ev_local_avg should be equal
            self.la_epochs.update(
                {epoch: int(epoch - int(self.ev_freq*np.floor(self.window/2)))})

        elif len(self.ev) < int(np.ceil(self.window/2)):
            self.la_epochs.update({epoch: -1})

        # since there is an offset between the current epoch and the local
        # average epoch, loop in the last epoch to compute the local average of
        # these number of elements: window, window - 1, window - 2, ..., ceil(window/2)
        elif epoch == self.epochs - 1:
            for i in range(int(np.ceil(self.window/2))):
                assert len(
                    self.ev[self.la_start_idx: self.la_end_idx]) == self.window - i
                self.ev_local_avg.append(
                    np.mean(self.ev[self.la_start_idx:self.la_end_idx + 1]))
                self.la_start_idx += 1

    def early_stop(self, epoch, factor=1.18, es_start_epoch=10, delta=4):
        """ Early stopping criterion
        Args:
            epoch (int): current epoch
            factor (float): threshold factor for the ration between the current
                and prefious eigenvalue. Default: 1.3
            es_start_epoch (int): until this epoch do not consider early
                stopping. Default: 20
            delta (int): factor influencing which previous local average we
                consider for early stopping. Default: 2
        """
        if int(self.la_epochs[epoch] - self.ev_freq*delta) >= es_start_epoch:
            # the current local average corresponds to
            # epoch - int(self.ev_freq*np.floor(self.window/2))
            current_la = self.ev_local_avg[-1]
            # by default take the local average corresponding to epoch
            # delta*self.ev_freq
            previous_la = self.ev_local_avg[-1 - delta]

            self.stop_search = current_la / previous_la > factor
            if self.stop_search:
                self.stop_epoch = int(
                    self.la_epochs[epoch] - self.ev_freq*delta)
                self.stop_genotype = self.genotypes[self.stop_epoch]


class AvgrageMeter(object):
    """Keeps track of most recent, average, sum, and count of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics."""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """Update the statistics with new value `val`."""
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def count_parameters_in_MB(model):
    """Count the number of parameters in a model."""
    return np.sum(np.fromiter((np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name), dtype=int))/1e6


def save_checkpoint(state, is_best, save, epoch, task_id):
    """Save a model checkpoint."""
    filename = f"checkpoint_{task_id}_{epoch}.pth.tar"
    filename = os.path.join(save, filename)

    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def load_checkpoint(model, optimizer, architect, save, la_tracker, epoch, task_id):
    """Load a model checkpoint."""
    filename = f"checkpoint_{task_id}_{epoch}.pth.tar"
    filename = os.path.join(save, filename)

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['state_dict'])
    model.alphas_normal.data = checkpoint['alphas_normal']
    model.alphas_reduce.data = checkpoint['alphas_reduce']
    optimizer.load_state_dict(checkpoint['optimizer'])
    architect.optimizer.load_state_dict(checkpoint['arch_optimizer'])
    la_tracker.ev = checkpoint['ev']
    la_tracker.ev_local_avg = checkpoint['ev_local_avg']
    la_tracker.genotypes = checkpoint['genotypes']
    la_tracker.la_epochs = checkpoint['la_epochs']
    la_tracker.la_start_idx = checkpoint['la_start_idx']
    la_tracker.la_end_idx = checkpoint['la_end_idx']
    lr = checkpoint['lr']
    return lr


def save(model, model_path):
    """Save the model."""
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    """Load the model."""
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    """Apply drop path regularization."""
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(
            x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    """Create experiment directory and save scripts."""
    if not os.path.exists(path):
        os.mkdir(path)
    print(f'Experiment dir : {path}')

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def index_to_mask(index, size):
    """Convert an index tensor to a boolean mask."""
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def gen_uniform_60_20_20_split(data):
    """Generate a uniform 60/20/20 data split."""
    skf = StratifiedKFold(10, shuffle=True, random_state=12345)
    idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]
    return cat(idx[:6], 0), cat(idx[6:8], 0), cat(idx[8:], 0)


def save_load_split(data, raw_dir, run, gen_splits):
    """Save or load data splits."""
    prefix = gen_splits.__name__[4:-6]
    path = osp.join(raw_dir, '..', '{}_{:03d}.pt'.format(prefix, run))

    if osp.exists(path):
        split = torch.load(path)
    else:
        split = gen_splits(data)
        torch.save(split, path)

    split = gen_splits(data)
    data.train_mask = index_to_mask(split[0], data.num_nodes)
    data.val_mask = index_to_mask(split[1], data.num_nodes)
    data.test_mask = index_to_mask(split[2], data.num_nodes)

    return data


def write_yaml_results_eval(args, results_file, result_to_log):
    """Write evaluation results to a YAML file."""
    setting = '_'.join([args.space, args.dataset_name])
    regularization = '_'.join(
        [str(args.search_dp), str(args.search_wd)]
    )
    results_file = os.path.join(args.save, f'{results_file}.yaml')

    try:
        with open(results_file, 'r') as f:
            result = yaml.load(f)
        if setting in result.keys():
            if regularization in result[setting].keys():
                if args.search_task_id in result[setting][regularization]:
                    result[setting][regularization][args.search_task_id].append(
                        result_to_log)
                else:
                    result[setting][regularization].update({args.search_task_id:
                                                           [result_to_log]})
            else:
                result[setting].update({regularization: {args.search_task_id:
                                                         [result_to_log]}})
        else:
            result.update({setting: {regularization: {args.search_task_id:
                                                      [result_to_log]}}})
        with open(results_file, 'w') as f:
            yaml.dump(result, f, Dumper=MyDumper, default_flow_style=False)
    except (AttributeError, FileNotFoundError):
        result = {
            setting: {
                regularization: {
                    args.search_task_id: [result_to_log]
                }
            }
        }
        with open(results_file, 'w') as f:
            yaml.dump(result, f, Dumper=MyDumper, default_flow_style=False)


def write_yaml_results(args, results_file, result_to_log):
    """Write results to a YAML file."""
    setting = '_'.join([args.space, args.dataset_name])
    regularization = '_'.join(
        [str(args.drop_path_prob), str(args.weight_decay)]
    )
    results_file = os.path.join(args.save, f'{results_file}.yaml')

    try:
        with open(results_file, 'r') as f:
            result = yaml.load(f)
        if setting in result.keys():
            if regularization in result[setting].keys():
                result[setting][regularization].update(
                    {args.task_id: result_to_log})
            else:
                result[setting].update(
                    {regularization: {args.task_id: result_to_log}})
        else:
            result.update(
                {setting: {regularization: {args.task_id: result_to_log}}})
        with open(results_file, 'w') as f:
            yaml.dump(result, f, Dumper=MyDumper, default_flow_style=False)
    except (AttributeError, FileNotFoundError):
        result = {
            setting: {
                regularization: {
                    args.task_id: result_to_log
                }
            }
        }
        with open(results_file, 'w') as f:
            yaml.dump(result, f, Dumper=MyDumper, default_flow_style=False)


def select_gpu():
    """Select a GPU based on memory availability."""
    sp = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.free,memory.total',
                          '--format=csv,nounits,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode('utf-8').split('\n')
    out_list = [x for x in out_list if x]
    out_list = [tuple(map(int, x.split(','))) for x in out_list]

    mem_ratio = [(i, free/total) for i, (free, total) in enumerate(out_list)]

    best_gpu = sorted(mem_ratio, key=lambda x: x[1], reverse=True)[0][0]

    print(
        f"Selected GPU {best_gpu} with free/total memory ratio of {mem_ratio[best_gpu][1]*100:.2f}%.")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(best_gpu)


def path_join(*elem):
    return os.path.join(*elem)


def ensure_directory(path):
    path = os.path.split(path)
    if not os.path.exists(path[0]):
        os.makedirs(path[0])


def save_json_file(path, data):
    ensure_directory(path)
    with open(path, "w") as json_file:
        json_file.write(json.dumps(data))


def append_json_file(path, data):
    ensure_directory(path)
    with open(path, 'a') as json_file:
        json_file.write(json.dumps(data))


def write_data(path, data):
    ensure_directory(path)
    with open(path, "w") as file:
        file.write(data)


def load_json_file(path):
    with open(path, "r") as json_file:
        data = json.loads(json_file.read())
    return data


def print_dict(dict_file):
    for key in dict_file.keys():
        print("\t {0}: {1}".format(key, dict_file[key]))
    print()
