from sklearn.metrics import f1_score
import datetime
import logging
import os.path as osp
import warnings

import ray
import torch
import torch.nn as nn
from Libs.Data.dataset import get_dataloader
from Libs.Exp.config import tune_args
from Libs.Models.model4tune import Network4Tune
from ray import air, tune
from ray.air import session
from ray.tune import ResultGrid
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from torch.autograd import Variable

warnings.simplefilter(action='ignore', category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(data, model, criterion, optimizer):
    model.train()
    total_loss, total_acc, total_macro_f1, total_micro_f1 = 0.0, [], [], []

    for train_data_batch in data[0]:
        snapshots = [train_data_batch[i].to(device)
                     for i in range(tune_args.snapshot_num)]
        target = Variable(train_data_batch[0].y).to(device)

        optimizer.zero_grad()
        output = model(snapshots).to(device)
        loss = criterion(output, target)
        total_loss += loss.item()

        _, pred = output.max(dim=-1)
        correct = pred.eq(train_data_batch[0].y).sum().item()
        total_acc.append(correct / len(train_data_batch[0].y))

        macro_f1 = f1_score(train_data_batch[0].y.cpu(
        ).numpy(), pred.cpu().numpy(), average='macro')
        micro_f1 = f1_score(train_data_batch[0].y.cpu(
        ).numpy(), pred.cpu().numpy(), average='micro')
        total_macro_f1.append(macro_f1)
        total_micro_f1.append(micro_f1)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), tune_args.grad_clip)
        optimizer.step()

    return total_loss / len(data[0].dataset), np.mean(total_acc), np.mean(total_macro_f1), np.mean(total_micro_f1)


def infer_epoch(data, model, criterion, test=False):
    model.eval()
    total_loss, total_acc,  total_macro_f1 = 0.0, [], []
    infer_data = data[2] if test else data[1]
    for val_data_batch in infer_data:
        snapshots = [val_data_batch[i].to(device)
                     for i in range(tune_args.snapshot_num)]
        target = Variable(val_data_batch[0].y).to(device)
        with torch.no_grad():
            output = model(snapshots).to(device)

        loss = criterion(output, target)
        total_loss += loss.item()
        _, pred = output.max(dim=-1)
        correct = pred.eq(val_data_batch[0].y).sum().item()
        total_acc.append(correct / len(val_data_batch[0].y))

        macro_f1 = f1_score(val_data_batch[0].y.cpu(
        ).numpy(), pred.cpu().numpy(), average='macro')
        total_macro_f1.append(macro_f1)

    return total_loss / len(data[0].dataset), np.mean(total_acc), np.mean(total_macro_f1)


def train_fast_dev(epochs=5):
    with open(tune_args.arch_filename, 'r') as arch_file:
        genotype = arch_file.read()
    data, n_features = get_dataloader()

    d_in = n_features
    d_hidden = tune_args.hidden_size
    d_out = 1

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    model = Network4Tune(
        genotype=genotype,
        criterion=criterion,
        in_dim=d_in,
        out_dim=d_out,
        hidden_size=d_hidden,
        # hidden_size=config['hidden_size'],
        num_layers=tune_args.num_layers,
        in_dropout=tune_args.in_dropout,
        # in_dropout=config['in_dropout'],
        out_dropout=tune_args.out_dropout,
        # out_dropout=config['out_dropout'],
        act=tune_args.activation,
        # act=config['activation'],
        is_mlp=False,
        args=tune_args
    )
    model = model.to(device)

    optimizer = torch.optim.Adagrad(
        model.parameters(),
        tune_args.learning_rate,
        # config['learning_rate'],
        weight_decay=tune_args.weight_decay,
        # config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(tune_args.epochs))
    best_val_acc = best_test_acc = 0
    best_val_f1 = best_test_f1 = 0
    for epoch in range(1):
        train_loss, train_acc, train_macro_f1, _ = train_epoch(
            data, model, criterion, optimizer)
        if tune_args.cos_lr:
            scheduler.step()

        val_loss, val_acc, val_macro_f1 = infer_epoch(
            data, model, criterion)
        test_loss, test_acc, test_macro_f1 = infer_epoch(
            data, model, criterion, test=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_test_f1 = test_macro_f1

        if epoch % 1 == 0:
            msg = f"[Epoch {epoch+1}/{tune_args.epochs}] with LR: {scheduler.get_last_lr()[0]:.10f}\n"
            msg += f"<Train> Train Loss: {train_loss:.04f} | Train Acc: {train_acc:.04f} | Train F1: {train_macro_f1:.04f}\n"
            msg += f"<Val> Val Loss: {val_loss:.04f} | Best Val Acc: {best_val_acc:.04f} | Best Val F1: {best_val_f1:.04f}\n"
            msg += f"<Test> Best Test Acc: {best_test_acc:.04f}  | Best Test F1: {best_test_f1:.04f} "
            print(msg)
            logging.info(msg)


def train4tune_ray(config):
    with open(osp.join(tune_args.root, tune_args.arch_filename), 'r') as arch_file:
        genotype = arch_file.read()
    # hidden_size = tune_args.hidden_size

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    data, n_features = get_dataloader()

    d_in = n_features
    d_out = 1
    model = Network4Tune(
        genotype=genotype,
        criterion=criterion,
        in_dim=d_in,
        out_dim=d_out,
        hidden_size=config['hidden_size'],
        num_layers=tune_args.num_layers,
        # in_dropout=tune_args.in_dropout,
        in_dropout=config['in_dropout'],
        # out_dropout=tune_args.out_dropout,
        out_dropout=config['out_dropout'],
        # act=tune_args.activation,
        act=config['activation'],
        is_mlp=False,
        args=tune_args
    )
    model = model.to(device)
    optimizer = config['optimizer'](
        model.parameters(),
        # tune_args.learning_rate,
        config['learning_rate'],
        # weight_decay=tune_args.weight_decay,
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(tune_args.epochs))
    best_val_acc = best_test_acc = 0
    best_val_f1 = best_test_f1 = 0
    for _ in range(tune_args.epochs):
        train_loss, train_acc, train_macro_f1, _ = train_epoch(
            data, model, criterion, optimizer)
        if tune_args.cos_lr:
            scheduler.step()

        val_loss, val_acc, val_macro_f1 = infer_epoch(
            data, model, criterion)
        test_loss, test_acc, test_f1 = infer_epoch(
            data, model, criterion, test=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
        # Send the current training result back to Tune
        session.report({"f1_score": best_test_f1})


def finetune(num_samples=100, max_num_epochs=100, gpus_per_trial=1):
    exp_name = f"{tune_args.dataset_name}/trial_log"
    storage_path = osp.join(tune_args.root, "Output/Logs/FineTune")
    trainable = tune.with_resources(
        train4tune_ray, {"cpu": 16, "gpu": gpus_per_trial})
    tuner = tune.Tuner(
        trainable,
        param_space={'model': 'AutoCPP',
                     'hidden_size': tune.choice([32, 64, 128, 256]),
                     'learning_rate': tune.loguniform(1e-4, 1e-1),
                     'weight_decay': tune.loguniform(5e-4, 5e-3),
                     'optimizer': tune.choice([torch.optim.Adagrad, torch.optim.Adam]),
                     'in_dropout': tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                     'out_dropout': tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                     'activation': tune.choice(['relu', 'elu', 'leaky_relu'])
                     },
        run_config=air.RunConfig(
            name=exp_name,
            stop={"training_iteration": max_num_epochs},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="f1_score",
                num_to_keep=5,
            ),
            storage_path=storage_path,
        ),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(
                metric='f1_score',
                mode='max',
                max_t=max_num_epochs,
                grace_period=1,
                reduction_factor=2
            ),
            num_samples=num_samples,
        ),
    )
    result_grid: ResultGrid = tuner.fit()
    result_grid.get_dataframe().to_csv(osp.join(tune_args.root,
                                                "Output/Results/",
                                                tune_args.dataset_name,
                                                "FineTune",
                                                f'result_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'))


def run():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(logging_level=logging.ERROR)
    finetune()


run()
