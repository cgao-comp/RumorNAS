import os
import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from Libs.Data.dataset_utils import (load_resource_labels,
                                     load_resource_trees,
                                     split_train_val_test)
from Libs.Exp.config import search_args
from torch_geometric.loader import DataLoader


class RumorsPropagationDataset(Dataset):
    def __init__(
        self,
        root: str = osp.join(os.getcwd(), 'Input'),
        name: str = 'Twitter15',
        type: str = 'sequential',
        snapshot_num: int = 5,
        split: str = "train",
        lower: int = 2,
        upper: int = 1e5,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        r"""
        The tree-structured rumors propagation graph classification datasets from the following two papers:
        `"Real-time Rumor Debunking on Twitter"` and `"Detecting Rumors from Microblogs with Recurrent Neural Networks"`
        Each dataset can be passed a transform, a pre_transform and a pre_filter function, which are None by default.
        Detailed information on the datasets can be found in the [(LINK)](https://github.com/majingCUHK/Rumor_RvNN).

        search_args:
                root (str): Root directory where the dataset should be saved.
                name (str): The name of the graph set (:obj:`"Twitter15"`, :obj:`"Twitter16"`, :obj:`"Weibo"`).
                split (str, optional): If :obj:`"train"`, loads the training dataset.
                    If :obj:`"val"`, loads the validation dataset.
                    If :obj:`"test"`, loads the test dataset.
                    (default: :obj:`"train"`)
                type (str): Two different ways of depicting the dynamic graphs as $T$ step graph snapshots.
                    If :obj:`"sequential"`, consider the ordering of the additional nodes and links of the propagation tree.
                    If :obj:`"temporal"`, consider temporal information of the propagation tree.
                    More details can be found in paper `"Dynamic Graph Convolutional Networks with Attention Mechanism for Rumor Detection on Social Media"`
                snapshot_num (int): The number of generated snapshots, which can be taken between {2, 3, 5, 10}.
                transform (callable, optional): A function/transform that takes in an
                    :obj:`torch_geometric.data.Data` object and returns a transformed
                    version. The data object will be transformed before every access.
                    (default: :obj:`None`)
                pre_transform (callable, optional): A function/transform that takes in
                    an :obj:`torch_geometric.data.Data` object and returns a
                    transformed version. The data object will be transformed before
                    being saved to disk. (default: :obj:`None`)
                pre_filter (callable, optional): A function that takes in an
                    :obj:`torch_geometric.data.Data` object and returns a boolean
                    value, indicating whether the data object should be included in the
                    final dataset. (default: :obj:`None`)
        """
        self.root = root
        self.name = name
        self.type = type
        tree_path = osp.join(
            self.root, f'raw_data/resources/{self.name}/data.TD_RvNN.vol_5000.txt')
        label_path = osp.join(
            root, f'raw_data/resources/{self.name}/{self.name}_label_all.txt')
        _, label_id_dict = load_resource_labels(label_path)
        self.tree_dict = load_resource_trees(tree_path)
        train_idx, val_idx, test_idx = split_train_val_test(label_id_dict)
        assert split in {'train', 'val', 'test'}
        self.split = split
        if self.split == 'test':
            self.idx = test_idx
        elif self.split == 'train':
            self.idx = train_idx
        elif self.split == 'val':
            self.idx = val_idx
        self.idx = list(filter(lambda id: id in self.tree_dict and lower <= len(
            self.tree_dict[id]) <= upper, self.idx))
        self.snapshots_path = osp.join(
            self.root, 'raw_data/data/graph', self.name, f'{self.type}_snapshot')
        assert snapshot_num in {2, 3, 5, 10}
        self.snapshot_num = snapshot_num
        super().__init__(root, transform, pre_transform, pre_filter)
        self.path = osp.join(
            self.processed_dir, f'{self.split}_{self.type}_{self.snapshot_num}.pt')
        self.slices = None

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed', self.name)

    @property
    def processed_file_names(self):
        return [
            f'train_{self.type}_{self.snapshot_num}.pt',
            f'val_{self.type}_{self.snapshot_num}.pt',
            f'test_{self.type}_{self.snapshot_num}.pt'
        ]

    def len(self):
        return len(self.idx)

    def get(self, index):
        # if osp.exists(self.path):
        #     self.snapshots = torch.load(self.path)
        # else:
        event_id = self.idx[index]
        self.snapshots = []
        for snapshot_index in range(self.snapshot_num):
            data = np.load(
                f"{self.snapshots_path}/{event_id}_{snapshot_index}_{self.snapshot_num}.npz",
                allow_pickle=True,
            )
            edgeindex = data['edge_index']
            new_td_edgeindex = edgeindex
            burow = list(edgeindex[1])
            bucol = list(edgeindex[0])
            new_bu_edgeindex = [burow, bucol]
            data = Data(
                x=torch.tensor(data['x'], dtype=torch.float32),
                y=torch.LongTensor([int(data['y'])]),
                edge_index=torch.LongTensor(new_td_edgeindex),
                edge_attr=torch.ones(new_td_edgeindex.shape[1]),
                BU_edge_index=torch.LongTensor(new_bu_edgeindex),
                root=torch.LongTensor(data['root']),
                root_index=torch.LongTensor([int(data['root_index'])]),
            )
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            self.snapshots.append(data)
        # torch.save(self.snapshots, self.path)
        return self.snapshots


def get_dataloader():
    train_dataset = RumorsPropagationDataset(
        name=search_args.dataset_name, type=search_args.dataset_type, snapshot_num=search_args.snapshot_num, split='train', )
    val_dataset = RumorsPropagationDataset(
        name=search_args.dataset_name, type=search_args.dataset_type, snapshot_num=search_args.snapshot_num, split='val')
    test_dataset = RumorsPropagationDataset(
        name=search_args.dataset_name, type=search_args.dataset_type, snapshot_num=search_args.snapshot_num, split='test')
    train_loader = DataLoader(train_dataset, batch_size=search_args.batch_size,
                              shuffle=True, num_workers=5)
    val_loader = DataLoader(val_dataset, batch_size=search_args.batch_size,
                            shuffle=False, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=search_args.batch_size,
                             shuffle=False, num_workers=5)
    data = [train_loader, val_loader, test_loader]
    d_in = train_dataset[0].num_features
    return data, d_in
