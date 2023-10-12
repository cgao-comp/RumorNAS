import random
import numpy as np
import re
import os


def split_train_val_test(label_id_dict, train_size=0.8):
    train_data = []
    val_data = []
    test_data = []

    for label in label_id_dict:
        random.shuffle(label_id_dict[label])
        event_ids = np.array(label_id_dict[label])
        total_samples = len(event_ids)
        train_samples = int(total_samples * train_size)  # 60% for training
        val_samples = int(total_samples * ((1 - train_size) / 2))  # 20% for validation
        # test_samples = total_samples - train_samples - val_samples  # Remaining 20% for testing

        train_ids = event_ids[:train_samples]
        val_ids = event_ids[train_samples:train_samples + val_samples]
        test_ids = event_ids[train_samples + val_samples:]

        train_data.extend(train_ids)
        val_data.extend(val_ids)
        test_data.extend(test_ids)

    random.shuffle(train_data)

    random.shuffle(val_data)
    random.shuffle(test_data)
    return train_data, val_data, test_data


def load_snapshot_trees(paths, id_label_dict, sequences_dict, snapshot_num):
    trees_dict = {}
    current_snapshot = 0
    for line in open(paths['resource_tree']):  # loop for root posts
        elements = line.strip().split('\t')
        event_id = elements[0]
        parent_index, child_index = elements[1], int(elements[2])
        word_features = elements[5]
        if event_id not in id_label_dict:
            continue
        if parent_index != 'None':  # not root
            continue
        if event_id not in trees_dict:
            current_snapshot = 0
            trees_dict[event_id] = {
                snapshot_index: {} for snapshot_index in range(snapshot_num)
            }
        for snapshot_index in range(current_snapshot, snapshot_num):
            trees_dict[event_id][snapshot_index][child_index] = {
                'parent_index': parent_index,
                'word_features': word_features,
            }

    print(len(trees_dict.keys()))
    print(list(trees_dict.keys())[:10])

    prev_event_id = None
    for line in open(paths['resource_tree']):
        elements = line.strip().split('\t')
        event_id, parent_index, child_index = elements[0], elements[1], int(
            elements[2])  # None
        _, _, word_features = int(elements[3]), int(elements[4]), elements[5]

        if prev_event_id != event_id:
            edge_index = 1  # responsive post count, without root node
            current_snapshot = 0

        prev_event_id = event_id

        if event_id not in id_label_dict:
            continue

        if parent_index == 'None':  # root post
            continue

        for snapshot_index in range(current_snapshot, snapshot_num):
            trees_dict[event_id][snapshot_index][child_index] = {
                'parent_index': parent_index,
                'word_features': word_features,
            }

        print(sequences_dict[event_id], '\t', edge_index,
              '\t', current_snapshot, snapshot_num, event_id)

        while current_snapshot < snapshot_num:
            if edge_index == sequences_dict[event_id][current_snapshot]:
                current_snapshot += 1
            else:
                break

        if current_snapshot == snapshot_num:  # next event_id
            continue

        if sequences_dict[event_id][current_snapshot - 1] != sequences_dict[event_id][current_snapshot]:
            edge_index += 1

    return trees_dict


def load_snapshot_trees_weibo(paths, id_label_dict, sequences_dict, snapshot_num):
    trees_dict = {}
    current_snapshot = 0
    for line in open(paths['resource_tree']):  # loop for root posts
        elements = line.strip().split('\t')
        event_id = elements[0]
        parent_index = elements[1]
        child_index = int(elements[2])
        word_features = elements[3]
        if event_id not in id_label_dict:
            continue
        if parent_index != 'None':  # not root
            continue
        if event_id not in trees_dict:
            current_snapshot = 0
            trees_dict[event_id] = {
                snapshot_index: {} for snapshot_index in range(snapshot_num)
            }
        for snapshot_index in range(current_snapshot, snapshot_num):
            trees_dict[event_id][snapshot_index][child_index] = {
                'parent_index': parent_index,
                'word_features': word_features,
            }

    print(len(trees_dict.keys()))
    print(list(trees_dict.keys())[:10])

    prev_event_id = None
    for line in open(paths['resource_tree']):
        elements = line.strip().split('\t')
        event_id, parent_index, child_index = elements[0], elements[1], int(
            elements[2])  # None
        word_features = elements[3]

        if prev_event_id != event_id:
            edge_index = 1  # responsive post count, without root node
            current_snapshot = 0

        prev_event_id = event_id

        if event_id not in id_label_dict:
            continue

        if parent_index == 'None':  # root post
            continue

        for snapshot_index in range(current_snapshot, snapshot_num):
            trees_dict[event_id][snapshot_index][child_index] = {
                'parent_index': parent_index,
                'word_features': word_features,
            }

        print(sequences_dict[event_id], '\t', edge_index,
              '\t', current_snapshot, snapshot_num, event_id)

        while current_snapshot < snapshot_num:
            if edge_index == sequences_dict[event_id][current_snapshot]:
                current_snapshot += 1
            else:
                break

        if current_snapshot == snapshot_num:  # next event_id
            continue

        if sequences_dict[event_id][current_snapshot - 1] != sequences_dict[event_id][current_snapshot]:
            edge_index += 1

    return trees_dict


class TweetNode(object):
    def __init__(self, index=None):
        self.index = index
        self.parent = None
        self.children = []
        self.word_index = []
        self.word_frequency = []


class TweetTree(object):
    def __init__(self, path, event_id, label, tree, snapshot_index, snapshot_num):
        self.graph_path = path
        self.event_id = event_id
        self.label = label
        self.tree = tree
        self.snapshot_index = snapshot_index
        self.snapshot_num = snapshot_num
        self.construct_tree()
        self.construct_matrices()  # tree -> edge_matrix
        self.construct_word_features()  # x_word_index, x_word_frequency
        self.save_local()

    @staticmethod
    def str2matrix(s):  # str = index:wordfreq index:wordfreq
        word_index, word_frequency = [], []
        for pair in s.split(' '):
            pair = pair.split(':')
            index, frequency = int(pair[0]), float(pair[1])
            if index <= 5000:
                word_index.append(index)
                word_frequency.append(frequency)
        return word_index, word_frequency

    def construct_tree(self):  # from tree_dict
        tree_dict = self.tree
        index2node = {i: TweetNode(index=i) for i in tree_dict}
        for j in tree_dict:
            child_index = j
            child_node = index2node[child_index]
            word_index, word_frequency = self.str2matrix(
                tree_dict[j]['word_features'])
            child_node.word_index = word_index
            child_node.word_frequency = word_frequency
            parent_index = tree_dict[j]['parent_index']
            if parent_index == 'None':  # root post
                root_index = child_index - 1
                root_word_index = child_node.word_index
                root_word_frequency = child_node.word_frequency
            else:  # responsive post
                parent_node = index2node[int(parent_index)]
                child_node.parent = parent_node
                parent_node.children.append(child_node)
        root_features = np.zeros([1, 5000])
        if len(root_word_index) > 0:
            root_features[0, np.array(root_word_index)] = np.array(
                root_word_frequency)
        self.index2node = index2node
        self.root_index = root_index
        self.root_features = root_features

    def construct_matrices(self):  # tree2matrix, adjacency
        index2node = self.index2node
        row = []
        col = []
        x_word_index_list = []
        x_word_frequency_list = []
        for index_i in sorted(list(index2node.keys())):
            child_index = [
                child_node.index for child_node in index2node[index_i].children]
            for index_j in sorted(child_index):
                row.append(index_i-1)
                col.append(index_j-1)
        edge_matrix = [row, col]  # TODO: shift
        # shift indices
        # - new adjacency matrix for PyTorch Geometric
        index_map = {}
        for shifted_index, i in enumerate(sorted(set(row).union(set(col)))):
            index_map[i] = shifted_index
            x_word_index_list.append(index2node[i+1].word_index)
            x_word_frequency_list.append(index2node[i+1].word_frequency)
        new_row = [index_map[row_elem] for row_elem in row]
        new_col = [index_map[col_elem] for col_elem in col]
        edge_matrix = [new_row, new_col]  # TODO: shift

        self.root_index = index_map[self.root_index]
        self.edge_matrix = edge_matrix
        self.x_word_index_list = x_word_index_list
        self.x_word_frequency_list = x_word_frequency_list

    def construct_word_features(self):
        x_word_index_list = self.x_word_index_list
        x_word_frequency_list = self.x_word_frequency_list
        x_x = np.zeros([len(x_word_index_list), 5000])
        for i in range(len(x_word_index_list)):
            if len(x_word_index_list[i]) > 0:
                x_x[i, np.array(x_word_index_list[i])] = np.array(
                    x_word_frequency_list[i])
        self.x_x = x_x

    def save_local(self):
        root_index = np.array(self.root_index)
        root_features = np.array(self.root_features)
        edge_index = np.array(self.edge_matrix)
        x_x = np.array(self.x_x)
        label = np.array(self.label)
        FILE_PATH = f"{self.graph_path}/{self.event_id}_{self.snapshot_index}_{self.snapshot_num}.npz"
        np.savez(  # save snapshots
            FILE_PATH,
            x=x_x, y=label, edge_index=edge_index,
            root_index=root_index, root=root_features
        )


def load_raw_labels(path):
    id_label_dict = {}
    label_id_dict = {
        'true': [], 'false': [], 'unverified': [], 'non-rumor': []
    }
    for line in open(path):
        label, event_id = line.strip().split(":")
        id_label_dict[event_id] = label
        label_id_dict[label].append(event_id)
    # print("PATH: {0}, LEN: {1}".format(path, len(id_label_dict)))
    # print([(key, len(label_id_dict[key])) for key in label_id_dict])
    return id_label_dict, label_id_dict


def load_raw_trees(path):
    pass


def load_resource_labels(path):
    id_label_dict = {}
    label_id_dict = {
        'true': [], 'false': [], 'unverified': [], 'non-rumor': []
    }
    num_labels = {'true': 0, 'false': 1, 'unverified': 2, 'non-rumor': 3}
    for line in open(path):
        elements = line.strip().split('\t')
        label, event_id = elements[0], elements[2]
        id_label_dict[event_id] = label
        label_id_dict[label].append(event_id)
    for key in id_label_dict:
        id_label_dict[key] = num_labels[id_label_dict[key]]
    # print("PATH: {0}, LEN: {1}".format(path, len(id_label_dict)))
    # print([(key, len(label_id_dict[key])) for key in label_id_dict])
    return id_label_dict, label_id_dict


def load_resource_trees(path):
    trees_dict = {}
    for line in open(path):
        elements = line.strip().split('\t')
        event_id = elements[0]
        parent_index = elements[1]
        child_index = int(elements[2])
        word_features = elements[5]
        if event_id not in trees_dict:
            trees_dict[event_id] = {}
        trees_dict[event_id][child_index] = {
            'parent_index': parent_index,
            'word_features': word_features,
        }
    # print('Resource trees count:', len(trees_dict), '\n')
    return trees_dict


def load_resource_labels_weibo(path):  # Weibo Dataset
    id_label_dict = {}
    label_id_dict = {'0': [], '1': []}
    num_labels = {'0': 0, '1': 1}
    for line in open(path):
        elements = line.strip().split(' ')
        label, event_id = elements[1], elements[0]
        id_label_dict[event_id] = label
        label_id_dict[label].append(event_id)
    for key in id_label_dict:
        id_label_dict[key] = num_labels[id_label_dict[key]]
    print("PATH: {0}, LEN: {1}".format(path, len(id_label_dict)))
    print([(key, len(label_id_dict[key])) for key in label_id_dict])
    return id_label_dict, label_id_dict


def load_resource_trees_weibo(path):  # Weibo
    trees_dict = {}
    for line in open(path):
        elements = line.strip().split('\t')
        event_id = elements[0]
        parent_index = elements[1]
        child_index = int(elements[2])
        word_features = elements[3]
        if event_id not in trees_dict:
            trees_dict[event_id] = {}
        trees_dict[event_id][child_index] = {
            'parent_index': parent_index,
            'word_features': word_features,
        }
    # print('resource trees count:', len(trees_dict), '\n')
    return trees_dict


def raw_tree_to_timestamps(raw_tree_path, timestamps_path):  # temporal
    temporal_info = {}
    for file_name in os.listdir(raw_tree_path):
        file_id = file_name[:-4]
        if file_id not in temporal_info:
            temporal_info[file_id] = []
        for _, line in enumerate(open(raw_tree_path + file_name)):
            elem_list = re.split(r"[\'\,\->\[\]]", line.strip())
            elem_list = [x.strip() for x in elem_list if x.strip()]
            src_user_id, src_post_id, src_time = elem_list[:3]
            dest_user_id, dest_post_id, dest_time = elem_list[3:6]
            if src_user_id == 'ROOT' and src_post_id == 'ROOT':
                _, _ = dest_user_id, dest_post_id  # root_user_id, root_post_id
            elif src_post_id != dest_post_id:  # responsive posts
                temporal_info[file_id].append(max(src_time, dest_time))
        temporal_info[file_id] = sorted(
            temporal_info[file_id], key=lambda x: float(x.strip()))
    return temporal_info


def retrieve_temporal_info(temporal_info, resource):  # trim or upsample
    # ------------------------------------------
    # Twitter: Sequential and Temporal Snapshots
    # ------------------------------------------
    resource_id_label_dict = resource['id_label_dict']
    resource_trees_dict = resource['trees_dict']
    for event_id in resource_id_label_dict:
        raw_timestamps_count = len(temporal_info[event_id])
        resource_trees_len = len(resource_trees_dict[event_id]) - 1
        if raw_timestamps_count > resource_trees_len:  # trim
            temporal_info[event_id] = temporal_info[event_id][:resource_trees_len]
        elif raw_timestamps_count < resource_trees_len:  # upsample
            diff_count = resource_trees_len - raw_timestamps_count
            if not len(temporal_info[event_id]):
                upsample = ['1.0'] * diff_count
            elif len(temporal_info[event_id]) >= diff_count:
                upsample = random.sample(temporal_info[event_id], diff_count)
            else:
                upsample = [random.choice(temporal_info[event_id])
                            for _ in range(diff_count)]
            temporal_info[event_id] += upsample
            temporal_info[event_id] = sorted(
                temporal_info[event_id], key=lambda x: float(x.strip()))
    return temporal_info


def retrieve_sequential_info_weibo(resource):
    # ---------------------------
    # Weibo: Sequential Snapshots
    # ---------------------------
    resource_id_label_dict = resource['id_label_dict']
    resource_trees_dict = resource['trees_dict']
    sequential_info = {
        event_id: ['1.0'] * (len(resource_trees_dict[event_id]) - 1)
        for event_id in resource_id_label_dict
        if event_id in resource_trees_dict
    }
    return sequential_info


# Load Temporal Information - Generate Sequential, Temporal Edge Index

def sequence_to_snapshot_index(temporal_info, snapshot_num):
    snapshot_edge_index = {}
    for event_id in temporal_info:
        if event_id not in snapshot_edge_index:
            snapshot_edge_index[event_id] = []
        sequence_len = len(temporal_info[event_id])
        base_edge_count = sequence_len % snapshot_num
        additional_edge_count = sequence_len // snapshot_num
        for snapshot_index in range(1, snapshot_num + 1):
            count = base_edge_count + additional_edge_count * snapshot_index
            snapshot_edge_index[event_id]
            snapshot_edge_index[event_id].append(count)
    return snapshot_edge_index


def temporal_to_snapshot_index(temporal_info, snapshot_num):
    snapshot_edge_index = {}
    for event_id in temporal_info:
        if event_id not in snapshot_edge_index:
            snapshot_edge_index[event_id] = []
        if not temporal_info[event_id]:
            snapshot_edge_index[event_id] = [0] * snapshot_num
            continue
        sequence = sorted(temporal_info[event_id],
                          key=lambda x: float(x.strip()))
        sequence = list(map(float, sequence))
        time_interval = (sequence[-1] - sequence[0]) / snapshot_num
        for snapshot_index in range(1, snapshot_num + 1):
            edge_count = 0
            for seq in sequence:
                if seq <= time_interval * snapshot_index + sequence[0]:
                    edge_count += 1
                else:
                    break
            snapshot_edge_index[event_id].append(edge_count)
        snapshot_edge_index[event_id].pop()
        snapshot_edge_index[event_id].append(len(temporal_info[event_id]))  #
    return snapshot_edge_index
