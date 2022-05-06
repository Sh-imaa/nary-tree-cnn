import os, sys, shutil
from collections import defaultdict

import treeDS


def generate_batch(data, batch_size, one_epoch=False):
    i1 = 0
    data_size = len(data)
    end_of_epoch = False
    while not (one_epoch and end_of_epoch):
        i2 = int(min(i1 + batch_size, data_size))
        new_batch = data[i1:i2]
        end_of_epoch = i2 == data_size
        i1 = i2 % data_size

        # pad the batch
        node_by_level = defaultdict(list)
        max_nodes = {}

        for tree in new_batch:
            treeDS.get_nodes_per_level(tree.root, node_by_level)

        for level, nodes in node_by_level.items():
            max_nodes[level] = max([len(n.c) for n in nodes])

        for tree in new_batch:
            treeDS.get_max_nodes(tree.root, max_nodes)
            treeDS.pad(tree.root)
        yield new_batch
