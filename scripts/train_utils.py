import os, sys, shutil
from collections import defaultdict

import treeDS


def generate_batch(data, batch_size):
    i1 = 0
    data_size = len(data)
    while True:
        i2 = min(i1 + batch_size, data_size)
        new_batch = data[i1:i2]
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


def copy_weight_files(save_dir, new_path, old_path):
    shutil.copyfile(save_dir + "%s.index" % old_path, save_dir + "%s.index" % new_path)
    shutil.copyfile(save_dir + "%s.meta" % old_path, save_dir + "%s.meta" % new_path)
    shutil.copyfile(
        save_dir + "%s.data-00000-of-00001" % old_path,
        save_dir + "%s.data-00000-of-00001" % new_path,
    )
