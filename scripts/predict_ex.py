import os, sys, shutil, time, itertools
import math, random
from collections import OrderedDict, defaultdict

import pickle, copy
import numpy as np
import argparse

import tensorflow as tf
from tensorflow.python.ops import variable_scope, init_ops

import utils
import treeDS

from nary_tree_cnn import TreeCNN

MODEL_STR = "tree_cnn_lr=%f_l2=%f_dr1=%f_dr2=%f_batch_size=%d.weights"
SAVE_DIR = "../weights/"
LOG_DIR = "./logs/"


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


class Config(object):
    """Holds model hyperparams and data information.
    Model objects are passed a Config() object at instantiation.
    """

    optimizer = "Adam"
    embed_size = 44
    label_size = 2
    early_stopping = 20
    act_fun = "relu"
    max_epochs = 50
    batch_size = 16
    dropout1 = 0.5
    dropout2 = 0.8
    lr = 0.0015
    lr_embd = 0.1
    l2 = 0
    diff_lr = False
    trainable = True
    name = "ASTD"

    model_name = MODEL_STR % (lr, l2, dropout1, dropout2, batch_size)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tree_path", required=True)
    parser.add_argument("-d", "--dataset", default="ATT", required=False)
    parser.add_argument("-b", "--batch", default=1, required=False)
    parser.add_argument("-p", "--data_path", default="../data", required=False)
    parser.add_argument("-m", "--model_path", default=None, required=False)

    args = parser.parse_args()

    config = Config()

    name = args.dataset
    config.batch_size = int(args.batch)
    data_path = args.data_path
    tree_path = args.tree_path

    config.data_path = os.path.join(
        data_path, "{}-balanced-not-linked.csv".format(name)
    )
    config.trees_path = os.path.join(data_path, "trees/{}".format(name))
    config.pre_trained_v_path = os.path.join(
        os.path.dirname(data_path), "pre_trained/cbow_300/{}/all_vocab/vectors.npy"
    ).format(name)
    config.pre_trained_i_path = os.path.join(
        os.path.dirname(data_path), "pre_trained/cbow_300/{}/all_vocab/w2indx.pkl"
    ).format(name)

    pickle_file = os.path.join(data_path, "generated_trees/{}.pkl".format(name))
    if os.path.isfile(pickle_file):
        f = open(pickle_file, "rb")
        data = pickle.load(f)
    else:
        data = treeDS.load_shrinked_trees(config.trees_path, config.data_path)
        f = open(pickle_file, "wb")
        pickle.dump(data, f)

    train_perc = int(len(data) * 0.9)
    train_data = data[:train_perc]
    dev_data = data[train_perc:]

    model = TreeCNN(config, train_data, dev_data)
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = SAVE_DIR + "%s.temp" % model.config.model_name
        print(model_path)
    pred, logits = model.predict_example(tree_path, model_path)
    print("Results: ", pred, logits)
