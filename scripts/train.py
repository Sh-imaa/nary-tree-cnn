import os, sys, shutil, time, itertools
import math, random
from collections import OrderedDict, defaultdict

import pickle
import numpy as np
import argparse

import tensorflow as tf
from tensorflow.python.ops import variable_scope, init_ops
import wandb

import utils
import treeDS
from nary_tree_cnn import TreeCNN, Config

MODEL_STR = "tree_cnn_lr=%f_l2=%f_dr1=%f_dr2=%f_batch_size=%d.weights"
SAVE_DIR = "../weights/"
LOG_DIR = "./logs/"

wandb.init(project="tree_cnn", entity="shimaa")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="ATT", required=False)
    parser.add_argument("-b", "--batch", default=32, required=False)
    parser.add_argument("-e", "--epoch", default=2, required=False)
    parser.add_argument("-p", "--data_path", default="../data", required=False)
    parser.add_argument("-k", "--bucketing", default="true", required=False)
    parser.add_argument("-s", "--save_dir", default="../weights", required=False)
    args = parser.parse_args()

    config = Config()

    name = args.dataset
    config.batch_size = int(args.batch)
    config.max_epochs = int(args.epoch)
    data_path = args.data_path
    save_dir = args.save_dir
    # TODO: change this
    SAVE_DIR = save_dir

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
        data = pickle.load(f, encoding="utf-8")
    else:
        data = treeDS.load_shrinked_trees(config.trees_path, config.data_path)
        f = open(pickle_file, "wb")
        pickle.dump(data, f)

    train_perc = int(len(data) * 0.9)
    train_data = data[:train_perc]
    dev_data = data[train_perc:]
    test_data = None

    if args.bucketing == "true":
        train_lens = [(len(t.get_words()), t) for t in train_data]
        train_lens.sort(key=lambda x: x[0])
        train_data = [t for i, t in train_lens]
        del train_lens
    model = TreeCNN(config, train_data, dev_data, test_data)

    start_time = time.time()
    stats = model.train(verbose=True)
    end_time = time.time()
    print("Done")
    print(
        "Time for training ",
        config.model_name,
        " is ",
        ((end_time - start_time) / 60),
        "mins",
    )
