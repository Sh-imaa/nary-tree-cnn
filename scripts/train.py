import os, sys, shutil, time, itertools
import math, random
from collections import OrderedDict, defaultdict

import pickle
import numpy as np
import argparse
import json
from addict import Dict

import tensorflow as tf
from tensorflow.python.ops import variable_scope, init_ops
import wandb

import utils
import treeDS
from nary_tree_cnn import TreeCNN
from trainer import Trainer

MODEL_STR = "tree_cnn_lr=%f_l2=%f_dr1=%f_dr2=%f_batch_size=%d.weights"
SAVE_DIR = "../weights/"
LOG_DIR = "./logs/"

wandb.init(project="tree_cnn", entity="shimaa")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ATT", required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--max_epochs", type=int, default=2, required=False)
    parser.add_argument("--data_path", type=str, default="../data", required=False)
    parser.add_argument("--save_dir", type=str, default="../weights", required=False)
    parser.add_argument(
        "--config_path", type=str, default="../configs/config.json", required=False
    )
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    config = Dict(json.load(open(args.config_path)))
    config.update(args.__dict__)
    config.model_name = MODEL_STR % (
        config.lr,
        config.l2,
        config.dropout1,
        config.dropout2,
        config.batch_size,
    )

    name = args.dataset
    data_path = args.data_path
    save_dir = args.save_dir
    # TODO: change this
    SAVE_DIR = save_dir

    config.dataset_path = os.path.join(
        config.data_path, "{}-balanced-not-linked.csv".format(name)
    )
    config.trees_path = os.path.join(config.data_path, "trees/{}".format(name))
    config.pre_trained_v_path = os.path.join(
        os.path.dirname(config.data_path),
        "pre_trained/cbow_300/{}/all_vocab/vectors.npy",
    ).format(name)
    config.pre_trained_i_path = os.path.join(
        os.path.dirname(config.data_path),
        "pre_trained/cbow_300/{}/all_vocab/w2indx.pkl",
    ).format(name)

    pickle_file = os.path.join(config.data_path, "generated_trees/{}.pkl".format(name))
    if os.path.isfile(pickle_file):
        f = open(pickle_file, "rb")
        data = pickle.load(f, encoding="utf-8")
    else:
        data = treeDS.load_shrinked_trees(config.trees_path, config.dataset_path)
        f = open(pickle_file, "wb")
        pickle.dump(data, f)

    train_perc = int(len(data) * 0.9)
    train_data = data[:train_perc]
    dev_data = data[train_perc:]
    f = open(os.path.join(args.save_dir, "dev_data.pkl"), "wb")
    pickle.dump(dev_data, f)
    f = open(os.path.join(args.save_dir, "train_data.pkl"), "wb")
    pickle.dump(train_data, f)

    if config.bucketing:
        train_lens = [(len(t.get_words()), t) for t in train_data]
        train_lens.sort(key=lambda x: x[0])
        train_data = [t for i, t in train_lens]
        del train_lens

    vocab = utils.Vocab_pre_trained_big(
        config.pre_trained_v_path,
        config.pre_trained_i_path,
        arabic=True,
    )

    f = open(os.path.join(args.save_dir, "vocab.pkl"), "wb")
    pickle.dump(vocab, f)

    config.embed_size = vocab.pre_trained_embeddings.shape[1]
    model = TreeCNN(config, vocab)
    trainer = Trainer(model, config, train_data, dev_data)

    start_time = time.time()
    stats = trainer.train(verbose=True)
    end_time = time.time()
    print("Done")
    print(
        "Time for training is",
        ((end_time - start_time) / 60),
        "mins",
    )
