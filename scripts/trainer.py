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
from train_utils import generate_batch, copy_weight_files

MODEL_STR = "tree_cnn_lr=%f_l2=%f_dr1=%f_dr2=%f_batch_size=%d.weights"
SAVE_DIR = "../weights/"
LOG_DIR = "./logs/"


class Trainer:
    def __init__(self, model):
        self.model = model

    def build_feed_dict(self, trees, train=True, no_label=False):
        nodes_list = []
        for tree in trees:
            treeDS.traverse(
                tree.root, lambda node, args: args.append(node), [nodes_list]
            )

        node_to_index = OrderedDict()
        for i in range(len(nodes_list)):
            node_to_index[nodes_list[i]] = i
        if train:
            dropout1 = self.model.config.dropout1
            dropout2 = self.model.config.dropout2
        else:
            dropout1 = 1.0
            dropout2 = 1.0
        feed_dict = {
            self.model.dropout1_placeholder: dropout1,
            self.model.dropout2_placeholder: dropout2,
            self.model.roots_placeholder: [
                node_to_index[node] for node in nodes_list if node.isRoot
            ],
            self.model.node_level_placeholder: [node.level for node in nodes_list],
            self.model.max_children_placeholder: [
                node.max_nodes for node in nodes_list
            ],
            self.model.children_placeholder: [
                " ".join([str(node_to_index[child]) for child in node.c])
                for node in nodes_list
            ],
            self.model.node_word_indices_placeholder: [
                self.model.vocab.encode(node.word) if node.word else -1
                for node in nodes_list
            ],
            self.model.n_examples_placeholder: len(trees),
        }
        if not no_label:
            feed_dict[self.model.labels_placeholder] = [t.label for t in trees]
        return feed_dict

    def predict(self, weights_path, get_pred_only=True, dataset=None, trees=None):
        """Make predictions from the provided model."""
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, weights_path)

            if dataset is None:
                if trees is None:
                    raise ValueError(
                        "You need to provide the trees or the name of the dataset."
                    )
                batch_generator = generate_batch(trees, len(trees))
                feed_dict = self.build_feed_dict(next(batch_generator), train=False)
            elif dataset == "train":
                feed_dict = self.model.feed_dict_train
            elif dataset == "val":
                feed_dict = self.model.feed_dict_dev
            if get_pred_only:
                root_prediction = sess.run(
                    [self.model.root_prediction],
                    feed_dict=feed_dict,
                )
                return root_prediction
            else:
                root_prediction, loss, root_acc, root_perc = sess.run(
                    [
                        self.model.root_prediction,
                        self.model.loss,
                        self.model.root_acc,
                        self.model.root_percision,
                    ],
                    feed_dict=feed_dict,
                )
                return root_prediction, loss, root_acc, root_perc

    def predict_example(self, tree_path, weights_path):
        t = treeDS.load_tree(tree_path)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, weights_path)
            batch_generator = generate_batch([t], 1)
            feed_dict = self.build_feed_dict(next(batch_generator), train=False)
            root_prediction, logits = sess.run(
                [self.model.root_prediction, self.model.scores], feed_dict=feed_dict
            )
            return root_prediction, logits

    def run_epoch(self, batches, new_model=False, verbose=True):
        random.shuffle(batches)
        with tf.Session() as sess:
            if new_model:
                sess.run(tf.global_variables_initializer())
            else:
                saver = tf.train.Saver()
                saver.restore(sess, SAVE_DIR + "%s.temp" % self.model.config.model_name)

            m = len(self.model.train_data)
            num_batches = int(m / self.model.config.batch_size) + 1
            last_loss = float("inf")
            total_time = 0
            for batch in range(num_batches):
                t1_ = time.time()
                feed_dict = batches[batch]
                loss_value, acc, _ = sess.run(
                    [self.model.full_loss, self.model.root_acc, self.model.train_op],
                    feed_dict=feed_dict,
                )
                if verbose:
                    sys.stdout.write(
                        "\r{} / {} :    loss = {} and acc = {}".format(
                            batch, num_batches, loss_value, acc
                        )
                    )
                    sys.stdout.flush()
            saver = tf.train.Saver()
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            saver.save(sess, SAVE_DIR + "%s.temp" % self.model.config.model_name)

    def train(self, verbose=True):
        stopped = -1
        best_val_epoch = 0
        best_dev_loss = float("inf")
        best_dev_acc = 0

        batch_generator = generate_batch(self.model.dev_data, len(self.model.dev_data))
        self.model.feed_dict_dev = self.build_feed_dict(
            next(batch_generator), train=False
        )
        batch_generator_train = generate_batch(
            self.model.train_data, len(self.model.train_data)
        )
        self.model.feed_dict_train = self.build_feed_dict(
            next(batch_generator_train), train=False
        )

        batches = []
        batch_generator = generate_batch(
            self.model.train_data, self.model.config.batch_size
        )
        num_batches = int(len(self.model.train_data) / self.model.config.batch_size) + 1
        for batch in range(num_batches):
            batches.append(self.build_feed_dict(next(batch_generator)))

        for epoch in range(self.model.config.max_epochs):
            print("\nepoch %d" % epoch)
            if epoch == 0:
                self.run_epoch(batches=batches, new_model=True)
            else:
                self.run_epoch(batches=batches)

            _, dev_loss, dev_acc, dev_prec = self.predict(
                SAVE_DIR + "%s.temp" % self.model.config.model_name,
                get_pred_only=False,
                dataset="val",
            )
            _, train_loss, train_acc, train_prec = self.predict(
                SAVE_DIR + "%s.temp" % self.model.config.model_name,
                get_pred_only=False,
                dataset="train",
            )
            print("\nDev loss : {} --- dev acc: {}".format(dev_loss, dev_acc))
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "dev_loss": dev_loss,
                    "dev_acc": dev_acc,
                    "dev_prec": dev_prec,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "train_prec": train_prec,
                }
            )

            if dev_acc > best_dev_acc:
                copy_weight_files(
                    SAVE_DIR,
                    self.model.config.model_name,
                    self.model.config.model_name + ".temp",
                )
                best_dev_acc = dev_acc
                best_dev_epoch = epoch

            if (dev_acc == best_dev_acc) and (dev_loss < best_dev_loss):
                copy_weight_files(
                    SAVE_DIR,
                    self.model.config.model_name,
                    self.model.config.model_name + ".temp",
                )
                copy_weight_files(
                    SAVE_DIR,
                    self.model.config.model_name + ".loss",
                    self.model.config.model_name + ".temp",
                )
                best_dev_loss = dev_loss
                best_dev_epoch = epoch

            if dev_loss < best_dev_loss:
                copy_weight_files(
                    SAVE_DIR,
                    self.model.config.model_name + ".loss",
                    self.model.config.model_name + ".temp",
                )
                best_dev_loss = dev_loss

            if (epoch - best_dev_epoch) > self.model.config.early_stopping:
                stopped = epoch
                break

        if verbose:
            sys.stdout.write("\r")
            sys.stdout.flush()

        print("\n\nstopped at %d\n" % stopped)
        print("best loss: ", best_dev_loss)
        print("best acc: ", best_dev_acc)
        # TODO: log best values

        return {"best_acc": best_dev_acc, "best_loss": best_dev_loss}
