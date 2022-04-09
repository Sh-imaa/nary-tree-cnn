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
from train_utils import generate_batch
from metrics import get_metrics


MODEL_STR = "tree_cnn_lr=%f_l2=%f_dr1=%f_dr2=%f_batch_size=%d.weights"
LOG_DIR = "./logs/"


class Trainer:
    def __init__(self, model, config, train_data, dev_data, wandb_name=""):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.dev_data = dev_data
        self.wandb_name = wandb_name

        print(
            "Training on {} examples, validating on {} examples.".format(
                len(self.train_data), len(self.dev_data)
            )
        )
        self.epoch_size = len(self.train_data)

        self.define_train_op()

    def define_train_op(self):
        if self.config.diff_lr:
            global_step = variable_scope.get_variable(
                "global_step",
                [],
                trainable=False,
                dtype=tf.int64,
                initializer=init_ops.constant_initializer(0, dtype=tf.int64),
            )
            train_op1 = tf.train.AdagradOptimizer(self.config.lr_embd).minimize(
                self.model.full_loss, var_list=[self.model.embeddings]
            )
            train_op2 = tf.train.AdagradOptimizer(self.config.lr).minimize(
                self.model.full_loss, var_list=self.model.var_list
            )
            self.train_op = tf.group(train_op1, train_op2)

        elif self.config.optimizer == "Adam":
            self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(
                self.model.full_loss
            )
        else:
            self.train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(
                self.model.full_loss
            )

    def build_feed_dict(self, trees, train=True, with_label=True):
        nodes_list = []
        for tree in trees:
            treeDS.traverse(
                tree.root, lambda node, args: args.append(node), [nodes_list]
            )

        node_to_index = OrderedDict()
        for i in range(len(nodes_list)):
            node_to_index[nodes_list[i]] = i
        if train:
            dropout1 = self.config.dropout1
            dropout2 = self.config.dropout2
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
        if with_label:
            feed_dict[self.model.labels_placeholder] = [t.label for t in trees]
        return feed_dict

    def predict(self, sess, get_loss=True, dataset=None, trees=None):
        """Make predictions from the provided model."""
        if dataset is None:
            if trees is None:
                raise ValueError(
                    "You need to provide the trees or the name of the dataset."
                )
            batch_generator = generate_batch(trees, len(trees))
            feed_dict = self.build_feed_dict(next(batch_generator), train=False)
        elif dataset == "train":
            feed_dict = self.feed_dict_train
        elif dataset == "val":
            feed_dict = self.feed_dict_dev
        if not get_loss:
            logits, root_prediction = sess.run(
                [self.model.scores, self.model.root_prediction],
                feed_dict=feed_dict,
            )
            return root_prediction, logits
        else:
            logits, root_prediction, loss = sess.run(
                [self.model.scores, self.model.root_prediction, self.model.loss],
                feed_dict=feed_dict,
            )
            return root_prediction, loss, logits

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

    def run_epoch(self, sess, batches, verbose=True):
        random.shuffle(batches)
        m = len(self.train_data)
        num_batches = int(m / self.config.batch_size) + 1
        last_loss = float("inf")
        total_time = 0
        for batch in range(num_batches):
            t1_ = time.time()
            feed_dict = batches[batch]
            loss_value, _ = sess.run(
                [self.model.full_loss, self.train_op],
                feed_dict=feed_dict,
            )
            if verbose:
                sys.stdout.write(
                    "\r{} / {} :    loss = {}".format(batch, num_batches, loss_value)
                )
                sys.stdout.flush()
        saver = tf.train.Saver()

    def train(self, new_model=True, verbose=True):
        stopped = -1
        best_val_epoch = 0
        best_dev_loss = float("inf")
        best_dev_acc = 0

        batch_generator = generate_batch(self.dev_data, len(self.dev_data))
        self.feed_dict_dev = self.build_feed_dict(next(batch_generator), train=False)
        batch_generator_train = generate_batch(self.train_data, len(self.train_data))
        self.feed_dict_train = self.build_feed_dict(
            next(batch_generator_train), train=False
        )

        batches = []
        batch_generator = generate_batch(self.train_data, self.config.batch_size)
        num_batches = int(len(self.train_data) / self.config.batch_size) + 1
        for batch in range(num_batches):
            batches.append(self.build_feed_dict(next(batch_generator)))
        with tf.Session() as sess:
            saver = tf.train.Saver()
            if new_model:
                sess.run(tf.global_variables_initializer())
            else:
                saver.restore(
                    sess, os.path.join(self.config.save_dir, self.config.model_name)
                )
            for epoch in range(self.config.max_epochs):
                print("\nepoch %d" % epoch)
                self.run_epoch(sess, batches=batches)

                dev_pred, dev_loss, dev_logits = self.predict(
                    sess,
                    dataset="val",
                )
                train_pred, train_loss, train_logits = self.predict(
                    sess,
                    dataset="train",
                )
                print(train_pred.shape)
                dev_metrics = get_metrics(self.dev_data, dev_pred, dev_logits)
                dev_acc = dev_metrics["acc"]
                train_metrics = get_metrics(self.train_data, train_pred, train_logits)

                print(
                    "\nDev loss : {} --- dev metrics: {}".format(dev_loss, dev_metrics)
                )
                dev_metrics = {
                    f"{self.wandb_name}dev_{k}": dev_metrics[k]
                    for k in dev_metrics.keys()
                }
                train_metrics = {
                    f"{self.wandb_name}train_{k}": train_metrics[k]
                    for k in train_metrics.keys()
                }
                logs = {}
                loss_logs = {
                    f"{self.wandb_name}epoch": epoch + 1,
                    f"{self.wandb_name}dev_loss": dev_loss,
                    f"{self.wandb_name}train_loss": train_loss,
                }
                logs.update(dev_metrics)
                logs.update(train_metrics)
                logs.update(loss_logs)
                wandb.log(logs)

                if (dev_acc > best_dev_acc) or (
                    (dev_acc == best_dev_acc) and (dev_loss < best_dev_loss)
                ):
                    saver.save(
                        sess, os.path.join(self.config.save_dir, self.config.model_name)
                    )
                    best_dev_acc = dev_acc
                    best_val_epoch = epoch
                    best_dev_loss = dev_loss

                if (epoch - best_val_epoch) > self.config.early_stopping:
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

    def get_metrics(self, data, preds):
        labels = [t.label for t in data]
        labels = np.array(labels)
        preds = np.array(preds)

        acc = (preds == labels).mean()
        tp = ((preds == labels) & (preds == 1)).sum()
        tn = ((preds == labels) & (preds == 0)).sum()
        fp = ((preds != labels) & (preds == 1)).sum()
        fn = ((preds != labels) & (preds == 0)).sum()

        perc = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_minor_1 = 2 * perc * recall / (perc + recall)
        f1_minor_0 = tn / (tn + 0.5 * (fp + fn))

        metrics = {
            "acc": acc,
            "perc": perc,
            "recall": recall,
            "f1_minor_1": f1_minor_1,
            "f1_minor_0": f1_minor_0,
        }

        return metrics
