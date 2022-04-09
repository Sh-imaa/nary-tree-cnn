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


SAVE_DIR = "../weights/"
LOG_DIR = "./logs/"

wandb.init(project="tree_cnn", entity="shimaa")


class TreeCNN:
    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab

        self.var_list = []

        # add input placeholders
        with tf.variable_scope("Input"):
            self.dropout1_placeholder = tf.placeholder(
                tf.float32, (None), name="dropout1_placeholder"
            )
            self.dropout2_placeholder = tf.placeholder(
                tf.float32, (None), name="dropout2_placeholder"
            )
            self.roots_placeholder = tf.placeholder(
                tf.int32, (None), name="roots_placeholder"
            )
            self.node_level_placeholder = tf.placeholder(
                tf.int32, (None), name="node_level_placeholder"
            )
            self.max_children_placeholder = tf.placeholder(
                tf.int32, (None), name="max_children_placholder"
            )
            self.children_placeholder = tf.placeholder(
                tf.string, (None), name="children_placeholder"
            )
            self.node_word_indices_placeholder = tf.placeholder(
                tf.int32, (None), name="node_word_indices_placeholder"
            )
            self.labels_placeholder = tf.placeholder(
                tf.int64, (None), name="labels_placeholder"
            )
            self.n_examples_placeholder = tf.placeholder(
                tf.float32, (None), name="n_examples_placeholder"
            )

            # add model variables

        with tf.variable_scope("Embeddings"):
            self.embeddings = tf.Variable(
                initial_value=self.vocab.pre_trained_embeddings,
                trainable=self.config.trainable,
                name="embeddings",
                dtype=tf.float32,
            )

        with tf.variable_scope("Composition"):
            filters = tf.get_variable(
                "filters", [2, self.config.embed_size, self.config.embed_size]
            )
            b = tf.get_variable("b", [self.config.embed_size])
            self.var_list.extend([filters, b])

        with tf.variable_scope("Projection"):
            U = tf.get_variable(
                "U",
                [self.config.embed_size, self.config.label_size],
                initializer=tf.initializers.truncated_normal(stddev=0.1),
            )
            bs = tf.get_variable(
                "bs",
                [1, self.config.label_size],
                initializer=tf.constant_initializer(0.1),
            )
            self.var_list.extend([U, bs])

        # build recursive graph
        tensor_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False,
        )

        def embed_words(word_indeces):
            words = tf.gather_nd(self.embeddings, tf.reshape(word_indeces, [-1, 1]))
            return tf.nn.dropout(words, self.dropout1_placeholder)

        def combine_children(children_tensors, m):
            children_tensors = tf.reshape(
                children_tensors, [-1, m, self.config.embed_size]
            )
            conv = tf.nn.conv1d(children_tensors, filters, stride=1, padding="SAME")
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            pooled = tf.reduce_max(h, axis=1, keepdims=False)

            return tf.nn.dropout(pooled, self.dropout1_placeholder)

        def loop_body(tensor_array, i):
            ind = tf.where(tf.equal(self.node_level_placeholder, i))
            node_word_index = tf.gather_nd(self.node_word_indices_placeholder, ind)
            children = tf.gather_nd(self.children_placeholder, ind)
            children_values = tf.to_int32(
                tf.string_to_number(tf.string_split(children).values)
            )
            max_nodes = tf.gather_nd(self.max_children_placeholder, ind)[0]
            node_tensor = tf.cond(
                tf.equal(i, 0),
                lambda: embed_words(node_word_index),
                lambda: combine_children(
                    tensor_array.gather(children_values), max_nodes
                ),
            )

            tensor_array = tensor_array.scatter(
                tf.cast(tf.reshape(ind, [-1]), tf.int32), node_tensor
            )
            i = tf.add(i, 1)
            return tensor_array, i

        max_level = tf.reduce_max(self.node_level_placeholder)
        loop_cond = lambda tensor_array, i: tf.less(i, tf.add(max_level, 1))
        self.tensor_array, _ = tf.while_loop(
            loop_cond,
            loop_body,
            [tensor_array, tf.constant(0, dtype=tf.int32)],
            parallel_iterations=1,
        )
        self.roots = self.tensor_array.gather(self.roots_placeholder)

        # add projection layer
        self.logits = tf.matmul(self.roots, U) + bs
        self.logits = tf.nn.dropout(self.logits, self.dropout2_placeholder)
        self.scores = tf.nn.softmax(self.logits, name="scores")
        self.root_prediction = tf.squeeze(
            tf.argmax(self.logits, 1), name="root_prediction"
        )
        # add loss layer
        regularization_loss = self.config.l2 * (
            tf.reduce_sum(tf.math.abs(filters)) + tf.reduce_sum(tf.math.abs(U))
        )

        self.loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels_placeholder
            )
        )
        self.loss = tf.divide(self.loss, self.n_examples_placeholder)

        self.full_loss = regularization_loss + self.loss
