import pickle
import argparse
import os
from collections import OrderedDict, defaultdict

import numpy as np
import tensorflow as tf

import treeDS
from train_utils import generate_batch
from metrics import get_metrics


def build_feed_dict(trees, vocab):
    nodes_list = []
    for tree in trees:
        treeDS.traverse(tree.root, lambda node, args: args.append(node), [nodes_list])

    node_to_index = OrderedDict()
    for i in range(len(nodes_list)):
        node_to_index[nodes_list[i]] = i

    feed_dict = {
        "Input/dropout1_placeholder:0": 1.0,
        "Input/dropout2_placeholder:0": 1.0,
        "Input/roots_placeholder:0": [
            node_to_index[node] for node in nodes_list if node.isRoot
        ],
        "Input/node_level_placeholder:0": [node.level for node in nodes_list],
        "Input/max_children_placholder:0": [node.max_nodes for node in nodes_list],
        "Input/children_placeholder:0": [
            " ".join([str(node_to_index[child]) for child in node.c])
            for node in nodes_list
        ],
        "Input/node_word_indices_placeholder:0": [
            vocab.encode(node.word) if node.word else -1 for node in nodes_list
        ],
        "Input/n_examples_placeholder:0": len(trees),
        "Input/labels_placeholder:0": [t.label for t in trees],
    }

    return feed_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32, required=False)

    args = parser.parse_args()

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        latest_checkpoint = tf.train.latest_checkpoint(args.weights_path)
        saver = tf.train.import_meta_graph(latest_checkpoint + ".meta")
        saver.restore(sess, latest_checkpoint)
        graph = tf.get_default_graph()
        graph_op = graph.get_operations()

        f = open(args.data_path, "rb")
        data = pickle.load(f, encoding="utf-8")

        batch_generator = generate_batch(data, args.batch_size, one_epoch=True)

        f = open(os.path.join(args.weights_path, "vocab.pkl"), "rb")
        vocab = pickle.load(f, encoding="utf-8")

        logits_all, preds_all = None, None
        for batch in batch_generator:
            feed_dict = build_feed_dict(batch, vocab)
            logits, preds = sess.run(
                ["scores:0", "root_prediction:0"], feed_dict=feed_dict
            )
            if preds_all is not None:
                logits_all = np.append(logits_all, logits, axis=0)
                preds_all = np.append(preds_all, preds, axis=0)
            else:
                logits_all = logits
                preds_all = preds

        metrics = get_metrics(data, preds_all, logits_all)
        print(f"The predictions are\n{preds_all}")
        print(metrics)
