import pickle
from collections import OrderedDict, defaultdict

import tensorflow as tf

import treeDS
from train_utils import generate_batch


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
    }

    return feed_dict


graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    latest_checkpoint = tf.train.latest_checkpoint("../weights")
    saver = tf.train.import_meta_graph(latest_checkpoint + ".meta")
    saver.restore(sess, latest_checkpoint)
    graph = tf.get_default_graph()
    graph_op = graph.get_operations()

    tree_path = "../data/trees/HTL_30/0/ex0.txt.out"
    pickle_file = "../weights/dev_data.pkl"
    f = open(pickle_file, "rb")
    dev_data = pickle.load(f, encoding="utf-8")

    t = treeDS.load_tree(tree_path)
    batch_generator = generate_batch(dev_data, 32)

    pickle_file = "../weights/vocab.pkl"
    f = open(pickle_file, "rb")
    vocab = pickle.load(f, encoding="utf-8")

    feed_dict = build_feed_dict(next(batch_generator), vocab)
    preds = sess.run(["scores:0", "root_prediction:0"], feed_dict=feed_dict)
    print("res", preds)
