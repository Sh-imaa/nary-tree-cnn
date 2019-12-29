import os, sys, shutil, time, itertools
import math, random, argparse
from collections import OrderedDict, defaultdict

import pickle
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import variable_scope, init_ops

import utils
import treeDS

MODEL_STR = 'tree_cnn_lr=%f_l2=%f_dr1=%f_dr2=%f_batch_size=%d.weights'
SAVE_DIR = '../weights/'
LOG_DIR = './logs/'

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

        for level, nodes in node_by_level.iteritems():
            max_nodes[level] = max([len(n.c) for n in nodes])
        
        for tree in new_batch:
          treeDS.get_max_nodes(tree.root, max_nodes)
          treeDS.pad(tree.root)
        yield new_batch

class Config(object):
  """Holds model hyperparams and data information.
  Model objects are passed a Config() object at instantiation.
  """
  optimizer = 'Adam'
  embed_size = 44
  label_size = 2
  early_stopping = 20
  act_fun = 'relu'
  max_epochs = 50
  batch_size = 16
  dropout1 = 0.5
  dropout2 = 0.8
  lr = 0.0015
  lr_embd = 0.1
  l2 = 0 
  diff_lr = False
  trainable = True
  name = 'ASTD'

  model_name = MODEL_STR % (lr, l2, dropout1, dropout2, batch_size)

class TreeCNN():

  def __init__(self, config, train_data, dev_data, test_data=None):
    self.config = config
    self.train_data = train_data
    self.dev_data = dev_data
    self.test_data = test_data

    print 'Training on {} examples, validating on {} examples.'.format(
      len(self.train_data), len(self.dev_data))
    
    self.var_list = []
    self.epoch_size = len(self.train_data)
    # Load train data and build vocabulary
    self.vocab = utils.Vocab_pre_trained_big(self.config.pre_trained_v_path,
      self.config.pre_trained_i_path,
      arabic=True)
    config.embed_size = self.vocab.pre_trained_embeddings.shape[1]

    # add input placeholders
    self.dropout1_placeholder = tf.placeholder(
        tf.float32, (None), name='dropout1_placeholder')
    self.dropout2_placeholder = tf.placeholder(
        tf.float32, (None), name='dropout2_placeholder')
    self.is_leaf_placeholder = tf.placeholder(
        tf.bool, (None), name='is_leaf_placeholder')
    self.children_placeholder = tf.placeholder(
        tf.string, (None), name='children_placeholder')
    self.node_word_indices_placeholder = tf.placeholder(
        tf.int32, (None), name='node_word_indices_placeholder')
    self.labels_placeholder = tf.placeholder(
        tf.int32, (None), name='labels_placeholder')

    # add model variables
    with tf.variable_scope('Embeddings'):
      embeddings = tf.Variable(initial_value=self.vocab.pre_trained_embeddings,
                    trainable=self.config.trainable,
                    name="embeddings",
                    dtype=tf.float32)

    with tf.variable_scope('Composition'):
      filters = tf.get_variable('filters', [2, self.config.embed_size, self.config.embed_size])
      b = tf.get_variable('b', [self.config.embed_size])
      self.var_list.extend([filters, b])


    with tf.variable_scope('Projection'):
      U = tf.get_variable('U', [self.config.embed_size, self.config.label_size],
                          initializer=tf.initializers.truncated_normal(stddev=0.1))
      bs = tf.get_variable('bs', [1, self.config.label_size], initializer=tf.constant_initializer(0.1))
      self.var_list.extend([U, bs])

    # build recursive graph

    tensor_array = tf.TensorArray(
        tf.float32,
        size=0,
        dynamic_size=True,
        clear_after_read=False,
        infer_shape=False)
            
    def embed_word(word_index):
      return tf.expand_dims(tf.gather(embeddings, word_index), 0)

    def combine_children(children_tensors):
      children_tensors = tf.reshape(children_tensors, [1, -1, self.config.embed_size])
      conv = tf.nn.conv1d(children_tensors, filters, stride=1, padding="SAME")
      h = tf.nn.relu(tf.nn.bias_add(conv, b))
      pooled  = tf.reduce_max(h, axis=1, keepdims=False)

      return tf.nn.dropout(pooled, self.dropout1_placeholder)

    def loop_body(tensor_array, i):
      node_is_leaf = tf.gather(self.is_leaf_placeholder, i)
      node_word_index = tf.gather(self.node_word_indices_placeholder, i)
      children = [tf.gather(self.children_placeholder, i)]
      children_values = tf.to_int32(tf.string_to_number(tf.string_split(children).values))
      node_tensor = tf.cond(
          node_is_leaf,
          lambda: embed_word(node_word_index),
          lambda: combine_children(tensor_array.gather(children_values)))
      tensor_array = tensor_array.write(i, node_tensor)
      i = tf.add(i, 1)
      return tensor_array, i

    loop_cond = lambda tensor_array, i: \
        tf.less(i, tf.squeeze(tf.shape(self.is_leaf_placeholder)))
    self.tensor_array, _ = tf.while_loop(loop_cond, loop_body,
                                         [tensor_array, 0],
                                         parallel_iterations=1)

    # add projection layer
    self.logits = tf.matmul(self.tensor_array.concat(), U) + bs
    self.root_logits = tf.matmul(
        self.tensor_array.read(self.tensor_array.size() - 1), U) + bs
    self.root_prediction = tf.squeeze(tf.argmax(self.root_logits, 1))

    # add loss layer
    regularization_loss = self.config.l2 * (
        tf.nn.l2_loss(filters) + tf.nn.l2_loss(U))
    included_indices = tf.where(tf.less(self.labels_placeholder, 2))
    self.full_loss = regularization_loss + tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.gather(self.logits, included_indices),labels=tf.gather(
                self.labels_placeholder, included_indices)))
    self.root_loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.root_logits,labels=self.labels_placeholder[-1:]))

    # add training op
    self.train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(
        self.full_loss)

  def build_feed_dict(self, node, label, train=True):
    nodes_list = [] 
    treeDS.traverse(node, lambda node, args: args.append(node), [nodes_list])
    node_to_index = OrderedDict()
    for i in xrange(len(nodes_list)):
      node_to_index[nodes_list[i]] = i
    if train:
      dropout1 = self.config.dropout1
      dropout2 = self.config.dropout2
    else:
      dropout1 = 1.0
      dropout2 = 1.0

    feed_dict = {
        self.dropout1_placeholder: dropout1,
        self.dropout2_placeholder: dropout2,
        self.is_leaf_placeholder: [node.isLeaf for node in nodes_list],
        self.node_word_indices_placeholder: [self.vocab.encode(node.word) if
                                             node.word else -1
                                             for node in nodes_list],
        self.children_placeholder: [' '.join([str(node_to_index[child]) for child in node.c])
                                  for node in nodes_list],
        self.labels_placeholder: [label],
    }
    return feed_dict

  def predict(self, trees, weights_path, get_loss=False):
    """Make predictions from the provided model."""
    results = []
    losses = []
    with tf.Session() as sess:
      saver = tf.train.Saver()
      saver.restore(sess, weights_path)
      for tree in trees:
        feed_dict = self.build_feed_dict(tree.root, tree.label, train=False)
        if get_loss:
          root_prediction, loss = sess.run(
              [self.root_prediction, self.root_loss], feed_dict=feed_dict)
          losses.append(loss)
        else:
          root_prediction = sess.run(self.root_prediction, feed_dict=feed_dict)
        results.append(root_prediction)
    return results, losses

  def run_epoch(self, new_model=False, verbose=True):
    loss_history = []
    # training
    random.shuffle(self.train_data)
    with tf.Session() as sess:
      if new_model:
        sess.run(tf.initialize_all_variables())
      else:
        saver = tf.train.Saver()
        saver.restore(sess, SAVE_DIR + '%s.temp' % self.config.model_name)
      for step, tree in enumerate(self.train_data):
        feed_dict = self.build_feed_dict(tree.root, tree.label)
        loss_value, _ = sess.run([self.full_loss, self.train_op],
                                 feed_dict=feed_dict)
        loss_history.append(loss_value)
        if verbose:
          sys.stdout.write('\r{} / {} :    loss = {}'.format(step, len(
              self.train_data), np.mean(loss_history)))
          sys.stdout.flush()
      saver = tf.train.Saver()
      if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
      saver.save(sess, SAVE_DIR + '%s.temp' % self.config.model_name)
    # statistics
    train_preds, _ = self.predict(self.train_data,
                                  SAVE_DIR + '%s.temp' % self.config.model_name)
    val_preds, val_losses = self.predict(
        self.dev_data,
        SAVE_DIR + '%s.temp' % self.config.model_name,
        get_loss=True)
    train_labels = [t.root.label for t in self.train_data]
    val_labels = [t.root.label for t in self.dev_data]
    train_acc = np.equal(train_preds, train_labels).mean()
    val_acc = np.equal(val_preds, val_labels).mean()

    print
    print 'Training acc (only root node): {}'.format(train_acc)
    print 'Valiation acc (only root node): {}'.format(val_acc)
    print self.make_conf(train_labels, train_preds)
    print self.make_conf(val_labels, val_preds)
    return train_acc, val_acc, loss_history, np.mean(val_losses)

  def train(self, verbose=True):
    complete_loss_history = []
    train_acc_history = []
    val_acc_history = []
    prev_epoch_loss = float('inf')
    best_val_loss = float('inf')
    best_val_epoch = 0
    stopped = -1
    for epoch in xrange(self.config.max_epochs):
      print 'epoch %d' % epoch
      if epoch == 0:
        train_acc, val_acc, loss_history, val_loss = self.run_epoch(
            new_model=True)
      else:
        train_acc, val_acc, loss_history, val_loss = self.run_epoch()
      complete_loss_history.extend(loss_history)
      train_acc_history.append(train_acc)
      val_acc_history.append(val_acc)

      #lr annealing
      epoch_loss = np.mean(loss_history)
      if epoch_loss > prev_epoch_loss * self.config.anneal_threshold:
        self.config.lr /= self.config.anneal_by
        print 'annealed lr to %f' % self.config.lr
      prev_epoch_loss = epoch_loss

      #save if model has improved on val
      if val_loss < best_val_loss:
        shutil.copyfile(SAVE_DIR + '%s.temp' % self.config.model_name,
                        SAVE_DIR + '%s' % self.config.model_name)
        best_val_loss = val_loss
        best_val_epoch = epoch

      # if model has not imprvoved for a while stop
      if epoch - best_val_epoch > self.config.early_stopping:
        stopped = epoch
        #break
    if verbose:
      sys.stdout.write('\r')
      sys.stdout.flush()

    print '\n\nstopped at %d\n' % stopped
    return {
        'loss_history': complete_loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
    }

  def make_conf(self, labels, predictions):
    confmat = np.zeros([2, 2])
    for l, p in itertools.izip(labels, predictions):
      confmat[l, p] += 1
    return confmat

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', "--dataset", default="ATT", required=False)
  parser.add_argument('-b', "--batch", default=32, required=False)
  parser.add_argument('-e', "--epoch", default=2, required=False)
  parser.add_argument('-p', "--data_path", default='../data', required=False)
  args = parser.parse_args()

  config = Config()

  name = args.dataset
  config.batch_size = args.batch
  config.max_epochs = args.epoch
  data_path = args.data_path
  
  config.data_path = os.path.join(data_path, '{}-balanced-not-linked.csv'.format(name))  
  config.trees_path = os.path.join(data_path, 'trees/{}'.format(name))
  config.pre_trained_v_path = os.path.join(os.path.dirname(data_path),
    'pre_trained/cbow_300/{}/all_vocab/vectors.npy').format(name)
  config.pre_trained_i_path = os.path.join(os.path.dirname(data_path),
    'pre_trained/cbow_300/{}/all_vocab/w2indx.pkl').format(name)

  pickle_file = os.path.join(data_path, 'generated_trees/{}.pkl'.format(name))
  if os.path.isfile(pickle_file):
    f = open(pickle_file,'r')
    data = pickle.load(f)
  else:
    data = treeDS.load_shrinked_trees(config.trees_path, config.data_path)
    f = open(pickle_file, "wb")
    pickle.dump(data, f)
  
  train_perc = int(len(data) * 0.9)
  train_data = data[:train_perc]
  dev_data = data[train_perc:]
  test_data = None

  model = TreeCNN(config, train_data, dev_data, test_data)

  start_time = time.time()
  stats = model.train(verbose=True)
  end_time = time.time()
  print 'Done'
  print 'Time for training ', config.model_name, ' is ', ((end_time - start_time) / 60), 'mins'   