import os, sys, shutil, time, itertools
import math, random
from collections import OrderedDict, defaultdict

import pickle
import numpy as np
import argparse

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

    print('Training on {} examples, validating on {} examples.'.format(
      len(self.train_data), len(self.dev_data)))
    
    self.var_list = []
    self.epoch_size = len(self.train_data)
    # Load train data and build vocabulary
    self.vocab = utils.Vocab_pre_trained_big(self.config.pre_trained_v_path,
      self.config.pre_trained_i_path,
      arabic=True)
    config.embed_size = self.vocab.pre_trained_embeddings.shape[1]
    # train_sents = [t.get_words() for t in self.train_data]
    # self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))

    # add input placeholders
    with tf.variable_scope('Input'):
      self.dropout1_placeholder = tf.placeholder(
        tf.float32, (None), name='dropout1_placeholder')
      self.dropout2_placeholder = tf.placeholder(
        tf.float32, (None), name='dropout2_placeholder')
      self.roots_placeholder = tf.placeholder(
        tf.int32, (None), name='roots_placeholder')
      self.node_level_placeholder = tf.placeholder(
        tf.int32, (None), name='node_level_placeholder')
      self.max_children_placeholder = tf.placeholder(
        tf.int32, (None), name='max_children_placholder')
      self.children_placeholder = tf.placeholder(
        tf.string, (None), name='children_placeholder')
      self.node_word_indices_placeholder = tf.placeholder(
        tf.int32, (None), name='node_word_indices_placeholder')
      self.labels_placeholder = tf.placeholder(
        tf.int64, (None), name='labels_placeholder')
      self.n_examples_placeholder = tf.placeholder(
        tf.float32, (None), name='n_examples_placeholder')

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


    def embed_words(word_indeces):
      words = tf.gather_nd(embeddings, tf.reshape(word_indeces, [-1, 1]))
      return tf.nn.dropout(words, self.dropout1_placeholder)
            
    def combine_children(children_tensors, m):
      children_tensors = tf.reshape(children_tensors, [-1, m, self.config.embed_size])
      conv = tf.nn.conv1d(children_tensors, filters, stride=1, padding="SAME")
      h = tf.nn.relu(tf.nn.bias_add(conv, b))
      pooled  = tf.reduce_max(h, axis=1, keepdims=False)

      return tf.nn.dropout(pooled, self.dropout1_placeholder)

        
    def loop_body(tensor_array, i):
      ind = tf.where(tf.equal(self.node_level_placeholder, i))
      node_word_index = tf.gather_nd(self.node_word_indices_placeholder, ind)
      children = tf.gather_nd(self.children_placeholder, ind)
      children_values = tf.to_int32(tf.string_to_number(tf.string_split(children).values))  
      max_nodes = tf.gather_nd(self.max_children_placeholder, ind)[0]
      node_tensor = tf.cond(
        tf.equal(i, 0),
        lambda: embed_words(node_word_index),
        lambda: combine_children(tensor_array.gather(children_values), max_nodes))

      tensor_array = tensor_array.scatter(tf.cast(
        tf.reshape(ind, [-1]), tf.int32), node_tensor)
      i = tf.add(i, 1)
      return tensor_array, i

    max_level = tf.reduce_max(self.node_level_placeholder)
    loop_cond = lambda tensor_array, i: tf.less(i, tf.add(max_level, 1))
    self.tensor_array, _ = tf.while_loop(
                                     loop_cond, loop_body,
                                     [tensor_array, tf.constant(0, dtype=tf.int32)],
                                     parallel_iterations=1)
    self.roots = self.tensor_array.gather(self.roots_placeholder)

    # add projection layer
    self.logits = tf.matmul(self.roots, U) + bs
    self.logits = tf.nn.dropout(self.logits, self.dropout2_placeholder)
    self.root_prediction = tf.squeeze(tf.argmax(self.logits, 1))
        
    # add loss layer
    regularization_loss = self.config.l2 * (
      tf.nn.l2_loss(filters) + tf.nn.l2_loss(U))
    # regularization_loss = self.config.l2 * (tf.nn.l2_loss(U))
    self.loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.labels_placeholder))
    self.loss = tf.divide(self.loss, self.n_examples_placeholder)

    self.full_loss = regularization_loss + self.loss
    self.root_acc = tf.reduce_mean(tf.cast(tf.equal(
      self.root_prediction, self.labels_placeholder), tf.float32))

    # add training op
    if self.config.diff_lr:
      global_step = variable_scope.get_variable(
        "global_step", [], trainable=False, dtype=tf.int64,
        initializer=init_ops.constant_initializer(0, dtype=tf.int64))
      train_op1 = tf.train.AdagradOptimizer(self.config.lr_embd).minimize(self.full_loss, var_list=[embeddings])
      train_op2 = tf.train.AdagradOptimizer(self.config.lr).minimize(self.full_loss, var_list=self.var_list)
      self.train_op = tf.group(train_op1, train_op2)

    elif self.config.optimizer == "Adam":
      self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(
          self.full_loss)
    else:
      self.train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(
          self.full_loss)

 
  def build_feed_dict(self, trees, train=True):
    nodes_list = []
    for tree in trees:
      treeDS.traverse(tree.root,  lambda node, args: args.append(node), [nodes_list])

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
      self.dropout1_placeholder: dropout1,
      self.dropout2_placeholder: dropout2,
      self.roots_placeholder: [node_to_index[node] for node in nodes_list if node.isRoot],
      self.node_level_placeholder: [node.level for node in nodes_list],
      self.max_children_placeholder: [node.max_nodes for node in nodes_list], 
      self.children_placeholder: [' '.join([str(node_to_index[child]) for child in node.c])
                                  for node in nodes_list],
      self.node_word_indices_placeholder: [self.vocab.encode(node.word) if
                                                         node.word else -1
                                                         for node in nodes_list],
      self.labels_placeholder: [t.label for t in trees],
      self.n_examples_placeholder : len(trees),
    }
    return feed_dict

  def predict(self, trees, weights_path, get_loss=False, dataset='test'):
    """Make predictions from the provided model."""
    with tf.Session() as sess:
      saver = tf.train.Saver()
      saver.restore(sess, weights_path)
      
      if dataset == 'test':
        batch_generator = generate_batch(trees, len(trees))
        feed_dict = self.build_feed_dict(next(batch_generator), train=False)
      elif dataset == 'train':
        feed_dict = self.feed_dict_train
      elif dataset == 'dev':
        feed_dict = self.feed_dict_dev

      if get_loss:
        root_prediction, root_loss, root_acc = sess.run(
              [self.root_prediction, self.full_loss, self.root_acc],
              feed_dict=feed_dict)
        return root_prediction, root_loss, root_acc
      else:
        root_prediction = sess.run(
          [self.root_prediction, self.root_acc],
          feed_dict=feed_dict)
        return root_prediction, root_acc

  def run_epoch(self, batches, new_model=False, verbose=True):
    random.shuffle(batches)
    with tf.Session() as sess:
      if new_model:
        sess.run(tf.global_variables_initializer())
      else:
        saver = tf.train.Saver()
        saver.restore(sess, SAVE_DIR + '%s.temp' % self.config.model_name)

      m = len(self.train_data)
      num_batches = int(m / self.config.batch_size) + 1
      last_loss = float('inf')
      for batch in range(num_batches):
        feed_dict = batches[batch]
        loss_value, acc, _ = sess.run(
          [self.full_loss, self.root_acc, self.train_op],
          feed_dict=feed_dict)
        if verbose:
          sys.stdout.write('\r{} / {} :    loss = {} and acc = {}'.format(batch, num_batches, loss_value, acc))
          sys.stdout.flush()
      saver = tf.train.Saver()
      if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
      saver.save(sess, SAVE_DIR + '%s.temp' % self.config.model_name)


  def train(self, verbose=True, test=False):
    stopped = -1
    best_val_epoch = 0
    best_dev_loss = float('inf')
    best_dev_acc = 0

    batch_generator = generate_batch(self.dev_data, len(self.dev_data))
    self.feed_dict_dev = self.build_feed_dict(next(batch_generator), train=False)

    batches = []
    batch_generator = generate_batch(self.train_data, self.config.batch_size)
    num_batches = int(len(self.train_data) / self.config.batch_size) + 1
    for batch in range(num_batches):
      batches.append(self.build_feed_dict(next(batch_generator)))

    for epoch in range(self.config.max_epochs):
      print('\nepoch %d' % epoch)
      if epoch == 0:
        self.run_epoch(batches=batches, new_model=True)
      else:
        self.run_epoch(batches=batches)

      _, dev_loss, dev_acc = self.predict(self.dev_data,
        SAVE_DIR + '%s.temp' % self.config.model_name,
        get_loss=True, dataset='dev')
      print('\nDev loss : {} --- dev acc: {}'.format(dev_loss, dev_acc))

      if dev_acc > best_dev_acc:
        self.copy_weight_files(self.config.model_name, self.config.model_name + '.temp')
        best_dev_acc = dev_acc
        best_dev_epoch = epoch

      if (dev_acc == best_dev_acc) and (dev_loss < best_dev_loss):
        self.copy_weight_files(self.config.model_name, self.config.model_name + '.temp')
        self.copy_weight_files(self.config.model_name + '.loss', self.config.model_name + '.temp')
        best_dev_loss = dev_loss
        best_dev_epoch = epoch
      
      if dev_loss < best_dev_loss:
        self.copy_weight_files(self.config.model_name + '.loss', self.config.model_name + '.temp')
        best_dev_loss = dev_loss

      if (epoch - best_dev_epoch) > self.config.early_stopping:
        stopped = epoch
        break

    if verbose:
      sys.stdout.write('\r')
      sys.stdout.flush()

    print('\n\nstopped at %d\n' % stopped)
    print('best loss: ', best_dev_loss)
    print('best acc: ', best_dev_acc)
    with open(os.path.join(os.path.dirname(data_path), 'logs/log.txt'), 'a+') as f:
      f.write('best loss: {}, best acc: {}, at epoch: {}'.format(
        best_dev_loss, best_dev_acc, best_dev_epoch))

   

    if test:
      _, test_loss, test_acc = self.predict(self.test_data,
            SAVE_DIR + '%s' % self.config.model_name,
            get_loss=True)
      print() 
      print('According to best acc')
      print('\nTest loss : {} --- test acc: {}'.format(test_loss, test_acc))

      _, test_loss, test_acc = self.predict(self.test_data,
            SAVE_DIR + '%s.loss' % self.config.model_name,
            get_loss=True)
      print()
      print('According to best loss')
      print('\nTest loss : {} --- test acc: {}'.format(test_loss, test_acc))


    return {
        'best_acc': best_dev_acc,
        'best_loss': best_dev_loss
    }

  def copy_weight_files(self, new_path, old_path):
    shutil.copyfile(SAVE_DIR + '%s.index' % old_path,
                        SAVE_DIR + '%s.index' % new_path)
    shutil.copyfile(SAVE_DIR + '%s.meta' % old_path,
                        SAVE_DIR + '%s.meta' % new_path)
    shutil.copyfile(SAVE_DIR + '%s.data-00000-of-00001' % old_path,
                        SAVE_DIR + '%s.data-00000-of-00001' % new_path)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', "--dataset", default="ATT", required=False)
  parser.add_argument('-b', "--batch", default=32, required=False)
  parser.add_argument('-e', "--epoch", default=2, required=False)
  parser.add_argument('-p', "--data_path", default='../data', required=False)
  parser.add_argument('-k', "--bucketing", default=True, required=False)
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
    f = open(pickle_file,'rb')
    data = pickle.load(f)
  else:
    data = treeDS.load_shrinked_trees(config.trees_path, config.data_path)
    f = open(pickle_file, "wb")
    pickle.dump(data, f)
  
  train_perc = int(len(data) * 0.9)
  train_data = data[:train_perc]
  dev_data = data[train_perc:]
  test_data = None

  if args.bucketing:
    train_lens = [(len(t.get_words()), t) for t in train_data]
    train_lens.sort(key=lambda x:x[0])
    train_data = [t for i, t in train_lens]
    del train_lens
  model = TreeCNN(config, train_data, dev_data, test_data)

  start_time = time.time()
  stats = model.train(verbose=True)
  end_time = time.time()
  print('Done')
  print('Time for training ', config.model_name, ' is ', ((end_time - start_time) / 60), 'mins')   