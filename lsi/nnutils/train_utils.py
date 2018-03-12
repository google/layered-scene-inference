#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic Training Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import numpy as np
import tensorflow as tf
from pyglib import log
from lsi.nnutils import helpers as nn_helpers


def define_default_flags(flags):
  """Default flags for a view synthesis trainer.

  Args:
    flags: loaded flags module
  """

  ## Flags for logging and snapshotting
  flags.DEFINE_string('checkpoint_dir', '/tmp/experiments/snapshots/',
                      'Root directory for tensorflow output files')
  flags.DEFINE_string(
      'pretrain_name', '',
      'Name of previous net to pretrain from. Empty -> from scratch')
  flags.DEFINE_integer(
      'pretrain_iter', 100000,
      'Iteration of previous net to pretrain from. Empty -> from scratch')

  ## Flags for training
  flags.DEFINE_integer('batch_size', 2, 'Size of minibatches')
  flags.DEFINE_integer('num_iter', 100000, 'Number of training iterations')
  flags.DEFINE_integer('img_height', 256, 'image height')
  flags.DEFINE_integer('img_width', 256, 'image width')
  flags.DEFINE_integer('log_freq', 5, 'logging frequency')
  flags.DEFINE_integer('checkpoint_freq', 20000, 'checkpoint frequency')
  flags.DEFINE_integer('save_latest_freq', 2000, 'latest model save frequency')
  flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
  flags.DEFINE_float('beta1', 0.9, 'Momentum term of adam')


class Trainer(object):
  """Generic class for a view synthesis trainer.
  """

  def __init__(self, opts):
    self.opts = opts

  def define_data_loader(self):
    """Should be implemented by the child class."""
    raise NotImplementedError

  def define_placeholders(self):
    """Define placeholders.
    """
    opts = self.opts
    img_height = opts.img_height
    img_width = opts.img_width
    bs = opts.batch_size

    self.imgs_src = tf.placeholder(
        tf.float32, [bs, img_height, img_width, 3], name='imgs_inp')
    self.imgs_trg = tf.placeholder(
        tf.float32, [bs, img_height, img_width, 3], name='imgs_trg')

    self.pixel_coords = nn_helpers.pixel_coords(bs, img_height, img_width)
    self.k_s = tf.placeholder(tf.float32, [bs, 3, 3], name='k_s')
    self.k_t = tf.placeholder(tf.float32, [bs, 3, 3], name='k_t')
    self.rot_mat = tf.placeholder(tf.float32, [bs, 3, 3], name='rot_mat')
    self.trans_mat = tf.placeholder(tf.float32, [bs, 3, 1], name='trans_mat')

  def define_pred_graph(self):
    """Should be implemented by the child class."""
    raise NotImplementedError

  def define_summary_ops(self):
    """Should be implemented by the child class."""
    raise NotImplementedError

  def define_loss_graph(self):
    """Should be implemented by the child class."""
    raise NotImplementedError

  def init_graph(self):
    self.define_placeholders()
    self.define_pred_graph()
    self.define_loss_graph()

  def define_train_op(self):
    """Defines the training op."""
    opts = self.opts
    with tf.name_scope('train_op'):
      train_vars = [var for var in tf.trainable_variables()]
      optim = tf.train.AdamOptimizer(opts.learning_rate, opts.beta1)
      self.grads_and_vars = optim.compute_gradients(
          self.total_loss, var_list=train_vars)
      self.train_op = optim.apply_gradients(self.grads_and_vars)
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      self.incr_global_step = tf.assign(self.global_step, self.global_step+1)

  def grad_debug_op(self):
    """Defines gradient debugging ops."""
    with tf.name_scope('grad_debug_op'):
      self.mean_grads = [
          tf.Print(
              tf.reduce_mean(gv[0]),
              [tf.reduce_mean(gv[0])], message=gv[1].name
          ) for gv in self.grads_and_vars if gv[0] is not None]

      debug_vars = {}

      for k in self.debug_vars:
        debug_vars[k] = self.debug_vars[k]

      debug_var_names = debug_vars.keys()
      debug_vars_list = [debug_vars[k] for k in debug_var_names]
      grad_debug_vars = tf.gradients(self.total_loss, debug_vars_list)
      self.mean_grads_debug = [
          tf.Print(
              tf.reduce_mean(grad_debug_vars[ix]),
              [tf.reduce_mean(grad_debug_vars[ix])],
              message=debug_var_names[ix] + '_grad: '
          ) for ix in range(len(debug_vars)) if grad_debug_vars[ix] is not None
      ]

  def feed(self):
    """Should be implemented by the child class.

    Returns:
      feed_dict: Loads data and returns feed dict for all input data to graph
    """
    raise NotImplementedError

  def train(self):
    """Training routine.
    """
    seed = 0
    tf.set_random_seed(seed)
    np.random.seed(seed)

    opts = self.opts
    self.debug_vars = {}
    self.define_data_loader()
    self.init_graph()
    self.define_train_op()
    self.define_summary_ops()
    # check_op = tf.add_check_numerics_ops()
    with tf.name_scope('parameter_count'):
      parameter_count = tf.reduce_sum([
          tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()
      ])
      var_list = [var for var in tf.model_variables()]

      self.saver = tf.train.Saver(
          var_list + [self.global_step],
          max_to_keep=10)

      if opts.pretrain_name:
        self.checkpoint = os.path.join(
            opts.checkpoint_dir, '..',
            opts.pretrain_name,
            'model-{}'.format(opts.pretrain_iter))

        self.restorer = nn_helpers.optimistic_restorer(
            self.checkpoint, var_list + [self.global_step])

    sv = tf.train.Supervisor(
        logdir=opts.checkpoint_dir,
        save_summaries_secs=0,
        saver=None
    )
    with sv.managed_session() as sess:
      # print('Trainable variables: ')
      # for var in tf.trainable_variables():
      #   print(var.name)
      print('parameter_count =', sess.run(parameter_count))

      # check if a previous checkpoint exists in current folder
      checkpoint = tf.train.latest_checkpoint(opts.checkpoint_dir)
      log.info('Previous checkpoint: ' + str(checkpoint))
      if checkpoint is not None:
        log.info('Restoring')
        print('Resume training from previous checkpoint: %s' % checkpoint)
        self.saver.restore(sess, checkpoint)

      if (checkpoint is None) and opts.pretrain_name:
        log.info('Restoring')
        print('Resume training from pretrained net: %s' % self.checkpoint)
        # self.saver.restore(sess, self.checkpoint)
        self.restorer.restore(sess, self.checkpoint)

      for step in range(1, opts.num_iter+1):
        log.info('Iter : %d', step)
        fetches = {
            'train': self.train_op,
            'global_step': self.global_step,
            'incr_global_step': self.incr_global_step,
            # 'check': check_op,
            # 'mean_grads': self.mean_grads,
            # 'mean_grads_debug': self.mean_grads_debug,
        }

        if step % opts.log_freq == 0:
          fetches['loss'] = self.total_loss
          fetches['summary'] = sv.summary_op

        results = sess.run(fetches, feed_dict=self.feed())
        gs = results['global_step']

        if step % opts.log_freq == 0:
          sv.summary_writer.add_summary(results['summary'], gs)
        if step % opts.save_latest_freq == 0:
          self.save(sess, opts.checkpoint_dir, 'latest')
        if step % opts.checkpoint_freq == 0:
          self.save(sess, opts.checkpoint_dir, gs)

  def save(self, sess, checkpoint_dir, step):
    model_name = 'model'
    print(' [*] Saving checkpoint to %s...' % checkpoint_dir)
    if step == 'latest':
      self.saver.save(
          sess, os.path.join(checkpoint_dir, model_name + '.latest'))
    else:
      self.saver.save(
          sess, os.path.join(checkpoint_dir, model_name),
          global_step=step)
