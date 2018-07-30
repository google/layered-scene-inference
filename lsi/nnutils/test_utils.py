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

"""Generic Testing Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.misc.pilutil
import scipy.io as sio

import tensorflow as tf
from absl import logging as log
from lsi.nnutils import helpers as nn_helpers
from lsi.visualization import html_utils


def define_default_flags(flags):
  """Default flags for a view synthesis trainer.

  Args:
    flags: loaded flags module
  """

  ## Flags for logging and snapshotting
  flags.DEFINE_string('checkpoint_dir', '/data0/shubhtuls/code/lsi/cachedir/snapshots/',
                      'Root directory for tensorflow output files')
  flags.DEFINE_string('results_vis_dir', '/data0/shubhtuls/code/lsi/cachedir/visualization/',
                      'Root directory for image output files')
  flags.DEFINE_string('results_eval_dir', '/data0/shubhtuls/code/lsi/cachedir/evaluation/',
                      'Root directory for results')
  flags.DEFINE_string(
      'exp_name', '',
      'Name of previous net to pretrain from.')
  flags.DEFINE_integer(
      'train_iter', 0,
      'Iteration of saved net to evaluate. 0 implies use latest')

  ## Flags for training
  flags.DEFINE_integer('batch_size', 2, 'Size of minibatches')
  flags.DEFINE_integer('num_eval_iter', 100, 'Number of testing iterations')
  flags.DEFINE_integer('img_height', 256, 'image height')
  flags.DEFINE_integer('img_width', 256, 'image width')
  flags.DEFINE_integer('visuals_freq', 10, 'logging frequency for visuals')
  flags.DEFINE_boolean('save_pred_results', False, 'Save predictions at every iter. Useful for KITTI depth eval.')


class Tester(object):
  """Generic class for a view synthesis tester.
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

  def define_metrics(self):
    """Should be implemented by the child class."""
    raise NotImplementedError

  def define_visuals(self):
    """Should be implemented by the child class."""
    raise NotImplementedError

  def init_graph(self):
    self.visuals = {}
    self.pred_results = {}
    self.define_placeholders()
    self.define_pred_graph()
    self.define_metrics()
    self.define_pred_results()
    self.define_visuals()

  def feed(self):
    """Should be implemented by the child class.

    Returns:
      feed_dict: Loads data and returns feed dict for all input data to graph
    """
    raise NotImplementedError

  def write_summary_page(self):
    """Create summary page.

    """
    summary_path = os.path.join(self.opts.results_vis_dir, 'summary.html')

    keys = self.visuals_keys
    keys.sort()
    table_rows = [html_utils.table_row(keys)]
    for vi in range(self.vis_iter-1):
      table_cols = [html_utils.image(
          'vis_iter_{}/{}.png'.format(vi, k), k,
          self.opts.img_height, self.opts.img_width
      ) for k in keys]
      table_rows.append(html_utils.table_row(table_cols))

    html = html_utils.html_page(html_utils.table(table_rows))
    with open(summary_path, mode='w') as f:
      f.write(html)

  def save_visuals(self, visuals):
    """Save visuals.

    Args:
      visuals: Dictionary of images.
    """
    self.visuals_keys = visuals.keys()
    imgs_dir = os.path.join(
        self.opts.results_vis_dir, 'vis_iter_{}'.format(self.vis_iter))
    if not os.path.exists(imgs_dir):
      os.makedirs(imgs_dir)
    for k in visuals:
      img_path = os.path.join(imgs_dir, k + '.png')
      with open(img_path, 'w') as f:
        if 'disp' in k:
          cmap = plt.get_cmap('magma')
          visuals[k] = cmap(visuals[k])
        scipy.misc.imsave(f, visuals[k])

    self.vis_iter += 1


  def save_preds(self, preds):
    """Save visuals.

    Args:
      preds: Dictionary of images.
    """
    preds['img_names'] = self.data_loader.src_image_names
    save_file = os.path.join(
        self.opts.results_eval_dir, 'iter_{}.mat'.format(self.pred_save_iter))
    sio.savemat(save_file, preds)
    self.pred_save_iter += 1


  def test(self):
    """Training routine.
    """
    log.info('Init Testing')
    seed = 0
    self.vis_iter = 0
    self.pred_save_iter = 0
    tf.set_random_seed(seed)
    np.random.seed(seed)

    opts = self.opts
    self.define_data_loader()
    self.init_graph()
    self.metrics_data = {}
    self.metrics_norm_data = {}
    for k in self.metrics:
      self.metrics_data[k] = []
      self.metrics_norm_data[k] = []

    with tf.name_scope('saver'):
      var_list = [var for var in tf.model_variables()]

      self.saver = tf.train.Saver(
          var_list)

      self.checkpoint = os.path.join(
          opts.checkpoint_dir,
          'model-{}'.format(opts.train_iter))

    with tf.Session() as sess:
      # print('Trainable variables: ')
      # for var in tf.model_variables():
        # print(var.name)

      if not os.path.exists(opts.results_eval_dir):
          os.makedirs(opts.results_eval_dir)

      # check if a previous checkpoint exists in current folder
      checkpoint = tf.train.latest_checkpoint(opts.checkpoint_dir)
      log.info('Previous checkpoint: ' + str(checkpoint))
      if opts.train_iter > 0:
        log.info('Loading net: %s', self.checkpoint)
        self.saver.restore(sess, self.checkpoint)

      if (opts.train_iter == 0) and (checkpoint is not None):
        log.info('Loading latest net: %s', checkpoint)
        self.saver.restore(sess, checkpoint)

      for step in range(1, opts.num_eval_iter+1):
        log.info('Iter : %d', step)
        fetches = [self.metrics, self.metrics_norm]
        if opts.save_pred_results:
          fetches.append(self.pred_results)

        if step % opts.visuals_freq == 0:
          fetches.append(self.visuals)

        results = sess.run(fetches, feed_dict=self.feed())
        log.info(results[0])
        for k in results[0]:
          self.metrics_data[k].append(results[0][k])
          self.metrics_norm_data[k].append(results[1][k])

        if opts.save_pred_results:
          self.save_preds(results[2])

        if step % opts.visuals_freq == 0:
          self.save_visuals(results[-1])

      self.write_summary_page()

      with open(os.path.join(opts.results_eval_dir, 'results.txt'), 'w') as eval_file:
        for k in self.metrics_data:
          self.metrics_data[k] = np.array(self.metrics_data[k])
          self.metrics_norm_data[k] = np.array(self.metrics_norm_data[k])

          eval_file.write('Mean {}: {}\n'.format(
              k, np.sum(self.metrics_data[k])/np.sum(self.metrics_norm_data[k])))
