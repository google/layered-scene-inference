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

"""Script for running layered predictor experiment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow as tf
from pyglib import app
from pyglib import flags
# from pyglib import log
from lsi.data.syntheticPlanes import utils as plane_utils
from lsi.loss import refinement as refine_utils
from lsi.nnutils import train_utils

FLAGS = flags.FLAGS

## Experiment Specific Flags
train_utils.define_default_flags(flags)
flags.DEFINE_string('exp_name', 'warp_net_debug',
                    'Name of the experiment')
flags.DEFINE_integer('n_layers', 5, 'Max number of layers')
flags.DEFINE_boolean('use_ms_loss', False, 'Use multi-scale loss')

## Flags specific to synthetic data loader
flags.DEFINE_string(
    'sun_imgs_dir',
    'data/sceneLayers/SUN2012/Images',
    'Directory where SUN dataset images are stored'
)


class Trainer(train_utils.Trainer):
  """Synthetic data trainer.
  """

  def define_data_loader(self):
    opts = self.opts
    self.data_loader = plane_utils.QueuedRandomTextureLoader(
        opts.sun_imgs_dir,
        batch_size=opts.batch_size,
        h=opts.img_height, w=opts.img_width)

  def define_pred_graph(self):
    """Prediction graph contruction.
    """
    opts = self.opts
    bs = opts.batch_size
    n_layers = opts.n_layers
    self.tform_rand = 0.2*(tf.random_uniform([n_layers, bs, 8]) - 0.5)
    self.imgs_src_pad = tf.pad(
        self.imgs_src, [[0, 0], [40, 40], [40, 40], [0, 0]])
    self.imgs_src_rsz = tf.image.resize_images(
        self.imgs_src_pad, [opts.img_height, opts.img_width])
    self.imgs_inp_rep = tf.tile(
        tf.expand_dims(self.imgs_src_rsz, axis=0), [n_layers, 1, 1, 1, 1])
    self.imgs_peturbed = refine_utils.corner_refine(
        self.imgs_inp_rep, self.tform_rand)
    self.refine_pred = 0.15*tf.tanh(refine_utils.refine_params(
        self.imgs_peturbed, self.imgs_src_rsz, 8))
    self.imgs_unpeturbed = refine_utils.corner_refine(
        self.imgs_peturbed, self.refine_pred)

  def define_summary_ops(self):
    """Summary ops contruction.
    """
    opts = self.opts
    tf.summary.scalar('total_loss', self.total_loss)
    tf.summary.scalar(
        'single_scale_loss',
        tf.reduce_mean(tf.abs(self.imgs_inp_rep - self.imgs_unpeturbed)))
    tf.summary.image('src_imgs', self.imgs_src_rsz)
    tf.summary.histogram('refine_pred', self.refine_pred)
    for p in range(opts.n_layers):
      tf.summary.image(
          'peturbed_img_%d' % p, self.imgs_peturbed[p, :, :, :, :])
      tf.summary.image(
          'unpeturbed_img_%d' % p, self.imgs_unpeturbed[p, :, :, :, :])

  def define_loss_graph(self):
    """Loss computation.
    """
    opts = self.opts
    h = opts.img_height
    w = opts.img_width
    bs = opts.batch_size
    n_layers = opts.n_layers
    if opts.use_ms_loss:
      self.total_loss = 0
      for ix in range(3):
        scale_fac = pow(2, ix)
        imgs_gt = tf.reshape(self.imgs_inp_rep, (bs*n_layers, h, w, 3))
        imgs_gt = tf.image.resize_images(imgs_gt, [h//scale_fac, w//scale_fac])
        imgs_gt = tf.reshape(
            imgs_gt, (n_layers, bs, h//scale_fac, w//scale_fac, 3))

        imgs_pet = tf.reshape(self.imgs_peturbed, (bs*n_layers, h, w, 3))
        imgs_pet = tf.image.resize_images(
            imgs_pet, [h//scale_fac, w//scale_fac])
        imgs_pet = tf.reshape(
            imgs_pet, (n_layers, bs, h//scale_fac, w//scale_fac, 3))

        imgs_unpeturbed = refine_utils.corner_refine(
            imgs_pet, self.refine_pred
        )
        self.total_loss += tf.reduce_mean(tf.abs(imgs_gt - imgs_unpeturbed))
    else:
      self.total_loss = tf.reduce_mean(
          tf.abs(self.imgs_inp_rep - self.imgs_unpeturbed))

  def feed(self):
    """Data loading wrapper.

    Returns:
      feed_dict: Loads data and returns feed dict for all input data to graph
    """
    img_src, _ = self.data_loader.load()

    feed_dict = {
        self.imgs_src: img_src
    }
    return feed_dict


def main(_):
  FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.exp_name)
  trainer = Trainer(FLAGS)
  trainer.train()


if __name__ == '__main__':
  app.run(main)
