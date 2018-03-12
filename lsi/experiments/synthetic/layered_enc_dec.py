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
from lsi.data.syntheticPlanes import data as synthetic_planes
from lsi.geometry import homography
from lsi.geometry import layers
from lsi.loss import loss
from lsi.nnutils import nets
from lsi.nnutils import train_utils

FLAGS = flags.FLAGS

## Experiment Specific Flags
train_utils.define_default_flags(flags)
flags.DEFINE_string('exp_name', 'synth_layer_pred_encdec',
                    'Name of the experiment')
flags.DEFINE_integer('n_layers', 10, 'Max number of layers')

## Flags specific to select data loader
flags.DEFINE_enum('dataset', 'synthetic', ['synthetic'], 'Dataset')

## Flags specific to synthetic data loader
flags.DEFINE_string(
    'pascal_objects_dir',
    'data/sceneLayers/sbd/objects',
    'Directory where images of pascal objects are stored'
)
flags.DEFINE_string(
    'sun_imgs_dir',
    'data/sceneLayers/SUN2012/Images',
    'Directory where SUN dataset images are stored'
)
flags.DEFINE_integer('n_obj_min', 1,
                     'Min number of foreground layers in synthetic data')
flags.DEFINE_integer('n_obj_max', 4,
                     'Max number of foreground layers in synthetic data')
flags.DEFINE_integer('n_box_planes', 5,
                     'Number of planes from the box to use')

## Flags related to the training (loss, CNN architecture etc.)
flags.DEFINE_float('self_cons_wt', 1.0,
                   'Weight for ordered self-consistency loss')
flags.DEFINE_float('trg_recons_wt', 1.0,
                   'Weight for target reconstruction loss')
flags.DEFINE_boolean('pre_loss_refine', False,
                     'Allow refining the predictions before trg loss')
flags.DEFINE_boolean('use_unet', False,
                     'Whether to use a CNN with skip connections')
flags.DEFINE_boolean('use_plane_sweep', False,
                     'Whether to use fixed depth planes')
flags.DEFINE_float('max_disp_psweep', 0.5,
                   'Inverse depth for closest plane')
flags.DEFINE_float('min_disp_psweep', 0.25,
                   'Inverse depth for farthest plane')
flags.DEFINE_float(
    'src_transform_wt', 0,
    'Weight for reconstruction loss by transforming src image to trg frame')
flags.DEFINE_float('bg_layer_disp', 1e-6,
                   'Disparity of bg layer: value automatically chosen in code')
flags.DEFINE_float('depth_softmax_temp', 1e-6,
                   'Softmax temperature: value automatically chosen in code')


class Trainer(train_utils.Trainer):
  """Synthetic data trainer.
  """

  def define_data_loader(self):
    opts = self.opts
    if opts.dataset == 'synthetic':
      self.data_loader = synthetic_planes.DataLoader(opts)

  def define_pred_graph(self):
    """Prediction graph contruction.
    """
    opts = self.opts
    bs = opts.batch_size
    n_layers = opts.n_layers
    if opts.use_unet:
      feat, feat_dec, enc_dec_int = nets.encoder_decoder_unet(
          self.imgs_src)
    else:
      feat, feat_dec, enc_dec_int = nets.encoder_decoder_simple(
          self.imgs_src)

    if opts.use_plane_sweep:
      self.plane_pred = nets.depthsweep_planes(
          opts.batch_size, n_layers=n_layers,
          min_disp=opts.min_disp_psweep, max_disp=opts.max_disp_psweep)
    else:
      self.plane_pred, _ = nets.plane_predictor(feat, n_layers=n_layers)
    if opts.shift_pred_planes:
      self.plane_shift = tf.placeholder(
          tf.float32, [bs, 1, 3], name='plane_shift')
      self.plane_pred = homography.shift_plane_eqns(
          self.plane_shift, self.plane_pred)

    self.tex_pred, _ = nets.pixelwise_predictor(
        feat_dec, nc=3, n_layers=n_layers, scope='tex')
    self.masks_pred, _ = nets.pixelwise_predictor(
        feat_dec, nc=1, n_layers=n_layers, scope='masks')

    ## Summary Ops
    for var in enc_dec_int.values():
      tf.summary.histogram(var.name + '/activation', var)
    # tf.summary.histogram(self.feat.op.name + '/activation', feat)

  def define_summary_ops(self):
    """Summary ops contruction.
    """
    opts = self.opts
    tf.summary.scalar('total_loss', self.total_loss)
    tf.summary.scalar('self_cons_loss', self.self_cons_loss)
    tf.summary.scalar('trg_recons_loss', self.trg_recons_loss)
    tf.summary.scalar('src_transform_loss', self.src_transform_loss)
    # for var in tf.trainable_variables():
    #   tf.summary.histogram(var.name + '/values', var)

    # for grad, var in self.grads_and_vars:
    #   tf.summary.histogram(var.op.name + '/gradients', grad)
    tf.summary.histogram(self.tex_pred.name + '/activation', self.tex_pred)
    tf.summary.histogram(
        self.masks_pred.name + '/activation', self.masks_pred)
    imgs_src_vis = tf.cast(self.imgs_src*255, tf.uint8)
    imgs_trg_vis = tf.cast(self.imgs_trg*255, tf.uint8)

    tf.summary.image('src_imgs', imgs_src_vis)
    tf.summary.image('trg_imgs', imgs_trg_vis)

    for p in range(opts.n_layers):
      var_norm = self.plane_pred[0]
      var_disp = self.plane_pred[1]
      tf.summary.histogram(
          var_norm.name + str(p) + '_x' + '/values', var_norm[p, :, :, 0])
      tf.summary.histogram(
          var_norm.name + str(p) + '_y' + '/values', var_norm[p, :, :, 1])
      tf.summary.histogram(
          var_norm.name + str(p) + '_z' + '/values', var_norm[p, :, :, 2])
      tf.summary.histogram(
          var_disp.name + str(p) + '/values', var_disp[p])
      tf.summary.image(
          'src_tex_pred_layer_%d' % p,
          tf.cast(self.tex_pred[p]*255, tf.uint8))
      tf.summary.image(
          'src_mask_pred_layer_%d' % p,
          tf.cast(self.masks_pred[p]*255, tf.uint8))

  def define_loss_graph(self):
    """Loss computation.
    """
    opts = self.opts

    id_rot = tf.tile(
        tf.expand_dims(tf.diag([1., 1., 1.]), axis=0), [opts.batch_size, 1, 1])
    id_trans = tf.zeros([opts.batch_size, 3, 1])

    _, _, disp_src_frame = layers.planar_transform(
        self.tex_pred, self.masks_pred, self.pixel_coords,
        self.k_s, self.k_s, id_rot, id_trans,
        self.plane_pred[0], self.plane_pred[1])

    self.self_cons_loss, self.cons_loss_interm = loss.ordered_composition_loss(
        self.tex_pred, self.masks_pred, disp_src_frame, self.imgs_src)

    imgs_trg_frame, masks_trg_frame, disp_trg_frame = layers.planar_transform(
        self.tex_pred, self.masks_pred, self.pixel_coords,
        self.k_s, self.k_t, self.rot_mat, self.trans_mat,
        self.plane_pred[0], self.plane_pred[1])
        # 0*self.plane_pred[0] + self.n_hat_gt,
        # 0*self.plane_pred[1] + self.a_gt)

    with tf.variable_scope('trg_recons_loss'):
      self.trg_recons_loss, trg_loss_vis = loss.depth_composition_loss(
          imgs_trg_frame, masks_trg_frame,
          disp_trg_frame, self.imgs_trg,
          min_disp=opts.bg_layer_disp,
          depth_softmax_temp=opts.depth_softmax_temp,
          pre_loss_refine=opts.pre_loss_refine)
      trg_loss_vis[0] = tf.cast(trg_loss_vis[0]*255, tf.uint8)

    self.total_loss = opts.trg_recons_wt*self.trg_recons_loss
    self.total_loss += opts.self_cons_wt*self.self_cons_loss

    with tf.variable_scope('src_transform_loss'):
      src_img_rep = tf.expand_dims(self.imgs_src, axis=0)
      src_img_rep *= tf.ones([opts.n_layers, 1, 1, 1, 1])
      src2trg_frame, _, _ = layers.planar_transform(
          src_img_rep, self.masks_pred, self.pixel_coords,
          self.k_s, self.k_t, self.rot_mat, self.trans_mat,
          self.plane_pred[0], self.plane_pred[1])
      self.src_transform_loss, _ = loss.depth_composition_loss(
          src2trg_frame, masks_trg_frame,
          disp_trg_frame, self.imgs_trg,
          min_disp=opts.bg_layer_disp,
          depth_softmax_temp=opts.depth_softmax_temp,
          pre_loss_refine=opts.pre_loss_refine)

      self.total_loss += opts.src_transform_wt*self.src_transform_loss

    # Summary Ops
    tf.summary.image(
        'pred_trg_imgs_composed',
        layers.compose(
            imgs_trg_frame, masks_trg_frame,
            disp_trg_frame, min_disp=opts.bg_layer_disp,
            depth_softmax_temp=opts.depth_softmax_temp))

    tf.summary.image(
        'src_transform_imgs_composed',
        layers.compose(
            src2trg_frame, masks_trg_frame,
            disp_trg_frame, min_disp=opts.bg_layer_disp,
            depth_softmax_temp=opts.depth_softmax_temp))

    for var in self.cons_loss_interm:
      tf.summary.scalar(var.name + '/activation', tf.reduce_mean(var))
      # tf.summary.histogram(var.name + '/activation', var)
    for p in range(opts.n_layers):
      masks_trg_frame_vis = tf.cast(masks_trg_frame*255, tf.uint8)
      imgs_trg_frame_vis = tf.cast(imgs_trg_frame*255, tf.uint8)
      tf.summary.image(
          'trg_tex_pred_layer_%d' % p, imgs_trg_frame_vis[p, :, :, :, :])
      tf.summary.image(
          'trg_mask_pred_layer_%d' % p, masks_trg_frame_vis[p, :, :, :, :])
      # tf.summary.image(
      #     'trg_disp_pred_layer_%d' % p, disp_trg_frame[p])
      tf.summary.image(
          'src_img_transform_layer_%d' % p, src2trg_frame[p, :, :, :, :])

    for p in range(opts.n_layers + 1):  # Also visualize escape probability
      tf.summary.image(
          'layer_prob_layer_%d' % p, trg_loss_vis[0][p, :, :, :, :])

    if opts.pre_loss_refine:
      tf.summary.image(
          'pred_trg_imgs_refined_composed',
          layers.compose(
              trg_loss_vis[1], trg_loss_vis[2],
              trg_loss_vis[3], min_disp=opts.bg_layer_disp,
              depth_softmax_temp=opts.depth_softmax_temp))
      trg_loss_vis[1] = tf.cast(trg_loss_vis[1]*255, tf.uint8)
      trg_loss_vis[2] = tf.cast(trg_loss_vis[2]*255, tf.uint8)

      for p in range(opts.n_layers):
        tf.summary.image(
            'trg_tex_pred_layer_refined_%d' % p,
            trg_loss_vis[1][p, :, :, :, :])
        tf.summary.image(
            'trg_mask_pred_layer_refined_%d' % p,
            trg_loss_vis[2][p, :, :, :, :])

  def feed(self):
    """Data loading wrapper.

    Returns:
      feed_dict: Loads data and returns feed dict for all input data to graph
    """
    opts = self.opts
    data_batch = self.data_loader.forward(opts.batch_size)
    if opts.dataset == 'synthetic':
      img_src, img_trg, k_s, k_t, rot_mat, trans_mat, _, _ = data_batch
    # n_hat_gt = np.transpose(n_hat_gt, (1, 0, 2, 3))
    # a_gt = np.transpose(a_gt, (1, 0, 2, 3))
    # log.info('Src Mean : %f', np.mean(img_src))
    # log.info('Trg Mean : %f', np.mean(img_trg))

    feed_dict = {
        self.imgs_src: img_src, self.imgs_trg: img_trg,
        self.k_s: k_s, self.k_t: k_t,
        self.rot_mat: rot_mat, self.trans_mat: trans_mat
    }
    if opts.shift_pred_planes:
      feed_dict[self.plane_shift] = plane_shift
    return feed_dict


def main(_):
  FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.exp_name)
  if FLAGS.dataset == 'synthetic':
    FLAGS.bg_layer_disp = 2e-1
    FLAGS.depth_softmax_temp = 0.4
  else:
    FLAGS.bg_layer_disp = 3e-4
    FLAGS.depth_softmax_temp = 20
  trainer = Trainer(FLAGS)
  trainer.train()

if __name__ == '__main__':
  app.run(main)
