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
"""Script for running ldi predictor experiment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from lsi.data.kitti import data as kitti_data
from lsi.data.syntheticPlanes import data as synthetic_planes
from lsi.geometry import ldi as ldi_utils
from lsi.loss import loss
from lsi.nnutils import helpers as nn_helpers
from lsi.nnutils import nets
from lsi.nnutils import train_utils
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

## Experiment Specific Flags
train_utils.define_default_flags(flags)
flags.DEFINE_string('exp_name', 'synth_ldi_pred_encdec',
                    'Name of the experiment')
flags.DEFINE_integer('n_layers', 2, 'Number of LDI layers')
flags.DEFINE_boolean(
    'pred_ldi_masks', False, """Predict masks for LDIs or use 1s.
    All experiments so far always used False""")

## Flags specific to select data loader
flags.DEFINE_enum('dataset', 'synthetic', ['synthetic', 'kitti'], 'Dataset')
flags.DEFINE_enum('data_split', 'train', ['all', 'train', 'val', 'test'],
                  'Dataset split')
flags.DEFINE_boolean('debug_synth_texture', False,
                     'Use gt LDI disps instead of predicted')

## Flags specific to synthetic data loader
flags.DEFINE_string('pascal_objects_dir', '/code/lsi/cachedir/sbd/objects',
                    'Directory where images of pascal objects are stored')
flags.DEFINE_string('sun_imgs_dir', '/datasets/SUN2012pascalformat/JPEGImages',
                    'Directory where SUN dataset images are stored')
flags.DEFINE_integer('n_obj_min', 1,
                     'Min number of foreground layers in synthetic data')
flags.DEFINE_integer('n_obj_max', 4,
                     'Max number of foreground layers in synthetic data')
flags.DEFINE_integer('n_box_planes', 5, 'Number of planes from the box to use')
flags.DEFINE_integer(
    'synth_ds_factor', 1,
    'Render synthetic data at a higher res and downsample by this factor')
flags.DEFINE_boolean('synth_dl_eval_data', False,
                     'Output gt info for synth data')

## Flags specific to kitti dataset
flags.DEFINE_string('kitti_data_root', '/datasets/kitti',
                    'Directory where kitti data images are cameras are stored')
flags.DEFINE_enum('kitti_dataset_variant', 'mview',
                  ['odom', 'mview', 'raw_city'], 'Kitti set to use')
flags.DEFINE_boolean('kitti_dl_disparities', False,
                     'Output gt info for kitti disparities')

## Flags related to the training (loss, CNN architecture etc.)
flags.DEFINE_float('self_cons_wt', 1.0,
                   'Weight for ordered self-consistency loss')
flags.DEFINE_boolean(
    'l0_self_cons', False,
    'If true, use layer 0 texture for self cons, else use composed image')
flags.DEFINE_float(
    'indep_splat_wt', 1.0,
    'Weight for reconstruction loss via forward splatting (min across layers)')
flags.DEFINE_float(
    'compose_splat_wt', 1.0,
    'Weight for reconstruction loss via layer composition forward splatting')
flags.DEFINE_float('splat_bdry_ignore', 0.1,
                   'Ignore this fraction of pixels along the boundary')
flags.DEFINE_float('zbuf_scale', 50, 'Scale for zbuffer weight computation')
flags.DEFINE_float('trg_splat_downsampling', 0.5,
                   'The forward splatted image is downsampled by this factor')
flags.DEFINE_float('disp_smoothness_wt', 0.1, 'Disparity should vary smoothly')
flags.DEFINE_float('incr_depth_wt', 10.0,
                   'Relative weight for depth increment loss')
flags.DEFINE_boolean('use_unet', True,
                     'Whether to use a CNN with skip connections')
flags.DEFINE_integer('n_layerwise_steps', 3,
                     'Number of independent per-layer up-conv steps')

## Dataset dependent flags : overridden in code
flags.DEFINE_float('bg_layer_disp', 1e-6,
                   'Disparity of bg layer: value automatically chosen in code')
flags.DEFINE_float('depth_softmax_temp', 1e-6,
                   'Softmax temperature: value automatically chosen in code')
flags.DEFINE_float(
    'disp_vis_scale', 255,
    'Disparity visualization scale: value automatically chosen in code')
flags.DEFINE_float(
    'max_disp', 0, """Inverse depth for closest plane.
    Value automatically chosen in code according to the dataset if set to 0.
    """)


class Trainer(train_utils.Trainer):
  """Synthetic data trainer.
  """

  def define_data_loader(self):
    opts = self.opts
    if opts.dataset == 'synthetic':
      self.data_loader = synthetic_planes.DataLoader(opts)
    elif opts.dataset == 'kitti':
      self.data_loader = kitti_data.DataLoader(opts)
      self.data_loader.define_queues()
      self.data_loader.preload_calib_files()

  def define_summary_ops(self):
    """Summary ops contruction.
    """
    opts = self.opts
    tf.summary.scalar('compose_splat_loss', self.compose_splat_loss)
    tf.summary.scalar('indep_splat_loss', self.indep_splat_loss)
    tf.summary.scalar('self_cons_loss', self.self_cons_loss)
    tf.summary.scalar('incr_depth_loss', self.incr_depth_loss)
    tf.summary.scalar('disp_smoothness_loss', self.disp_smoothness_loss)
    tf.summary.scalar('total_loss', self.total_loss)
    imgs_src_vis = tf.cast(self.imgs_src * 255, tf.uint8)
    imgs_trg_vis = tf.cast(self.imgs_trg * 255, tf.uint8)

    tf.summary.image('src_imgs', imgs_src_vis)
    tf.summary.image('trg_imgs', imgs_trg_vis)

    for p in range(opts.n_layers):
      tf.summary.image('src_tex_pred_layer_%d' % p,
                       tf.cast(self.ldi_src[0][p] * 255, tf.uint8))
      tf.summary.image(
          'src_disp_pred_layer_%d' % p,
          tf.cast(self.ldi_src[2][p] * opts.disp_vis_scale, tf.uint8))
      tf.summary.image('trg_tex_pred_layer_%d' % p,
                       tf.cast(self.ldi_trg[0][p] * 255, tf.uint8))
      tf.summary.image(
          'trg_disp_pred_layer_%d' % p,
          tf.cast(self.ldi_trg[2][p] * opts.disp_vis_scale, tf.uint8))
      if opts.pred_ldi_masks:
        tf.summary.image('src_mask_pred_layer_%d' % p,
                         tf.cast(self.ldi_src[1][p] * 255, tf.uint8))
        tf.summary.image('trg_mask_pred_layer_%d' % p,
                         tf.cast(self.ldi_trg[1][p] * 255, tf.uint8))

    for p in range(opts.n_layers):
      tf.summary.histogram('pred_disp_' + str(p), self.ldi_src[2][p])

  def define_pred_graph(self):
    """Prediction graph contruction.
    """
    opts = self.opts
    if opts.debug_synth_texture:
      img_height = opts.img_height
      img_width = opts.img_width
      bs = opts.batch_size
      nl = opts.n_layers
      self.src_ldi_gt_disps = tf.placeholder(
          tf.float32, [nl, bs, img_height, img_width, 1],
          name='src_ldi_gt_disps')
      self.trg_ldi_gt_disps = tf.placeholder(
          tf.float32, [nl, bs, img_height, img_width, 1],
          name='trg_ldi_gt_disps')

    self.focal_disps = None

    # Transform from trg to src frame
    self.inv_rot_mat = nn_helpers.transpose(self.rot_mat)
    self.inv_trans_mat = -tf.matmul(self.inv_rot_mat, self.trans_mat)
    n_layers = opts.n_layers
    if opts.use_unet:
      _, feat_dec_src, skip_feat_src, _ = nets.encoder_decoder_unet(
          self.imgs_src, nl_diff_enc_dec=opts.n_layerwise_steps)
      _, feat_dec_trg, skip_feat_trg, _ = nets.encoder_decoder_unet(
          self.imgs_trg, reuse=True, nl_diff_enc_dec=opts.n_layerwise_steps)
    else:
      _, feat_dec_src, skip_feat_src, _ = nets.encoder_decoder_simple(
          self.imgs_src, nl_diff_enc_dec=opts.n_layerwise_steps)
      _, feat_dec_trg, skip_feat_trg, _ = nets.encoder_decoder_simple(
          self.imgs_trg, reuse=True, nl_diff_enc_dec=opts.n_layerwise_steps)
    self.ldi_src = nets.ldi_predictor(
        feat_dec_src,
        n_layers=n_layers,
        n_layerwise_steps=opts.n_layerwise_steps,
        skip_feat=skip_feat_src,
        pred_masks=opts.pred_ldi_masks)
    self.ldi_src[2] *= opts.max_disp
    self.ldi_trg = nets.ldi_predictor(
        feat_dec_trg,
        n_layers=n_layers,
        reuse=True,
        n_layerwise_steps=opts.n_layerwise_steps,
        skip_feat=skip_feat_trg,
        pred_masks=opts.pred_ldi_masks)
    self.ldi_trg[2] *= opts.max_disp
    # Select inverse depths predicted for the 1st layer of the LDIs
    if opts.debug_synth_texture:
      self.ldi_src[2] = 0 * self.ldi_src[2] + 1 * self.src_ldi_gt_disps
      self.ldi_trg[2] = 0 * self.ldi_trg[2] + 1 * self.trg_ldi_gt_disps

    self.disps_src = self.ldi_src[2][0]
    self.disps_trg = self.ldi_trg[2][0]

  def feed(self):
    """Data loading wrapper.

    Returns:
      feed_dict: Loads data and returns feed dict for all input data to graph
    """
    opts = self.opts
    data_batch = self.data_loader.forward(opts.batch_size)
    if opts.dataset == 'synthetic' and opts.synth_dl_eval_data:
      (img_src, img_trg, k_s, k_t, rot_mat, trans_mat, _, _, disp_s_fg,
       disp_s_bg, disp_t_fg, disp_t_bg, _, _) = data_batch
    elif opts.dataset == 'kitti' and opts.kitti_dl_disparities:
      (img_src, img_trg, k_s, k_t, rot_mat, trans_mat, disp_s_fg,
       disp_t_fg) = data_batch
    else:
      img_src, img_trg, k_s, k_t, rot_mat, trans_mat = data_batch

    feed_dict = {
        self.imgs_src: img_src,
        self.imgs_trg: img_trg,
        self.k_s: k_s,
        self.k_t: k_t,
        self.rot_mat: rot_mat
    }

    feed_dict[self.trans_mat] = trans_mat

    if opts.debug_synth_texture:
      feed_dict[self.src_ldi_gt_disps] = np.stack(
          [disp_s_fg, disp_s_bg], axis=0)
      feed_dict[self.trg_ldi_gt_disps] = np.stack(
          [disp_t_fg, disp_t_bg], axis=0)

    return feed_dict

  def define_loss_graph(self):
    """Loss computation.
    """
    opts = self.opts
    # Self-consisteny for source image
    if opts.l0_self_cons:
      self.self_cons_loss_src = tf.reduce_mean(
          tf.abs(self.imgs_src - self.ldi_src[0][0]))
    else:
      self.self_cons_loss_src = loss.zbuffer_composition_loss(
          self.ldi_src[0],
          self.ldi_src[1],
          self.ldi_src[2],
          self.imgs_src,
          zbuf_scale=opts.zbuf_scale,
          bg_layer_disp=opts.bg_layer_disp,
          max_disp=opts.max_disp)

    # Self-consisteny for target image
    if opts.l0_self_cons:
      self.self_cons_loss_trg = tf.reduce_mean(
          tf.abs(self.imgs_trg - self.ldi_trg[0][0]))
    else:
      self.self_cons_loss_trg = loss.zbuffer_composition_loss(
          self.ldi_trg[0],
          self.ldi_trg[1],
          self.ldi_trg[2],
          self.imgs_trg,
          zbuf_scale=opts.zbuf_scale,
          bg_layer_disp=opts.bg_layer_disp,
          max_disp=opts.max_disp)

    self.self_cons_loss = self.self_cons_loss_src + self.self_cons_loss_trg

    # Recons loss for trg/src image via src/trg image LDI
    self.indep_splat_loss = 0
    self.compose_splat_loss = 0
    for sp_loss_type in ['indep', 'compose']:
      use_compose_splat = (sp_loss_type == 'compose')
      for loss_img_name in ['trg', 'src']:
        if loss_img_name == 'trg':
          to_recons_img = self.imgs_trg
          recons_splat, _ = ldi_utils.forward_splat(
              self.ldi_src,
              self.pixel_coords,
              self.k_s,
              self.k_t,
              self.rot_mat,
              self.trans_mat,
              focal_disps=self.focal_disps,
              compose_layers=use_compose_splat,
              trg_downsampling=opts.trg_splat_downsampling,
              zbuf_scale=opts.zbuf_scale,
              bg_layer_disp=opts.bg_layer_disp,
              max_disp=opts.max_disp)
        else:
          to_recons_img = self.imgs_src
          recons_splat, _ = ldi_utils.forward_splat(
              self.ldi_trg,
              self.pixel_coords,
              self.k_t,
              self.k_s,
              self.inv_rot_mat,
              self.inv_trans_mat,
              focal_disps=self.focal_disps,
              compose_layers=use_compose_splat,
              trg_downsampling=opts.trg_splat_downsampling,
              zbuf_scale=opts.zbuf_scale,
              bg_layer_disp=opts.bg_layer_disp,
              max_disp=opts.max_disp)

        ## Forward splatting loss
        to_recons_img_downsampled = tf.image.resize_images(
            to_recons_img,
            recons_splat.get_shape().as_list()[2:4],
            method=tf.image.ResizeMethod.AREA)
        pwise_splat_loss = tf.reduce_min(  # min across layers
            tf.reduce_mean(  # mean across channels
                tf.abs(to_recons_img_downsampled - recons_splat),
                axis=4),
            axis=0)
        # ignore loss around boundary splat_bdry_ignore
        _, loss_h, loss_w = pwise_splat_loss.get_shape().as_list()
        x_min = int(round(loss_w * opts.splat_bdry_ignore))
        x_max = loss_w - x_min
        y_min = int(round(loss_h * opts.splat_bdry_ignore))
        y_max = loss_h - y_min
        pwise_splat_loss = pwise_splat_loss[:, y_min:y_max, x_min:x_max]

        if use_compose_splat:
          self.compose_splat_loss += tf.reduce_mean(pwise_splat_loss)
        else:
          self.indep_splat_loss += tf.reduce_mean(pwise_splat_loss)

        # Visualize splatted images
        if use_compose_splat:
          tf.summary.image(loss_img_name + '_splat',
                           tf.cast(recons_splat[0] * 255, tf.uint8))
        else:
          for l in range(recons_splat.get_shape().as_list()[0]):
            tf.summary.image(loss_img_name + '_splat_' + str(l),
                             tf.cast(recons_splat[l] * 255, tf.uint8))

        # Visualize splat losses
        lwise_splat_loss = tf.reduce_mean(  # mean across channels
            tf.abs(to_recons_img_downsampled - recons_splat),
            axis=4,
            keep_dims=True)
        tf.summary.image(
            loss_img_name + '_' + sp_loss_type + '_splat_loss',
            tf.reduce_min(  # min across layers
                lwise_splat_loss, axis=0))
        tf.summary.image(
            loss_img_name + '_' + sp_loss_type + '_splat_loss_index',
            tf.cast(
                tf.argmin(  # min across layers
                    lwise_splat_loss, axis=0),
                tf.float32))
        if lwise_splat_loss.get_shape().as_list()[0] == 2:
          # see how much the 2nd layer helps
          loss_diff = tf.nn.relu(lwise_splat_loss[0] - lwise_splat_loss[1])
          tf.summary.image(loss_img_name + '_splat_loss_diff', loss_diff)

    ## Smooth Disp Loss
    self.disp_smoothness_loss = 0
    self.disp_smoothness_loss += ldi_utils.disp_smoothness_loss(self.ldi_src[2])
    self.disp_smoothness_loss += ldi_utils.disp_smoothness_loss(self.ldi_trg[2])

    ## Increasing disp loss
    self.incr_depth_loss = 0.0
    self.incr_depth_loss += loss.decreasing_disp_loss(self.ldi_src[2])
    self.incr_depth_loss += loss.decreasing_disp_loss(self.ldi_trg[2])

    self.total_loss = 0.0
    if opts.self_cons_wt > 0:
      self.total_loss += opts.self_cons_wt * self.self_cons_loss
    if opts.compose_splat_wt > 0:
      self.total_loss += opts.compose_splat_wt * self.compose_splat_loss
    if opts.indep_splat_wt > 0:
      self.total_loss += opts.indep_splat_wt * self.indep_splat_loss
    if opts.incr_depth_wt > 0:
      incr_loss_wt = opts.incr_depth_wt / opts.max_disp
      self.total_loss += incr_loss_wt * self.incr_depth_loss
    if opts.disp_smoothness_wt > 0:
      smoothness_wt = opts.disp_smoothness_wt / (opts.max_disp * opts.max_disp)
      self.total_loss += smoothness_wt * self.disp_smoothness_loss


def main(_):
  FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.exp_name)
  if FLAGS.dataset == 'synthetic':
    FLAGS.bg_layer_disp = 2e-1
    FLAGS.depth_softmax_temp = 0.4
    if FLAGS.max_disp == 0:
      FLAGS.max_disp = 1.

  elif FLAGS.dataset == 'kitti':
    FLAGS.bg_layer_disp = 1e-3
    FLAGS.depth_softmax_temp = 0.4
    if FLAGS.max_disp == 0:
      FLAGS.max_disp = 0.4

  FLAGS.disp_vis_scale = 255 / FLAGS.max_disp
  trainer = Trainer(FLAGS)
  trainer.train()


if __name__ == '__main__':
  app.run(main)
