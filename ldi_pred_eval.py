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
"""Script for evaluating LDI prediction experiment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging as log
from lsi.data.kitti import data as kitti_data
from lsi.data.syntheticPlanes import data as synthetic_planes
from lsi.geometry import ldi as ldi_utils
from lsi.geometry import projection
from lsi.nnutils import helpers as nn_helpers
from lsi.nnutils import nets
from lsi.nnutils import test_utils
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

## Experiment-specific flags.
test_utils.define_default_flags(flags)
flags.DEFINE_integer('n_layers', 3, 'Maximum number of layers.')
flags.DEFINE_boolean('pred_ldi_masks', False,
                     'Predict masks for LDIs or use 1s.')
flags.DEFINE_boolean('batch_norm_training', True,
                     'Use training mode for batch norm.')

## Flags specific to selecting data loader.
flags.DEFINE_enum('dataset', 'synthetic', ['synthetic', 'kitti'], 'Dataset')
flags.DEFINE_enum('data_split', 'val', ['all', 'train', 'val', 'test'],
                  'Dataset split.')
flags.DEFINE_boolean('debug_synth_texture', False,
                     'Use ground-truth LDI disparities instead of predicted '
                     'disparities.')

## Flags specific to synthetic data loader.
flags.DEFINE_string('pascal_objects_dir', '/code/lsi/cachedir/sbd/objects',
                    'Directory containing images of PASCAL objects.')
flags.DEFINE_string('sun_imgs_dir', '/datasets/SUN2012pascalformat/JPEGImages',
                    'Directory containing SUN dataset images.')
flags.DEFINE_integer('n_obj_min', 1,
                     'Minimum number of foreground layers in synthetic data.')
flags.DEFINE_integer('n_obj_max', 3,
                     'Maximum number of foreground layers in synthetic data.')
flags.DEFINE_integer('n_box_planes', 5, 'Number of planes to use from a 3D box '
                     'containing the scene.')
flags.DEFINE_integer(
    'synth_ds_factor', 1, 'Render synthetic data at a higher resolution and '
    'downsample by this factor (to achieve antialiased '
    'renders.)')
flags.DEFINE_boolean('synth_dl_eval_data', True,
                     'Output ground truth information for synthetic data.')

## Flags specific to kitti dataset.
flags.DEFINE_string('kitti_data_root', '/datasets/kitti',
                    'Directory containing KITTI images and cameras.')
flags.DEFINE_enum('kitti_dataset_variant', 'raw_city',
                  ['odom', 'mview', 'raw_city'], 'Kitti set to use.')
flags.DEFINE_boolean('kitti_dl_disparities', True,
                     'Output ground truth KITTI disparities.')

## Flags related to the training (loss, CNN architecture etc.).
flags.DEFINE_float('splat_bdry_ignore', 0,
                   'Ignore this fraction of pixels along the boundary.')
flags.DEFINE_float('zbuf_scale', 10, 'Scale for zbuffer weight computation.')
flags.DEFINE_float('trg_splat_downsampling', 1.0,
                   'The forward splatted image is downsampled by this factor.')
flags.DEFINE_boolean('use_unet', True,
                     'Whether to use a CNN with skip connections.')
flags.DEFINE_integer('n_layerwise_steps', 3,
                     'Number of independent per-layer up-convolution steps.')

## Dataset dependent flags : overridden in code.
flags.DEFINE_float('bg_layer_disp', 1e-6,
                   'Disparity of bg layer: value automatically chosen in code.')
flags.DEFINE_float(
    'disp_vis_scale', 255,
    'Disparity visualization scale: value automatically chosen in code.')
flags.DEFINE_float(
    'max_disp', 0, 'Inverse depth for closest plane. '
    'Value automatically chosen in code according to the dataset if set to 0.')


class Tester(test_utils.Tester):
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

    if opts.dataset == 'synthetic':
      img_height = opts.img_height
      img_width = opts.img_width
      bs = opts.batch_size

      self.src_gt_disp = tf.placeholder(
          tf.float32, [bs, img_height, img_width, 1], name='src_gt_disp')
      self.trg_gt_disp = tf.placeholder(
          tf.float32, [bs, img_height, img_width, 1], name='trg_gt_disp')

      self.src_gt_disp_bg = tf.placeholder(
          tf.float32, [bs, img_height, img_width, 1], name='src_gt_disp_bg')
      self.trg_gt_disp_bg = tf.placeholder(
          tf.float32, [bs, img_height, img_width, 1], name='trg_gt_disp_bg')

      self.src_gt_tex_bg = tf.placeholder(
          tf.float32, [bs, img_height, img_width, 3], name='src_gt_disp_bg')
      self.trg_gt_tex_bg = tf.placeholder(
          tf.float32, [bs, img_height, img_width, 3], name='trg_gt_disp_bg')

      src2trg_mat = projection.forward_projection_matrix(
          self.k_s, self.k_t, self.rot_mat, self.trans_mat)
      trg2src_mat = projection.inverse_projection_matrix(
          self.k_s, self.k_t, self.rot_mat, self.trans_mat)
      self.disocclusion_mask_src = projection.disocclusion_mask(
          self.src_gt_disp, self.trg_gt_disp, self.pixel_coords, src2trg_mat)
      self.disocclusion_mask_trg = projection.disocclusion_mask(
          self.trg_gt_disp, self.src_gt_disp, self.pixel_coords, trg2src_mat)

    if opts.dataset == 'kitti':
      img_height = opts.img_height
      img_width = opts.img_width
      bs = opts.batch_size

      self.src_gt_disp = tf.placeholder(
          tf.float32, [bs, img_height, img_width, 1], name='src_gt_disp')
      self.trg_gt_disp = tf.placeholder(
          tf.float32, [bs, img_height, img_width, 1], name='trg_gt_disp')
      self.disocclusion_mask_src = tf.equal(self.src_gt_disp, 0)
      self.disocclusion_mask_trg = tf.equal(self.trg_gt_disp, 0)

    self.focal_disps = None

    # Transform from trg to src frame.
    self.inv_rot_mat = nn_helpers.transpose(self.rot_mat)
    self.inv_trans_mat = -tf.matmul(self.inv_rot_mat, self.trans_mat)
    n_layers = opts.n_layers
    if opts.use_unet:
      _, feat_dec_src, skip_feat_src, _ = nets.encoder_decoder_unet(
          self.imgs_src,
          nl_diff_enc_dec=opts.n_layerwise_steps,
          is_training=opts.batch_norm_training)
      _, feat_dec_trg, skip_feat_trg, _ = nets.encoder_decoder_unet(
          self.imgs_trg,
          reuse=True,
          nl_diff_enc_dec=opts.n_layerwise_steps,
          is_training=opts.batch_norm_training)
    else:
      _, feat_dec_src, skip_feat_src, _ = nets.encoder_decoder_simple(
          self.imgs_src,
          nl_diff_enc_dec=opts.n_layerwise_steps,
          is_training=opts.batch_norm_training)
      _, feat_dec_trg, skip_feat_trg, _ = nets.encoder_decoder_simple(
          self.imgs_trg,
          reuse=True,
          nl_diff_enc_dec=opts.n_layerwise_steps,
          is_training=opts.batch_norm_training)
    self.ldi_src = nets.ldi_predictor(
        feat_dec_src,
        n_layers=n_layers,
        n_layerwise_steps=opts.n_layerwise_steps,
        skip_feat=skip_feat_src,
        pred_masks=opts.pred_ldi_masks,
        is_training=opts.batch_norm_training)
    self.ldi_src[2] *= opts.max_disp
    self.ldi_trg = nets.ldi_predictor(
        feat_dec_trg,
        n_layers=n_layers,
        reuse=True,
        n_layerwise_steps=opts.n_layerwise_steps,
        skip_feat=skip_feat_trg,
        pred_masks=opts.pred_ldi_masks,
        is_training=opts.batch_norm_training)
    self.ldi_trg[2] *= opts.max_disp

    # Select inverse depths predicted for the 1st layer of the LDIs.
    if opts.debug_synth_texture:
      self.ldi_src[2] = 0 * self.ldi_src[2] + 1 * self.src_ldi_gt_disps
      self.ldi_trg[2] = 0 * self.ldi_trg[2] + 1 * self.trg_ldi_gt_disps

    self.disps_src = self.ldi_src[2][0]
    self.disps_trg = self.ldi_trg[2][0]

  def feed(self):
    """Data loading wrapper.

    Returns:
      feed_dict: Loads data and returns feed dict for all input data to graph.
    """
    opts = self.opts
    data_batch = self.data_loader.forward(opts.batch_size)
    if opts.dataset == 'synthetic' and opts.synth_dl_eval_data:
      (img_src, img_trg, k_s, k_t, rot_mat, trans_mat, _, _, disp_s_fg,
       disp_s_bg, disp_t_fg, disp_t_bg, img_s_bg, img_t_bg) = data_batch
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

    if (opts.dataset == 'synthetic' and
        opts.synth_dl_eval_data) or (opts.dataset == 'kitti' and
                                     opts.kitti_dl_disparities):
      feed_dict[self.src_gt_disp] = disp_s_fg
      feed_dict[self.trg_gt_disp] = disp_t_fg

    if opts.dataset == 'synthetic' and opts.synth_dl_eval_data:
      feed_dict[self.src_gt_disp_bg] = disp_s_bg
      feed_dict[self.trg_gt_disp_bg] = disp_t_bg
      feed_dict[self.src_gt_tex_bg] = img_s_bg
      feed_dict[self.trg_gt_tex_bg] = img_t_bg

    if opts.debug_synth_texture:
      feed_dict[self.src_ldi_gt_disps] = np.stack(
          [disp_s_fg, disp_s_bg], axis=0)
      feed_dict[self.trg_ldi_gt_disps] = np.stack(
          [disp_t_fg, disp_t_bg], axis=0)

    return feed_dict

  def define_visuals(self):
    b = 0
    self.visuals['src_im'] = self.imgs_src[b]
    self.visuals['trg_im'] = self.imgs_trg[b]

    if self.opts.dataset == 'synthetic':
      self.visuals['trg_gt_disp_bg'] = self.trg_gt_disp_bg[b, :, :, 0]
      self.visuals['src_gt_disp_bg'] = self.src_gt_disp_bg[b, :, :, 0]
      self.visuals['trg_gt_tex_bg'] = self.trg_gt_tex_bg[b]
      self.visuals['src_gt_tex_bg'] = self.src_gt_tex_bg[b]

    for l in range(self.opts.n_layers):
      self.visuals['src_tex_pred_layer_%d' % l] = self.ldi_src[0][l][b]
      self.visuals['src_disp_pred_layer_%d' %
                   l] = (self.ldi_src[2][l][b, :, :, 0]) / self.opts.max_disp
      self.visuals['trg_tex_pred_layer_%d' % l] = self.ldi_trg[0][l][b]
      self.visuals['trg_disp_pred_layer_%d' %
                   l] = (self.ldi_trg[2][l][b, :, :, 0]) / self.opts.max_disp

  def define_pred_results(self):
    self.pred_results['src_im'] = self.imgs_src
    self.pred_results['trg_im'] = self.imgs_trg
    self.pred_results['src_disp_pred'] = self.ldi_src[2][0][:, :, :, 0]
    self.pred_results['trg_disp_pred'] = self.ldi_trg[2][0][:, :, :, 0]

  def define_metrics(self):
    """Metrics computation.
    """
    opts = self.opts

    # Reconstruction loss for trg/src image via src/trg image LDI.
    self.compose_splat_loss = 0
    compute_disoccluded_error = (
        opts.dataset == 'synthetic' or
        (opts.dataset == 'kitti' and opts.kitti_dl_disparities))
    compute_depth_error = (opts.dataset == 'synthetic')
    compute_bg_pred_error = (opts.dataset == 'synthetic')
    compute_fg_pred_error = (opts.dataset == 'synthetic')

    self.valid_pixel_count = 0

    if compute_depth_error:
      self.depth_splat_loss = 0

    if compute_depth_error and compute_disoccluded_error:
      self.depth_splat_loss_disocc = 0

    if compute_disoccluded_error:
      self.compose_splat_loss_disocc = 0
      self.valid_disocc_pixel_count = 0

    if compute_bg_pred_error:
      self.bg_tex_error = 0
      self.bg_disp_error = 0
      self.valid_bg_pixel_count = 0

    if compute_fg_pred_error:
      self.fg_tex_error = 0
      self.fg_disp_error = 0
      self.valid_fg_pixel_count = 0

    for loss_img_name in ['trg', 'src']:
      if loss_img_name == 'trg':
        to_recons_img = self.imgs_trg
        if compute_disoccluded_error:
          to_recons_disocc_mask = self.disocclusion_mask_trg
        if compute_depth_error:
          to_recons_disp = self.trg_gt_disp
        recons_splat, _, recons_disp = ldi_utils.forward_splat(
            self.ldi_src,
            self.pixel_coords,
            self.k_s,
            self.k_t,
            self.rot_mat,
            self.trans_mat,
            focal_disps=self.focal_disps,
            compose_layers=True,
            compute_trg_disp=True,
            trg_downsampling=opts.trg_splat_downsampling,
            zbuf_scale=opts.zbuf_scale,
            bg_layer_disp=opts.bg_layer_disp,
            max_disp=opts.max_disp)
        if opts.dataset == 'synthetic':
          valid_mask = tf.cast(
              tf.greater(self.trg_gt_disp, opts.bg_layer_disp), tf.float32)
        else:
          valid_mask = tf.ones((opts.batch_size, opts.img_height,
                                opts.img_width, 1))

      else:
        to_recons_img = self.imgs_src
        if compute_disoccluded_error:
          to_recons_disocc_mask = self.disocclusion_mask_src
        if compute_depth_error:
          to_recons_disp = self.src_gt_disp
        recons_splat, _, recons_disp = ldi_utils.forward_splat(
            self.ldi_trg,
            self.pixel_coords,
            self.k_t,
            self.k_s,
            self.inv_rot_mat,
            self.inv_trans_mat,
            focal_disps=self.focal_disps,
            compose_layers=True,
            compute_trg_disp=True,
            trg_downsampling=opts.trg_splat_downsampling,
            zbuf_scale=opts.zbuf_scale,
            bg_layer_disp=opts.bg_layer_disp,
            max_disp=opts.max_disp)
        if opts.dataset == 'synthetic':
          valid_mask = tf.cast(
              tf.greater(self.src_gt_disp, opts.bg_layer_disp), tf.float32)
        else:
          valid_mask = tf.ones((opts.batch_size, opts.img_height,
                                opts.img_width, 1))

      ## Forward splatting loss.
      to_recons_img_downsampled = tf.image.resize_images(
          to_recons_img,
          recons_splat.get_shape().as_list()[2:4],
          method=tf.image.ResizeMethod.AREA)
      valid_mask = tf.image.resize_images(
          valid_mask,
          recons_splat.get_shape().as_list()[2:4],
          method=tf.image.ResizeMethod.AREA)
      # Ignore pixels that might have aliasing.
      valid_mask = tf.cast(tf.greater(valid_mask, 0.95), tf.float32)
      valid_mask = valid_mask[:, :, :, 0]

      if compute_disoccluded_error:
        to_recons_disocc_mask = tf.cast(to_recons_disocc_mask, tf.float32)
        to_recons_disocc_mask_ds = tf.image.resize_images(
            to_recons_disocc_mask,
            recons_splat.get_shape().as_list()[2:4],
            method=tf.image.ResizeMethod.AREA)
        to_recons_disocc_mask = tf.cast(
            tf.greater(to_recons_disocc_mask, 0.95), tf.float32)
        to_recons_disocc_mask_ds = to_recons_disocc_mask_ds[:, :, :, 0]

      if compute_depth_error:
        to_recons_disp_ds = tf.image.resize_images(
            to_recons_disp,
            recons_splat.get_shape().as_list()[2:4],
            method=tf.image.ResizeMethod.AREA)
        pwise_depth_splat_loss = tf.reduce_min(  # min across layers
            tf.reduce_mean(  # mean across channels
                tf.abs(to_recons_disp_ds - recons_disp),
                axis=4),
            axis=0)

      pwise_splat_loss = tf.reduce_min(  # min across layers
          tf.reduce_mean(  # mean across channels
              tf.abs(to_recons_img_downsampled - recons_splat),
              axis=4),
          axis=0)

      # Ignore loss around boundary of size splat_bdry_ignore.
      _, loss_h, loss_w = pwise_splat_loss.get_shape().as_list()
      x_min = int(round(loss_w * opts.splat_bdry_ignore))
      x_max = loss_w - x_min
      y_min = int(round(loss_h * opts.splat_bdry_ignore))
      y_max = loss_h - y_min
      centre_mask = tf.ones((opts.batch_size, y_max - y_min, x_max - x_min))
      padding = tf.constant([[0, 0], [y_min, loss_h - y_max],
                             [x_min, loss_w - x_max]])
      centre_mask = tf.pad(centre_mask, padding)
      centre_mask *= valid_mask

      pwise_splat_loss *= centre_mask
      if compute_depth_error:
        pwise_depth_splat_loss *= centre_mask

      if compute_disoccluded_error:
        self.visuals['{}_disocc_mask'.format(
            loss_img_name)] = to_recons_disocc_mask_ds[0] * centre_mask[0]

        self.compose_splat_loss_disocc += tf.reduce_sum(
            pwise_splat_loss * to_recons_disocc_mask_ds)
        self.valid_disocc_pixel_count += tf.reduce_sum(
            centre_mask * to_recons_disocc_mask_ds)

      self.compose_splat_loss += tf.reduce_sum(pwise_splat_loss)
      self.valid_pixel_count += tf.reduce_sum(centre_mask)

      if compute_depth_error:
        self.depth_splat_loss += tf.reduce_sum(pwise_depth_splat_loss)

        if compute_disoccluded_error:
          self.depth_splat_loss_disocc += tf.reduce_sum(
              pwise_depth_splat_loss * to_recons_disocc_mask_ds)

      self.visuals['{}_vs_loss'.format(loss_img_name)] = pwise_splat_loss[0]
      self.visuals['{}_splat_tex'.format(loss_img_name)] = recons_splat[0, 0]

      if compute_depth_error:
        self.visuals['{}_gt_disp'.format(loss_img_name)] = to_recons_disp[
            0, :, :, 0]
        self.visuals['{}_splat_disp'.format(loss_img_name)] = recons_disp[
            0, 0, :, :, 0]
        self.visuals['{}_depth_loss'.format(
            loss_img_name)] = pwise_depth_splat_loss[0]

    self.metrics = {'compose_loss': self.compose_splat_loss}

    self.metrics_norm = {'compose_loss': self.valid_pixel_count}

    if compute_bg_pred_error:
      valid_mask_src = tf.cast(
          tf.greater(self.src_gt_disp, self.src_gt_disp_bg), tf.float32)
      valid_mask_trg = tf.cast(
          tf.greater(self.trg_gt_disp, self.trg_gt_disp_bg), tf.float32)

      src_bg_tex_error = tf.abs(self.ldi_src[0][opts.n_layers - 1] -
                                self.src_gt_tex_bg)
      trg_bg_tex_error = tf.abs(self.ldi_trg[0][opts.n_layers - 1] -
                                self.trg_gt_tex_bg)

      self.bg_tex_error += tf.reduce_sum(src_bg_tex_error * valid_mask_src) / 3
      self.bg_tex_error += tf.reduce_sum(trg_bg_tex_error * valid_mask_trg) / 3

      self.valid_bg_pixel_count += tf.reduce_sum(valid_mask_trg +
                                                 valid_mask_src)

      src_bg_disp_error = tf.abs(self.ldi_src[2][opts.n_layers - 1] -
                                 self.src_gt_disp_bg)
      trg_bg_disp_error = tf.abs(self.ldi_trg[2][opts.n_layers - 1] -
                                 self.trg_gt_disp_bg)

      self.bg_disp_error += tf.reduce_sum(src_bg_disp_error * valid_mask_src)
      self.bg_disp_error += tf.reduce_sum(trg_bg_disp_error * valid_mask_trg)

      self.metrics['bg_tex_error'] = self.bg_tex_error
      self.metrics['bg_disp_error'] = self.bg_disp_error

      self.metrics_norm['bg_tex_error'] = self.valid_bg_pixel_count
      self.metrics_norm['bg_disp_error'] = self.valid_bg_pixel_count

    if compute_fg_pred_error:
      valid_mask_src = tf.cast(
          tf.greater(self.src_gt_disp, opts.bg_layer_disp), tf.float32)
      valid_mask_trg = tf.cast(
          tf.greater(self.trg_gt_disp, opts.bg_layer_disp), tf.float32)
      self.visuals['valid_mask_src'] = valid_mask_src[0, :, :, 0]
      self.visuals['valid_mask_trg'] = valid_mask_trg[0, :, :, 0]

      src_fg_tex_error = tf.abs(self.ldi_src[0][0] - self.imgs_src)
      trg_fg_tex_error = tf.abs(self.ldi_trg[0][0] - self.imgs_trg)

      self.fg_tex_error += tf.reduce_sum(src_fg_tex_error * valid_mask_src) / 3
      self.fg_tex_error += tf.reduce_sum(trg_fg_tex_error * valid_mask_trg) / 3

      self.valid_fg_pixel_count += tf.reduce_sum(valid_mask_trg +
                                                 valid_mask_src)

      src_fg_disp_error = tf.abs(self.ldi_src[2][0] - self.src_gt_disp)
      trg_fg_disp_error = tf.abs(self.ldi_trg[2][0] - self.trg_gt_disp)

      self.fg_disp_error += tf.reduce_sum(src_fg_disp_error * valid_mask_src)
      self.fg_disp_error += tf.reduce_sum(trg_fg_disp_error * valid_mask_trg)

      self.metrics['fg_tex_error'] = self.fg_tex_error
      self.metrics['fg_disp_error'] = self.fg_disp_error

      self.metrics_norm['fg_tex_error'] = self.valid_fg_pixel_count
      self.metrics_norm['fg_disp_error'] = self.valid_fg_pixel_count

    if compute_disoccluded_error:
      self.metrics['compose_loss_disocc'] = self.compose_splat_loss_disocc
      self.metrics_norm['compose_loss_disocc'] = self.valid_disocc_pixel_count

    if compute_depth_error:
      self.metrics['depth_loss'] = self.depth_splat_loss
      self.metrics_norm['depth_loss'] = self.valid_pixel_count

      if compute_disoccluded_error:
        self.metrics['depth_loss_disocc'] = self.depth_splat_loss_disocc
        self.metrics_norm['depth_loss_disocc'] = self.valid_disocc_pixel_count


def main(_):
  FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.exp_name)
  FLAGS.results_eval_dir = os.path.join(FLAGS.results_eval_dir, FLAGS.exp_name)
  FLAGS.results_vis_dir = os.path.join(FLAGS.results_vis_dir, FLAGS.exp_name)

  if FLAGS.dataset == 'synthetic':
    FLAGS.bg_layer_disp = 2e-1
    if FLAGS.max_disp == 0:
      FLAGS.max_disp = 1.

  elif FLAGS.dataset == 'kitti':
    FLAGS.bg_layer_disp = 1e-3
    if FLAGS.max_disp == 0:
      FLAGS.max_disp = 0.4

  FLAGS.disp_vis_scale = 255 / FLAGS.max_disp
  tester = Tester(FLAGS)
  log.info('Init Testing')
  tester.test()


if __name__ == '__main__':
  app.run(main)
