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

"""Module for layered depth images.

An LDI will be represented as {textures, masks, disps} where
textures: L X B X H X W X C
masks: L X B X H X W X 1 (typically binary values)
disps: L X B X H X W X 1 (inverse depth, typically decreasing across layers)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lsi.geometry import projection
from lsi.geometry import sampling
from lsi.nnutils import helpers as nn_helpers


def gradient(pred):
  """Compute x and y gradients.

  Args:
    pred:  L X B X H X W X C
  Returns:
    dx: x gradient
    dy: y gradient
  """
  dy = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
  dx = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
  return dx, dy


def disp_smoothness_loss(pred_disp):
  """Penalize the gradients of the gradients.

  Args:
    pred_disp:  L X B X H X W X 1
  Returns:
    dx: x gradient
    dy: y gradient
  """
  dx, dy = gradient(pred_disp)
  dx2, dxdy = gradient(dx)
  dydx, dy2 = gradient(dy)

  dx2_loss = tf.reduce_mean(tf.abs(dx2))
  dxdy_loss = tf.reduce_mean(tf.abs(dxdy))
  dydx_loss = tf.reduce_mean(tf.abs(dydx))
  dy2_loss = tf.reduce_mean(tf.abs(dy2))
  # dx2_loss = tf.Print(dx2_loss, [dx2_loss], message='dx2_loss')
  # dxdy_loss = tf.Print(dxdy_loss, [dxdy_loss], message='dxdy_loss')
  # dydx_loss = tf.Print(dydx_loss, [dydx_loss], message='dydx_loss')
  # dy2_loss = tf.Print(dy2_loss, [dy2_loss], message='dy2_loss')
  return dx2_loss + dxdy_loss + dydx_loss + dy2_loss


def forward_splat(
    ldi_src, pixel_coords_src, k_s, k_t, rot, t,
    focal_disps=None,
    compose_layers=True,
    compute_trg_disp=False,
    trg_downsampling=1, bg_layer_disp=0, max_disp=1, zbuf_scale=10):
  """Forward splat the ldi_src.

  Currently limited to output a target rendering of the same size as source.

  Args:
    ldi_src: Source frame LDI image
    pixel_coords_src: B X H_s X W_s X 3;
        pixel (u,v,1) coordinates of source image pixels.
    k_s: intrinsics for source cameras, are B X X 3 X 3 matrices
    k_t: intrinsics for target cameras, are B X 3 X 3 matrices
    rot: relative rotation, are B X 3 X 3 matrices
    t: B X 3 X 1, translations from source to target camera
    focal_disps: Optional B X 1 X 1 X 1 tensor s.t.
        src_disp -= focal_disps when computing pixel shifts. Used for lytro data
    compose_layers: Output a per-layer splat image or a single rendering
    compute_trg_disp: Output trg frame forward splatted disp
    trg_downsampling: Target image is reduced by a factor compared to src image
    bg_layer_disp: disparity for white background layer
    max_disp: disparity for closest possible depth - used for normalization
    zbuf_scale: Scale for zbuffer weight computation
  Returns:
    trg_img: nl_out X B X H_s X W_s X nC
    trg_wts: nl_out X B X H_s X W_s X 1
    trg_disp: nl_out X B X H_s X W_s X 1, forward splatted depth map
    where nl_out = 1 if combine_layers=True, nl_out = n_layers o.w.
  """
  with tf.name_scope('forward_splat'):
    imgs_ldi, masks_ldi, disps_ldi = ldi_src
    n_layers, bs, h_s, w_s, nc = imgs_ldi.get_shape().as_list()
    h_t = h_s*trg_downsampling
    w_t = w_s*trg_downsampling
    bg_wt = nn_helpers.zbuffer_weights(bg_layer_disp/max_disp, scale=zbuf_scale)
    # bg_wt = tf.Print(bg_wt, [bg_wt, bg_layer_disp, max_disp])
    # start with a white canvas but with small weight
    trg_img = []
    trg_wts = []
    trg_disp = []
    for l in range(n_layers):
      trg_img.append(
          tf.ones([bs, h_t, w_t, nc], dtype='float32')*bg_wt)
      trg_wts.append(
          tf.ones([bs, h_t, w_t, 1], dtype='float32')*bg_wt)
      trg_disp.append(
          tf.zeros([bs, h_t, w_t, 1], dtype='float32')*bg_wt)

    src2trg_mat = projection.forward_projection_matrix(k_s, k_t, rot, t)

    for l in range(n_layers):
      disps_l = disps_ldi[l]
      if focal_disps is not None:
        disps_l -= focal_disps

      coords_src = tf.concat([pixel_coords_src, disps_l], axis=-1)
      coords_trg = nn_helpers.transform_pts(coords_src, src2trg_mat)
      uv_coords_trg, normalizer, disps_trg = tf.split(
          coords_trg, [2, 1, 1], axis=3)
      uv_coords_trg = nn_helpers.divide_safe(uv_coords_trg, normalizer)
      uv_coords_trg *= trg_downsampling
      disps_trg = nn_helpers.divide_safe(disps_trg, normalizer)

      if focal_disps is not None:
        disps_trg += focal_disps

      pixel_wts = nn_helpers.zbuffer_weights(
          disps_trg/max_disp, scale=zbuf_scale)*masks_ldi[l]

      weighted_src_image = imgs_ldi[l] * pixel_wts
      weighted_disp_image = disps_trg * pixel_wts
      trg_img[l] = sampling.splat(
          weighted_src_image, uv_coords_trg, init_trg_image=trg_img[l])
      trg_wts[l] = sampling.splat(
          pixel_wts, uv_coords_trg, init_trg_image=trg_wts[l])
      trg_disp[l] = sampling.splat(
          weighted_disp_image, uv_coords_trg, init_trg_image=trg_disp[l])

    trg_img = tf.stack(trg_img)
    trg_wts = tf.stack(trg_wts)
    trg_disp = tf.stack(trg_disp)

    # trg_wts = tf.Print(
    #     trg_wts,
    #     [tf.reduce_mean(trg_wts)], message='trg_wts mean')

    trg_disp = nn_helpers.divide_safe(trg_disp, trg_wts)

    if compose_layers:
      trg_img = tf.reduce_sum(trg_img, axis=0, keep_dims=True)
      trg_wts = tf.reduce_sum(trg_wts, axis=0, keep_dims=True)
      # trg_disp = tf.reduce_sum(trg_disp, axis=0, keep_dims=True)
      trg_disp = tf.reduce_max(trg_disp, axis=0, keep_dims=True)

    trg_img = nn_helpers.divide_safe(trg_img, trg_wts)
    # trg_disp = nn_helpers.divide_safe(trg_disp, trg_wts)

    # trg_img = tf.Print(
    #     trg_img,
    #     [tf.reduce_mean(trg_img)], message='trg_img mean')
    if compute_trg_disp:
      return trg_img, trg_wts, trg_disp
    else:
      return trg_img, trg_wts
