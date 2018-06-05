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


def ldi_join(ldi):
  """Concatenate the ldi channels to form a single Tensor."""
  return tf.concat(ldi, axis=-1)


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


def ldi_split(ldi_joined):
  """Split a joined LDI."""
  ndims = len(ldi_joined.get_shape())
  nc = ldi_joined.get_shape().as_list()[-1]
  imgs, masks, disps = tf.split(ldi_joined, [nc-2, 1, 1], axis=ndims-1)
  return [imgs, masks, disps]


def inverse_sample(ldi_src, disps_trg, pixel_coords_trg, k_s, k_t, rot, t):
  """sample source ldi at locations corresponding to projected trg coordinates.

  Args:
    ldi_src: Source frame LDI image
    disps_trg: are B X H_t X W_t X 1, inverse depths
    pixel_coords_trg: B X H_t X W_t X 3;
        pixel (u,v,1) coordinates of target image pixels.
    k_s: intrinsics for source cameras, are B X X 3 X 3 matrices
    k_t: intrinsics for target cameras, are B X 3 X 3 matrices
    rot: relative rotation, are B X 3 X 3 matrices
    t: B X 3 X 1, translations from source to target camera
  Returns:
    sample_ldi: {ldi00, ldi01, ldi10, ldi11}
    sample_wts: {wts00, wts01, wts10, wts11}
        We return 4 ldi's corresponding to the 4 surrounding pixel coordinates.
    disps_src: Pixel-wise disparity of trg pixels when projected into src frame
  """
  with tf.name_scope('ldi_inverse_sampling'):
    ndims = len(pixel_coords_trg.get_shape())
    nlayers = ldi_src[0].get_shape().as_list()[0]
    coords_rep_mult = tf.ones([nlayers] + [1 for _ in range(ndims)])

    trg2src_mat = projection.inverse_projection_matrix(k_s, k_t, rot, t)
    coords_trg = tf.concat([pixel_coords_trg, disps_trg], axis=-1)
    coords_src = nn_helpers.transform_pts(coords_trg, trg2src_mat)

    uv_coords_src, normalizer, disps_src = tf.split(
        coords_src, [2, 1, 1], axis=ndims-1)
    uv_coords_src = nn_helpers.divide_safe(uv_coords_src, normalizer)
    disps_src = nn_helpers.divide_safe(disps_src, normalizer)

    uv_coords_src = tf.expand_dims(uv_coords_src, axis=0)
    uv_coords_src *= coords_rep_mult  # becomes L X B X H X W X 2

    ldi_src_joined = ldi_join(ldi_src)
    sampled_imgs, sampled_wts = sampling.bilinear_wrapper(
        ldi_src_joined, uv_coords_src, compose=False)
    sampled_ldis = [ldi_split(si) for si in sampled_imgs]
    # same weights are repeated L times, we select the first one
    sampled_wts = [wt[0] for wt in sampled_wts]
    return sampled_ldis, sampled_wts, disps_src


def inverse_sample_points(ldi_src, point_coords, k_s, rot, t):
  """sample source ldi at locations corresponding to projected trg coordinates.

  Args:
    ldi_src: Source frame LDI image
    point_coords: B X N X 3, 3D coordinates of points in world frame
    k_s: intrinsics for source cameras, are B X X 3 X 3 matrices
    rot: rotation matrix from world to camera frame, are B X 3 X 3 matrices
    t: B X 3 X 1, translations from world to camera frame
      rot*p + t, where p is in world frame, transforms p to camera frame
  Returns:
    sample_tex: L X B X N X C sampled per-layers textures for each point
    sample_disp: L X B X N X 1 sampled per-layers inverse depths for each point
    proj_disp: B X N X 1 inverse depths for points projected
        (geometrically computed)
  """
  with tf.name_scope('inverse_sample_points'):
    nlayers = ldi_src[0].get_shape().as_list()[0]
    bs = point_coords.get_shape().as_list()[0]
    n_pts = point_coords.get_shape().as_list()[1]

    # pad with ones to get homogeneous coords
    ones_pad = tf.ones((bs, n_pts, 1))
    coords_world = tf.concat([point_coords, ones_pad], axis=-1)
    # reshape to use existing codebase which
    # requires B X H X W X 4 format for points
    coords_world = tf.reshape(coords_world, [bs, n_pts, 1, 4])

    ndims = len(coords_world.get_shape())
    coords_rep_mult = tf.ones([nlayers] + [1 for _ in range(ndims)])

    world2src_mat = tf.matmul(
        projection.pad_intrinsic(k_s),
        projection.pad_extrinsic(rot, t))

    coords_src = nn_helpers.transform_pts(coords_world, world2src_mat)

    uv_coords_src, normalizer, proj_disp = tf.split(
        coords_src, [2, 1, 1], axis=ndims-1)
    uv_coords_src = nn_helpers.divide_safe(uv_coords_src, normalizer)
    proj_disp = nn_helpers.divide_safe(proj_disp, normalizer)
    proj_disp = proj_disp[:, :, 0, :]

    uv_coords_src = tf.expand_dims(uv_coords_src, axis=0)
    uv_coords_src *= coords_rep_mult  # becomes L X B X N X 1 X 2
    ldi_src_joined = ldi_join(ldi_src)

    sampled_imgs = sampling.bilinear_wrapper(
        ldi_src_joined, uv_coords_src, compose=True)
    sampled_ldis = ldi_split(sampled_imgs)

    sample_tex = sampled_ldis[0][:, :, :, 0, :]
    sample_disp = sampled_ldis[2][:, :, :, 0, :]
    return sample_tex, sample_disp, proj_disp


def pixelwise_discrepancy(
    ldi, img, disp, disp_diff_wt=1, pix_diff_wt=1, unmatch_cost=10):
  """For each pixel, compute the min error w.r.t the layers.

  Args:
    ldi: [imgs, masks, disps]
        [L X B X H X W X C, L X B X H X W X 1, L X B X H X W X 1]
    img: B X H X W X C textures
    disp: B X H X W X 1 inverse depth
    disp_diff_wt: Cost associated with disparity discrepancy
    pix_diff_wt: Cost associated with pixel value discrepancy
    unmatch_cost: Cost for matching to an invalid ldi pixel (mask=0)
  Returns:
    pixelwise_cost: B X H X W X 1
    pixelwise_layer: B X H X W X 1 (index of selected layer)
  """
  with tf.name_scope('pixelwise_discrepancy'):
    imgs_ldi, masks_ldi, disps_ldi = ldi
    img = tf.expand_dims(img, axis=0)
    disp = tf.expand_dims(disp, axis=0)

    # Add a bg layer with white texture and mask=1
    imgs_ldi = tf.concat([imgs_ldi, tf.ones(img.get_shape())], axis=0)
    masks_ldi = tf.concat([masks_ldi, tf.ones(disp.get_shape())], axis=0)
    disps_ldi = tf.concat([disps_ldi, tf.zeros(disp.get_shape())], axis=0)
    n_layers = imgs_ldi.get_shape().as_list()[0]

    # img_cost = tf.reduce_mean(
    #     tf.square(imgs_ldi-img), axis=-1, keep_dims=True)
    # disp_cost = tf.square(disps_ldi-disp)
    img_cost = tf.reduce_mean(tf.abs(imgs_ldi-img), axis=-1, keep_dims=True)
    disp_cost = tf.abs(disps_ldi-disp)
    layerwise_cost = masks_ldi*(pix_diff_wt*img_cost + disp_diff_wt*disp_cost)
    layerwise_cost += (1-masks_ldi)*unmatch_cost
    pixelwise_cost = tf.reduce_min(layerwise_cost, axis=0, keep_dims=False)
    pixelwise_layer = tf.argmin(layerwise_cost, axis=0)
    layer_onehot = tf.one_hot(pixelwise_layer, n_layers, axis=0)
    argmin_img = tf.reduce_sum(layer_onehot*imgs_ldi, axis=0, keep_dims=False)
    argmin_disp = tf.reduce_sum(layer_onehot*disps_ldi, axis=0, keep_dims=False)

    # print(layer_onehot.get_shape().as_list())

    # print(imgs_ldi.get_shape().as_list())
    # print(argmin_img.get_shape().as_list())

    # print(disps_ldi.get_shape().as_list())
    # print(argmin_disp.get_shape().as_list())
    return pixelwise_cost, (pixelwise_layer, argmin_img, argmin_disp)


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
