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

"""Module for layer transformations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lsi.geometry import homography
from lsi.geometry import sampling
from lsi.nnutils import helpers as nn_helpers


def compose(
    imgs, masks, dmaps, soft=False, min_disp=1e-6, depth_softmax_temp=1):
  """composes layer images to output a single image.

  Args:
    imgs: are L X [...] X C, typically RGB images per layer
    masks: L X [...] X 1, indicating which layer pixels are valid
    dmaps: L X [...] X 1, indicating per pixel inverse depth
    soft: Whether to average (soft=True) or select argmax
    min_disp: Assumed disparity for the bg plane
    depth_softmax_temp: Denominator for exponentiation of negative depths
  Returns:
    [...] X C images after weighted sampling each layer per pixel.
  Assumes the first dimension corresponds to layers.
  """
  n_layers = imgs.get_shape().as_list()[0]
  shape_bg_img = imgs.get_shape().as_list()
  shape_bg_img[0] = 1

  shape_bg_mask = masks.get_shape().as_list()
  shape_bg_mask[0] = 1

  with tf.name_scope('layer_composition'):
    dmaps = tf.nn.relu(dmaps)

    bg_img = tf.ones(shape_bg_img)
    bg_mask = tf.ones(shape_bg_mask)
    bg_disp = tf.ones(shape_bg_mask)*min_disp

    imgs = tf.concat([imgs, bg_img], 0)
    masks = tf.concat([masks, bg_mask], 0)
    dmaps = tf.concat([dmaps, bg_disp], 0)

    selection_mask = nn_helpers.soft_z_buffering(
        masks, dmaps, depth_softmax_temp=depth_softmax_temp)

    if not soft:
      selection_mask = tf.one_hot(tf.argmax(selection_mask), n_layers+1, axis=0)

    avg_img = tf.reduce_sum(selection_mask*imgs, axis=0)
    return avg_img


def compose_depth(
    masks, dmaps, bg_layer=False, min_disp=1e-6, depth_softmax_temp=1):
  """composes layer images to output a single depth maps.

  Args:
    masks: L X [...] X 1, indicating which layer pixels are valid
    dmaps: L X [...] X 1, indicating per pixel inverse depth
    bg_layer: Whether to render the background layer's disp
    min_disp: Assumed disparity for the bg plane
    depth_softmax_temp: Denominator for exponentiation of negative depths
  Returns:
    [...] X 1 inverse depth map.
  Assumes the first dimension corresponds to layers.
  """
  n_layers = masks.get_shape().as_list()[0]

  shape_bg_mask = masks.get_shape().as_list()
  shape_bg_mask[0] = 1

  with tf.name_scope('layer_composition_depth'):
    dmaps = tf.nn.relu(dmaps)

    bg_mask = tf.ones(shape_bg_mask)
    bg_disp = tf.ones(shape_bg_mask)*min_disp

    masks = tf.concat([masks, bg_mask], 0)
    dmaps = tf.concat([dmaps, bg_disp], 0)

    if bg_layer:
      dmaps_selection = tf.reduce_max(dmaps) - dmaps[0:n_layers]
      dmaps_selection = tf.concat([dmaps_selection, bg_disp], 0)
    else:
      dmaps_selection = dmaps
    selection_mask = nn_helpers.soft_z_buffering(
        masks, dmaps_selection, depth_softmax_temp=depth_softmax_temp)

    selection_mask = tf.one_hot(tf.argmax(selection_mask), n_layers+1, axis=0)

    depth_img = tf.reduce_sum(selection_mask*dmaps, axis=0)
    return depth_img


def planar_transform(imgs, masks, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a):
  """transforms imgs, masks and computes dmaps according to planar transform.

  Args:
    imgs: are L X [...] X C, typically RGB images per layer
    masks: L X [...] X 1, indicating which layer pixels are valid
    pixel_coords_trg: [...] X H_t X W_t X 3;
        pixel (u,v,1) coordinates of target image pixels.
    k_s: intrinsics for source cameras, are [...] X 3 X 3 matrices
    k_t: intrinsics for target cameras, are [...] X 3 X 3 matrices
    rot: relative rotation, are [...] X 3 X 3 matrices
    t: [...] X 3 X 1, translations from source to target camera
    n_hat: L X [...] X 1 X 3, plane normal w.r.t source camera frame
    a: L X [...] X 1 X 1, plane equation displacement
  Returns:
    imgs_transformed: L X [...] X C images in trg frame
    masks_transformed: L X [...] X 1 masks in trg frame
    dmaps_trg: L X [...] X 1, indicating per pixel inverse depth
  Assumes the first dimension corresponds to layers.
  """
  with tf.name_scope('planar_transform'):
    n_layers = imgs.get_shape().as_list()[0]
    rot_rep_dims = [n_layers]
    rot_rep_dims += [1 for _ in range(len(k_s.get_shape()))]

    cds_rep_dims = [n_layers]
    cds_rep_dims += [1 for _ in range(len(pixel_coords_trg.get_shape()))]

    k_s = tf.tile(tf.expand_dims(k_s, axis=0), rot_rep_dims)
    k_t = tf.tile(tf.expand_dims(k_t, axis=0), rot_rep_dims)
    t = tf.tile(tf.expand_dims(t, axis=0), rot_rep_dims)
    rot = tf.tile(tf.expand_dims(rot, axis=0), rot_rep_dims)
    pixel_coords_trg = tf.tile(tf.expand_dims(
        pixel_coords_trg, axis=0), cds_rep_dims)

    ndims_img = len(imgs.get_shape())
    imgs_masks = tf.concat([imgs, masks], axis=ndims_img-1)
    imgs_masks_trg = homography.transform_plane_imgs(
        imgs_masks, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a)
    imgs_trg, masks_trg = tf.split(imgs_masks_trg, [3, 1], axis=ndims_img-1)

    dmaps_trg = homography.trg_disp_maps(
        pixel_coords_trg, k_t, rot, t, n_hat, a)

    return imgs_trg, masks_trg, dmaps_trg

