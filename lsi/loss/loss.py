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

"""TensorFlow utils for loss function implementations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lsi.nnutils import helpers as nn_helpers


def event_prob(layer_masks):
  """Per-pixel layer assignment probs using an ordered multiplication.

  Args:
    layer_masks: L X [...] X 1, indicating which layer pixels are valid
  Returns:
    layer_probs: L X [...] X 1, indicating prob. of layer assignment
    escape_probs: 1 X [...] X 1, indicating prob that no layer is assigned
  """
  with tf.name_scope('event_prob'):
    eps = 1e-6
    # so that masks are not exactly 0 or 1
    # layer_masks = (1-eps)*layer_masks + 0.5*eps
    layer_masks = tf.clip_by_value(layer_masks, eps, 1-eps)

    # layer_masks = tf.Print(
    #     layer_masks,
    #     [tf.reduce_mean(layer_masks)], message='layer_masks mean')

    log_inv_m = tf.log(1-layer_masks)

    # log_inv_m = tf.Print(
    #     log_inv_m, [tf.reduce_mean(log_inv_m)], message='log_inv_m mean')

    log_prob = tf.cumsum(log_inv_m, axis=0) - log_inv_m + tf.log(layer_masks)

    # log_prob = tf.Print(
    #     log_prob, [tf.reduce_mean(log_prob)], message='log_prob mean')

    layer_probs = tf.exp(log_prob, name='layer_probs')
    escape_probs = 1-tf.reduce_sum(layer_probs, axis=0, keep_dims=True)
    return layer_probs, escape_probs


def decreasing_disp_loss(layer_disps):
  """Penalizes if successive disparities across layers increase.

  Args:
    layer_disps: L X [...] X 1, laywewise per pixel disparity
  Returns:
    err: scalar error
  """
  n_layers = layer_disps.get_shape().as_list()[0]
  if n_layers == 1:
    return 0
  disps_pre = layer_disps[0:n_layers-1]
  disps_pre = tf.stop_gradient(disps_pre)
  disps_post = layer_disps[1:n_layers]
  disps_incr = tf.nn.relu(disps_post - disps_pre)
  return tf.reduce_mean(disps_incr)


def zbuffer_composition_loss(
    layer_imgs, layer_masks,
    layer_disps, trg_imgs,
    bg_layer_disp=0, max_disp=1, zbuf_scale=10):
  """Depth+Mask based composition loss between predictions and target.

  First computes per-pixel layer assignment probs using depth+masks based
  normalization, and then penalizes inconsistency in a weighed manner.
  Assumes a default white background image (to penalize the ray escaping).

  Args:
    layer_imgs: are L X [...] X C, typically RGB images per layer
    layer_masks: L X [...] X 1, indicating which layer pixels are valid
    layer_disps: L X [...] X 1, laywewise per pixel disparity
    trg_imgs: [...] X C targets
    bg_layer_disp: Assumed disparity for the bg plane
    max_disp: Used for normalization
    zbuf_scale: Denominator for exponentiation of negative depths
  Returns:
    err: scalar error
  """
  # add a layer with white color, disp=max_disp
  shape_bg_img = layer_imgs.get_shape().as_list()
  shape_bg_img[0] = 1

  shape_bg_mask = layer_masks.get_shape().as_list()
  shape_bg_mask[0] = 1

  with tf.name_scope('zbuffer_composition_loss'):
    bg_img = tf.ones(shape_bg_img)
    bg_mask = tf.ones(shape_bg_mask)
    bg_disp = tf.ones(shape_bg_mask)*bg_layer_disp

    layer_imgs = tf.concat([layer_imgs, bg_img], 0)
    layer_masks = tf.concat([layer_masks, bg_mask], 0)
    layer_disps = tf.concat([layer_disps, bg_disp], 0)

    layer_probs = nn_helpers.zbuffer_weights(
        layer_disps/max_disp, scale=zbuf_scale)*layer_masks
    probs_sum = tf.reduce_sum(layer_probs, axis=0, keep_dims=True)
    layer_probs = nn_helpers.divide_safe(layer_probs, probs_sum)

    layerwise_cost = tf.square(layer_imgs-trg_imgs)*layer_probs
    layerwise_cost = tf.reduce_sum(layerwise_cost, axis=0)
    layerwise_cost = 0.5*tf.reduce_mean(layerwise_cost)

    return layerwise_cost

