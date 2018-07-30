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

"""Misc helper functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import logging as log


def optimistic_restorer(save_file, vars_all=None):
  """Restores the variables which are present in save_file.

  Args:
    save_file: checkpoint file
    vars_all: optional list of the variable superset

  Returns:
    restorer: object on which we should call restore(sess, save_file)
  """
  if vars_all is None:
    vars_all = tf.global_variables()
  reader = tf.train.NewCheckpointReader(save_file)
  saved_shapes = reader.get_variable_to_shape_map()
  var_names = sorted([
      (var.name, var.name.split(':')[0]
      ) for var in vars_all if var.name.split(':')[0] in saved_shapes
  ])
  var_names_new = sorted([
      var.name for var in vars_all if var.name.split(':')[0] not in saved_shapes
  ])
  log.info('Number of new variables: ' + str(len(var_names_new)))
  log.info(var_names_new)
  restore_vars = []
  name2var = dict(zip(
      [x.name.split(':')[0] for x in tf.global_variables()],
      tf.global_variables()
  ))
  with tf.variable_scope('', reuse=True):
    for var_name, saved_var_name in var_names:
      curr_var = name2var[saved_var_name]
      var_shape = curr_var.get_shape().as_list()
      if var_shape == saved_shapes[saved_var_name]:
        restore_vars.append(curr_var)
      else:
        log.info('Different shape than saved: ' + var_name)
  restorer = tf.train.Saver(restore_vars)
  return restorer


def transpose(rot):
  """Transposes last two dimensions.

  Args:
      rot: relative rotation, are [...] X M X N matrices
  Returns:
      rot_t: [...] X N X M matrices
  """
  with tf.name_scope('transpose'):
    n_inp_dim = len(rot.get_shape())
    perm = range(n_inp_dim)
    perm[-1] = n_inp_dim - 2
    perm[-2] = n_inp_dim - 1
    rot_t = tf.transpose(rot, perm=perm)
    return rot_t


def divide_safe(num, den, name=None):
  eps = 1e-8
  den += eps*tf.cast(tf.equal(den, 0), 'float32')
  return tf.divide(num, den, name=name)


def leaky_relu(x, alpha=0.2):
  """Leaky ReLU.

  Args:
    x: input tensor
    alpha: negative slope
  Returns:
    x_bar : post-activation tensor
  """
  return tf.nn.relu(x) - alpha*(tf.nn.relu(-x))


def pixel_coords(bs, h, w):
  """Creates a bs X h X w X 3 tensor with (x,y,1) coord at each pixel.

  Args:
    bs: batch_size (number of meshgrid repetitions)
    h: number of rows
    w: number of columns
  Returns:
    bs X h X w X 3 tensor with (x,y,1) coord at each pixel.
    Note : these coordinates are 0.5 indexed
  """
  with tf.name_scope('pixel_coords'):
    ones_w = tf.ones((1, 1, w))
    ones_h = tf.ones((1, h, 1))
    ones_b = tf.ones((bs, 1, 1))

    range_h = tf.cast(tf.reshape(tf.range(h) + 1, (1, h, 1)), 'float32')
    range_w = tf.cast(tf.reshape(tf.range(w) + 1, (1, 1, w)), 'float32')

    # subtracting 0.5 so that pixel centres correspond to 0.5
    # for example, the top left pixel centre is at (0.5, 0.5)
    ys = ones_b*range_h*ones_w - 0.5
    xs = ones_b*ones_h*range_w - 0.5
    ones = ones_b*ones_h*ones_w

    return tf.stack([xs, ys, ones], axis=3)


def transform_pts(pts_coords_init, tform_mat):
  """Transforms input points according to the transformation.

  Args:
      pts_coords_init : [...] X H X W X D; pixelwise coordinates.
      tform_mat : [...] X D X D; desired matrix transformation
  Returns:
      pts_coords : [...] X H X W X D; transformed coordinates.
  """
  with tf.name_scope('transform_pts'):
    tform_mat_size = tform_mat.get_shape().as_list()
    n_dims_t = len(tform_mat_size)
    pts_init_size = pts_coords_init.get_shape().as_list()
    pts_transform_size = [tform_mat_size[ix] for ix in range(n_dims_t)]
    pts_transform_size[-2] = -1

    pts_coords_init_reshape = tf.reshape(pts_coords_init, pts_transform_size)

    tform_mat_transpose = transpose(tform_mat)
    pts_mul = tf.matmul(pts_coords_init_reshape, tform_mat_transpose)
    pts_coords_transformed = tf.reshape(pts_mul, pts_init_size)
    return pts_coords_transformed


def soft_z_buffering(layer_masks, layer_disps, depth_softmax_temp=1):
  """Computes pixelwise probability for belonging to each layer.

  Args:
    layer_masks: L X [...] X 1, indicating which layer pixels are valid
    layer_disps: L X [...] X 1, laywewise per pixel disparity
    depth_softmax_temp: Denominator for exponentiation of negative depths
  Returns:
    layer_probs: L X [...] X 1, indicating prob. of layer assignment
  """
  eps = 1e-8
  layer_disps = tf.nn.relu(layer_disps)
  layer_depths = divide_safe(1, layer_disps)
  log_depth_probs = -layer_depths/depth_softmax_temp

  log_layer_probs = tf.log(layer_masks + eps) + log_depth_probs
  log_layer_probs -= tf.reduce_max(log_layer_probs, axis=0, keep_dims=True)
  layer_probs = tf.exp(log_layer_probs)
  probs_sum = tf.reduce_sum(layer_probs, axis=0, keep_dims=True)
  layer_probs = tf.divide(layer_probs, probs_sum)
  return layer_probs


def enforce_inreasing_depth(
    disp_deltas, use_prod_disp_step=False, base_disp_reduction=0):
  """Compute decreasing, non-negative inverse depths based on disparity deltas.

  Args:
    disp_deltas: L X [...] disparity deltas
        the 1st prediction represents the predicted inverse depth of 1st layer
        the later [L-1] represent succesive reductions in inverse depth
    use_prod_disp_step: disps of succesive layers vary multiplicatively
    base_disp_reduction: min difference between succesive disps (unless < 0)
  Returns:
    disps_ldi: L X [..], non-negative inverse depths
  """
  disp_deltas = tf.nn.relu(disp_deltas)
  n_layers = disp_deltas.get_shape().as_list()[0]
  if n_layers == 1:
    return disp_deltas
  elif use_prod_disp_step:
    disp_deltas = tf.clip_by_value(disp_deltas, 1e-3, 1)
    return tf.cumprod(disp_deltas, axis=0, exclusive=False)
  else:
    disp_deltas += base_disp_reduction
    disp_layer0, _ = tf.split(disp_deltas, [1, n_layers-1], axis=0)
    disp_deltas_cumulative = tf.cumsum(disp_deltas, axis=0, exclusive=False)
    disps_ldi = tf.nn.relu(
        2*disp_layer0 - disp_deltas_cumulative - base_disp_reduction)
    return disps_ldi


def enforce_bg_occupied(ldi_masks):
  """Enforce that the last layer's mask has all ones.

  Args:
    ldi_masks: L X [...] masks
  Returns:
    ldi_masks: L X [..], masks with last layer set as 1
  """
  n_layers = ldi_masks.get_shape().as_list()[0]
  if n_layers == 1:
    return ldi_masks*0 + 1
  else:
    masks_fg, masks_bg = tf.split(ldi_masks, [n_layers-1, 1], axis=0)
    masks_bg = masks_bg*0 + 1
    return tf.concat([masks_fg, masks_bg], axis=0)


def zbuffer_weights(disps, scale=50):
  """Compute decreasing, non-negative inverse depths based on disparity deltas.

  Args:
    disps: [...] inverse depths, between 0 (far) and 1 (closest possible depth)
    scale: multiplicative factor before exponentiation
  Returns:
    z_buf_wts: [..], higher weights for closer things
  """
  pos_disps = tf.cast(tf.greater(disps, 0), tf.float32)
  disps = tf.clip_by_value(disps, 0, 1)
  disps -= 0.5  # subtracting a constant just divides all weights by a fraction
  wts = tf.exp(disps*scale)*pos_disps
  return wts
