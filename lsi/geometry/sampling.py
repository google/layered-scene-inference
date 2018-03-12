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

"""Module for bilinear sampling.

This implementation is based on
a previous implementation in the TensorFlow repo -
https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def _repeat(x, n_repeats):
  rep = tf.transpose(
      tf.expand_dims(tf.ones(shape=tf.stack([
          n_repeats,
      ])), 1), [1, 0])
  rep = tf.cast(rep, 'float32')
  x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
  return tf.reshape(x, [-1])


def bilinear(imgs, coords, compose=True):
  """bilinear sampling function.

  Args:
    imgs: are B X H_s X W_s X C
    coords: B X H_t X W_t X 2 indicating the source image pixels to copy from
    compose: whether to return one image, or return 4 images with weights
  Returns:
    B X H_t X W_t X C images after bilinear sampling from input.
      Coordinates outside the image are sampled as 0.
  """
  with tf.name_scope('bilinear_sampling'):

    coords -= 0.5
    # code below requires 0 in 'coords' -> first pixel
    # whereas actually, 0.5 in coords -> centre of first pixel

    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    inp_size = imgs.get_shape()
    coord_size = coords.get_shape()
    out_size = coords.get_shape().as_list()
    out_size[3] = imgs.get_shape().as_list()[3]

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    ## bilinear interp weights
    wt_x0 = (x1 - coords_x)
    wt_x1 = (coords_x - x0)
    wt_y0 = (y1 - coords_y)
    wt_y1 = (coords_y - y0)

    ## Whether points project outside the grid
    valid_x0 = tf.cast(tf.equal(x0, x0_safe), 'float32')
    valid_x1 = tf.cast(tf.equal(x1, x1_safe), 'float32')
    valid_y0 = tf.cast(tf.equal(y0, y0_safe), 'float32')
    valid_y1 = tf.cast(tf.equal(y1, y1_safe), 'float32')

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(
        _repeat(
            tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
            coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = x0_safe + base_y0
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    if compose:
      output = tf.add_n([
          valid_x0 * valid_y0 * wt_x0 * wt_y0 * im00,
          valid_x0 * valid_y1 * wt_x0 * wt_y1 * im01,
          valid_x1 * valid_y0 * wt_x1 * wt_y0 * im10,
          valid_x1 * valid_y1 * wt_x1 * wt_y1 * im11
      ])
    else:
      out_ims = [
          valid_x0 * valid_y0 * im00, valid_x0 * valid_y1 * im01,
          valid_x1 * valid_y0 * im10, valid_x1 * valid_y1 * im11
      ]
      out_wts = [wt_x0 * wt_y0, wt_x0 * wt_y1, wt_x1 * wt_y0, wt_x1 * wt_y1]
      output = out_ims, out_wts

    return output


def bilinear_wrapper(imgs, coords, compose=True):
  """Wrapper around bilinear sampling function, handles arbitrary input sizes.

  Args:
    imgs: are [...] X H_s X W_s X C
    coords: [...] X H_t X W_t X 2 indicating the source pixels to copy from
    compose: whether to return one image, or return 4 images with weights
  Returns:
    [...] X H_t X W_t X C images after bilinear sampling from input.
  """
  with tf.name_scope('bilinear_sampling_wrapper'):
    # the bilinear sampling code only handles 4D input, so we'll need to reshape
    init_dims = imgs.get_shape().as_list()[:-3:]
    end_dims_img = imgs.get_shape().as_list()[-3::]
    end_dims_coords = coords.get_shape().as_list()[-3::]
    prod_init_dims = init_dims[0]
    for ix in range(1, len(init_dims)):
      prod_init_dims *= init_dims[ix]

    imgs = tf.reshape(imgs, [prod_init_dims] + end_dims_img)
    coords = tf.reshape(
        coords, [prod_init_dims] + end_dims_coords)

    if compose:
      imgs_sampled = bilinear(imgs, coords, compose=compose)
      imgs_sampled = tf.reshape(
          imgs_sampled, init_dims + imgs_sampled.get_shape().as_list()[-3::])
      return imgs_sampled
    else:
      imgs_sampled, wts_sampled = bilinear(imgs, coords, compose=compose)
      imgs_shape = init_dims + imgs_sampled[0].get_shape().as_list()[-3::]
      wts_shape = init_dims + wts_sampled[0].get_shape().as_list()[-3::]
      imgs_sampled = [tf.reshape(t, imgs_shape) for t in imgs_sampled]
      wts_sampled = [tf.reshape(t, wts_shape) for t in wts_sampled]
      return imgs_sampled, wts_sampled


def splat(src_image, tgt_coords, init_trg_image):
  """Splat pixels from the src_image to target coordinates.

  Code from tinghuiz.
  Args:
    src_image: source image [batch, height_s, width_s, channels]
    tgt_coords: target coordinates [batch, height_s, width_s, 2]
    init_trg_image: initial target image [batch, height_t, width_t, channels]
  Returns:
    A new target image.
  """
  with tf.name_scope('forward_splat'):
    tgt_coords -= 0.5
    # code below requires 0 in 'coords' -> first pixel
    # whereas actually, 0.5 in coords -> centre of first pixel
    batch, h_src, w_src, channels = src_image.get_shape().as_list()
    _, h_trg, w_trg, _ = init_trg_image.get_shape().as_list()
    num_pixels_src = h_src * w_src
    num_pixels_trg = h_trg * w_trg
    x = tgt_coords[:, :, :, 0]
    y = tgt_coords[:, :, :, 1]

    x0 = tf.floor(x)
    x1 = x0 + 1
    y0 = tf.floor(y)
    y1 = y0 + 1

    y_max = tf.cast(h_trg, 'float32') - 1
    x_max = tf.cast(w_trg, 'float32') - 1
    zero = tf.zeros([1], dtype='float32')

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    # bilinear splat weights, with points outside the grid having weight 0
    wt_x0 = (x1 - x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
    wt_x1 = (x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
    wt_y0 = (y1 - y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
    wt_y1 = (y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

    wt_tl = wt_x0 * wt_y0
    wt_tr = wt_x1 * wt_y0
    wt_bl = wt_x0 * wt_y1
    wt_br = wt_x1 * wt_y1

    # Clamp small weights to zero for gradient numerical stability (IMPORTANT!)
    wt_tl *= tf.cast(tf.greater(wt_tl, 1e-3), 'float32')
    wt_tr *= tf.cast(tf.greater(wt_tr, 1e-3), 'float32')
    wt_bl *= tf.cast(tf.greater(wt_bl, 1e-3), 'float32')
    wt_br *= tf.cast(tf.greater(wt_br, 1e-3), 'float32')

    # Four copies of the value image weighted by bilinear weights
    values_tl = tf.reshape(src_image * wt_tl[:, :, :, None],
                           [batch, num_pixels_src, channels])
    values_tr = tf.reshape(src_image * wt_tr[:, :, :, None],
                           [batch, num_pixels_src, channels])
    values_bl = tf.reshape(src_image * wt_bl[:, :, :, None],
                           [batch, num_pixels_src, channels])
    values_br = tf.reshape(src_image * wt_br[:, :, :, None],
                           [batch, num_pixels_src, channels])

    inds_tl = tf.cast(
        tf.reshape(x0_safe + y0_safe * w_trg, [batch, -1]), 'int32')
    inds_tr = tf.cast(
        tf.reshape(x1_safe + y0_safe * w_trg, [batch, -1]), 'int32')
    inds_bl = tf.cast(
        tf.reshape(x0_safe + y1_safe * w_trg, [batch, -1]), 'int32')
    inds_br = tf.cast(
        tf.reshape(x1_safe + y1_safe * w_trg, [batch, -1]), 'int32')

    init_trg_image = tf.reshape(
        init_trg_image, [batch, num_pixels_trg, channels])
    tgt_image = []
    for c in range(channels):
      curr_tgt = init_trg_image[:, :, c]
      curr_tgt = batch_scatter_add_tensor(curr_tgt, inds_tl, values_tl[:, :, c])
      curr_tgt = batch_scatter_add_tensor(curr_tgt, inds_tr, values_tr[:, :, c])
      curr_tgt = batch_scatter_add_tensor(curr_tgt, inds_bl, values_bl[:, :, c])
      curr_tgt = batch_scatter_add_tensor(curr_tgt, inds_br, values_br[:, :, c])
      tgt_image.append(curr_tgt)
    tgt_image = tf.stack(tgt_image, axis=2)
    return tf.reshape(tgt_image, [batch, h_trg, w_trg, channels])


def scatter_add_tensor(init, indices, updates):
  """Adds sparse updates to a variable reference.

  From https://github.com/tensorflow/tensorflow/issues/2358
  See also: https://github.com/tensorflow/tensorflow/issues/8102
  If multiple indices reference the same location, their contributions add.
  Requires updates.shape = indices.shape + ref.shape[1:].

  Args:
    init: initialization of the output tensor.
    indices: A tensor of indices into the first dimension of init. Must
      be one of the following types: int32, int64.
    updates: A tensor of updated values to add to init.
  Returns:
    A new tensor with scattering updates to init.
  """
  with tf.name_scope('scatter_add_tensor'):
    init = tf.convert_to_tensor(init)
    indices = tf.convert_to_tensor(indices)
    updates = tf.convert_to_tensor(updates)
    out_shape = tf.shape(init, out_type=indices.dtype)
    scattered_updates = tf.scatter_nd(indices, updates, out_shape)
    with tf.control_dependencies(
        [tf.assert_equal(
            out_shape, tf.shape(scattered_updates, out_type=indices.dtype))]):
      output = tf.add(init, scattered_updates)
    return output


def batch_scatter_add_tensor(init, indices, updates):
  """Performs scatter_add_tensor in batches.

  If multiple indices reference the same location, their contributions add.
  Args:
    init: initialization of the output tensor. [batch, #points]
    indices: A tensor of indices into the first dimension of init. Must
      be one of the following types: int32, int64. [batch, #updates]
    updates: A tensor of updated values to add to init. [batch, #updates]
  Returns:
    Scattered result [batch, #points].
  """
  with tf.name_scope('batch_scatter_add_tensor'):
    batch, out_size = init.get_shape().as_list()
    _, num_updates = updates.get_shape().as_list()
    init = tf.reshape(init, [-1])
    offset = np.zeros((batch, num_updates), dtype=np.int32)
    for b in range(batch):
      offset[b] = b * out_size
    offset_tf = tf.constant(
        offset.tolist(), dtype=tf.int32, shape=[batch, num_updates])
    indices += offset_tf
    indices = tf.reshape(indices, [batch * num_updates, 1])
    updates = tf.reshape(updates, [batch * num_updates])
    output = scatter_add_tensor(init, indices, updates)
    output = tf.reshape(output, [batch, out_size])
    return output
