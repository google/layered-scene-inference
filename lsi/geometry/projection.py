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
"""Utils for perspective projection.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lsi.geometry import sampling
from lsi.nnutils import helpers as nn_helpers
import tensorflow as tf


def pad_intrinsic(k_mat):
  """Convert a 3X3 intrinsic matrix to 4X4.

  Args:
    k_mat: relative rotation, are [...] X 3 X 3
  Returns:
    k_mat_padded: [...] X 4 X 4
  """
  with tf.name_scope('pad_intrinsic'):
    k_shape = k_mat.get_shape().as_list()
    zeros01 = tf.zeros(k_shape[:-1] + [1])
    zeros10 = tf.zeros(k_shape[:-2] + [1] + [k_shape[-1]])
    ones11 = tf.ones(k_shape[:-2] + [1, 1])
    k_mat_padded = tf.concat(
        [
            tf.concat([k_mat, zeros01], axis=-1),
            tf.concat([zeros10, ones11], axis=-1)
        ],
        axis=-2)
    return k_mat_padded


def pad_extrinsic(rot_mat, trans_mat):
  """Concatenates rotation and translation, and pads to get a 4X4 matrix.

  Args:
    rot_mat: rotation, are [...] X 3 X 3
    trans_mat: translation, are [...] X 3 X 3
  Returns:
    ext_mat: [...] X 4 X 4
  """
  with tf.name_scope('pad_extrinsic'):
    r_shape = rot_mat.get_shape().as_list()
    zeros10 = tf.zeros(r_shape[:-2] + [1] + [r_shape[-1]])
    ones11 = tf.ones(r_shape[:-2] + [1, 1])
    ext_mat = tf.concat(
        [
            tf.concat([rot_mat, trans_mat], axis=-1),
            tf.concat([zeros10, ones11], axis=-1)
        ],
        axis=-2)
    return ext_mat


def forward_projection_matrix(k_s, k_t, rot, t):
  """Projection matrix for transforming a src pixel coordinates to target frame.

  Args:
      k_s: intrinsics for source cameras, are [...] X 3 X 3 matrices
      k_t: intrinsics for target cameras, are [...] X 3 X 3 matrices
      rot: relative rotation from source to target, are [...] X 3 X 3 matrices
      t: [...] X 3 X 1 translations from source to target camera
  Returns:
      transform: [...] X 4 X 4 projection matrix
  """
  with tf.name_scope('forward_projection_matrix'):
    k_s_inv = tf.matrix_inverse(k_s, name='k_s_inv')
    return tf.matmul(
        pad_intrinsic(k_t),
        tf.matmul(pad_extrinsic(rot, t), pad_intrinsic(k_s_inv)))


def inverse_projection_matrix(k_s, k_t, rot, t):
  """Projection matrix for transforming a trg pixel coordinates to src frame.

  Args:
      k_s: intrinsics for source cameras, are [...] X 3 X 3 matrices
      k_t: intrinsics for target cameras, are [...] X 3 X 3 matrices
      rot: relative rotation from source to target, are [...] X 3 X 3 matrices
      t: [...] X 3 X 1 translations from source to target camera
  Returns:
      transform: [...] X 4 X 4 projection matrix
  """
  with tf.name_scope('inverse_projection_matrix'):
    k_t_inv = tf.matrix_inverse(k_t, name='k_t_inv')
    rot_inv = nn_helpers.transpose(rot)
    t_inv = -1 * tf.matmul(rot_inv, t)
    return tf.matmul(
        pad_intrinsic(k_s),
        tf.matmul(pad_extrinsic(rot_inv, t_inv), pad_intrinsic(k_t_inv)))


def disocclusion_mask(disps_src,
                      disps_trg,
                      pixel_coords_src,
                      src2trg_mat,
                      thresh=1e-2):
  """Projection matrix for transforming a trg pixel coordinates to src frame.

  Args:
      disps_src: are B X H_t X W_t X 1, inverse depths in source frame
      disps_trg: are B X H_t X W_t X 1, inverse depths in target frame
      pixel_coords_src: B X H_t X W_t X 3
        pixel (u,v,1) coordinates of source image pixels.
      src2trg_mat: B X 4 X 4 matrix for transforming source coordinate to target
      thresh: difference threshold
  Returns:
      disp_diff: B X H_t X W_t X 1
        difference in sampled and projection based disparity for src image
  """
  with tf.name_scope('disocclusion_mask'):
    ndims = len(pixel_coords_src.get_shape())
    _, h_t, w_t, _ = disps_trg.get_shape().as_list()
    coords_src = tf.concat([pixel_coords_src, disps_src], axis=-1)
    coords_trg = nn_helpers.transform_pts(coords_src, src2trg_mat)

    uv_coords_trg, normalizer, disps_src2trg = tf.split(
        coords_trg, [2, 1, 1], axis=ndims - 1)
    uv_coords_trg = nn_helpers.divide_safe(uv_coords_trg, normalizer)
    disps_src2trg = nn_helpers.divide_safe(disps_src2trg, normalizer)

    u_coords_trg, v_coords_trg = tf.split(uv_coords_trg, [1, 1], axis=ndims - 1)
    truncation_mask = tf.cast(tf.greater(u_coords_trg, w_t), tf.float32)
    truncation_mask += tf.cast(tf.greater(v_coords_trg, h_t), tf.float32)
    truncation_mask += tf.cast(tf.less(u_coords_trg, 0), tf.float32)
    truncation_mask += tf.cast(tf.less(v_coords_trg, 0), tf.float32)
    truncation_mask = tf.cast(tf.greater(truncation_mask, 0), tf.float32)

    disps_src2trg_sampled = sampling.bilinear_wrapper(
        disps_trg, uv_coords_trg, compose=True)
    disocc_mask = tf.greater(
        tf.abs(disps_src2trg - disps_src2trg_sampled), thresh)

    return (1 - truncation_mask) * tf.cast(disocc_mask, tf.float32)
