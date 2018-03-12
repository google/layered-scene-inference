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

"""TensorFlow utils for image transformations via homographies.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lsi.geometry import sampling
from lsi.nnutils import helpers as nn_helpers


def inv_homography(k_s, k_t, rot, t, n_hat, a):
  """Computes inverse homography matrix.

  Args:
      k_s: intrinsics for source cameras, are [...] X 3 X 3 matrices
      k_t: intrinsics for target cameras, are [...] X 3 X 3 matrices
      rot: relative rotation, are [...] X 3 X 3 matrices
      t: [...] X 3 X 1, translations from source to target camera
      n_hat: [...] X 1 X 3, plane normal w.r.t source camera frame
      a: [...] X 1 X 1, plane equation displacement
  Returns:
      homography: [...] X 3 X 3 inverse homography matrices
  """
  with tf.name_scope('inv_homography'):
    rot_t = nn_helpers.transpose(rot)
    k_t_inv = tf.matrix_inverse(k_t, name='k_t_inv')

    denom = a - tf.matmul(tf.matmul(n_hat, rot_t), t)
    numerator = tf.matmul(tf.matmul(tf.matmul(rot_t, t), n_hat), rot_t)
    inv_hom = tf.matmul(
        # tf.matmul(k_s, rot_t + tf.divide(numerator, denom)),
        tf.matmul(k_s, rot_t + nn_helpers.divide_safe(numerator, denom)),
        k_t_inv, name='inv_hom')
    return inv_hom


def inv_homography_dmat(k_t, rot, t, n_hat, a):
  """Computes M where M*(u,v,1) = d_t.

  Args:
      k_t: intrinsics for target cameras, are [...] X 3 X 3 matrices
      rot: relative rotation, are [...] X 3 X 3 matrices
      t: [...] X 3 X 1, translations from source to target camera
      n_hat: [...] X 1 X 3, plane normal w.r.t source camera frame
      a: [...] X 1 X 1, plane equation displacement
  Returns:
      d_mat: [...] X 1 X 3 matrices
  """
  with tf.name_scope('inv_homography'):
    rot_t = nn_helpers.transpose(rot)
    k_t_inv = tf.matrix_inverse(k_t, name='k_t_inv')

    denom = a - tf.matmul(tf.matmul(n_hat, rot_t), t)
    # d_mat = tf.divide(
    d_mat = nn_helpers.divide_safe(
        -1 * tf.matmul(tf.matmul(n_hat, rot_t), k_t_inv), denom, name='dmat')
    return d_mat


def normalize_homogeneous(pts_coords):
  """Converts homogeneous coordinates to regular coordinates.

  Args:
      pts_coords : [...] X n_dims_coords+1; Homogeneous coordinates.
  Returns:
      pts_coords_uv_norm : [...] X n_dims_coords;
          normal coordinates after dividing by the last entry.
  """
  with tf.name_scope('normalize_homogeneous'):
    pts_size = pts_coords.get_shape().as_list()
    n_dims = len(pts_size)
    n_dims_coords = pts_size[-1] - 1

    pts_coords_uv, pts_coords_norm = tf.split(
        pts_coords, [n_dims_coords, 1], axis=n_dims - 1)
    # pts_coords_norm = tf.Print(
    #     pts_coords_norm,
    #     [tf.reduce_min(tf.abs(pts_coords_norm))],
    #     message='pts_coords_norm min: '
    # )
    # return tf.divide(pts_coords_uv, pts_coords_norm)
    return nn_helpers.divide_safe(pts_coords_uv, pts_coords_norm)


def transform_plane_imgs(imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a):
  """Transforms input imgs via homographies for corresponding planes.

  Args:
    imgs: are [...] X H_s X W_s X C
    pixel_coords_trg: [...] X H_t X W_t X 3; pixel (u,v,1) coordinates.
    k_s: intrinsics for source cameras, are [...] X 3 X 3 matrices
    k_t: intrinsics for target cameras, are [...] X 3 X 3 matrices
    rot: relative rotation, are [...] X 3 X 3 matrices
    t: [...] X 3 X 1, translations from source to target camera
    n_hat: [...] X 1 X 3, plane normal w.r.t source camera frame
    a: [...] X 1 X 1, plane equation displacement
  Returns:
    [...] X H_t X W_t X C images after bilinear sampling from input.
      Coordinates outside the image are sampled as 0.
  """
  with tf.name_scope('transform_plane_imgs'):
    # k_s = tf.Print(
    #     k_s, [tf.reduce_mean(k_s)], message='k_s mean: '
    # )
    # k_t = tf.Print(
    #     k_t, [tf.reduce_mean(k_t)], message='k_t mean: '
    # )
    # rot = tf.Print(
    #     rot, [tf.reduce_mean(rot)], message='rot mean: '
    # )
    # t = tf.Print(
    #     t, [tf.reduce_mean(t)], message='t mean: '
    # )

    hom_t2s_planes = inv_homography(k_s, k_t, rot, t, n_hat, a)

    # hom_t2s_planes = tf.Print(
    #     hom_t2s_planes, [tf.reduce_mean(hom_t2s_planes)],
    #     message='hom_t2s_planes mean: '
    # )

    pixel_coords_t2s = nn_helpers.transform_pts(
        pixel_coords_trg, hom_t2s_planes)
    # pixel_coords_t2s = tf.Print(
    #     pixel_coords_t2s, [tf.reduce_mean(pixel_coords_t2s)],
    #     message='pixel_coords_t2s mean: '
    # )

    pixel_coords_t2s = normalize_homogeneous(pixel_coords_t2s)

    # pixel_coords_t2s = tf.Print(
    #     pixel_coords_t2s, [tf.reduce_mean(pixel_coords_t2s)],
    #     message='pixel_coords_t2s+norm mean: '
    # )

    imgs_s2t = sampling.bilinear_wrapper(imgs, pixel_coords_t2s)
    return imgs_s2t


def transform_plane_eqns(rot, t, n_hat, a):
  """Transforms plane euqations according to frame transformation.

  Args:
    rot: relative rotation, are [...] X 3 X 3 matrices
    t: [...] X 3 X 1, translations from source to target camera
    n_hat: [...] X 1 X 3, plane normal w.r.t source camera frame
    a: [...] X 1 X 1, plane equation displacement
  Returns:
    n_hat_t: [...] X 1 X 3, plane normal w.r.t target camera frame
    a_t: [...] X 1 X 1, plane plane equation displacement
  """
  with tf.name_scope('transform_plane_eqns'):
    rot_t = nn_helpers.transpose(rot)
    n_hat_t = tf.matmul(n_hat, rot_t)
    a_t = a - tf.matmul(n_hat, tf.matmul(rot_t, t))
    return n_hat_t, a_t


def shift_plane_eqns(plane_shift, pred_planes):
  """For every p on the original plane, p+plane_shift lies on the new plane.

  Args:
    plane_shift: relative rotation, are B X 1 X 3 points
    pred_planes: [n_hat, a]
        n_hat is [...] X 1 X 3, plane normal w.r.t source camera frame
        a is [...] X 1 X 1, plane equation displacement
  Returns:
    shifted_planes: [n_hat_t, a_t] where
        n_hat_t: [...] X 1 X 3, plane normal after shifting
        a_t: [...] X 1 X 1, plane plane equation displacement after plane shift
  """
  n_hat = pred_planes[0]
  a = pred_planes[1]
  n_hat_t = n_hat
  # plane_shift = tf.Print(plane_shift, [plane_shift])
  a_t = a - tf.reduce_sum(plane_shift*n_hat, axis=-1, keepdims=True)
  # a_t = tf.Print(a_t, [a_t])
  return [n_hat_t, a_t]


def trg_disp_maps(pixel_coords_trg, k_t, rot, t, n_hat, a):
  """Computes pixelwise inverse depth for target pixels via plane equations.

  Args:
    pixel_coords_trg: [...] X H_t X W_t X 3; pixel (u,v,1) coordinates.
    k_t: intrinsics for target cameras, are [...] X 3 X 3 matrices
    rot: relative rotation, are [...] X 3 X 3 matrices
    t: [...] X 3 X 1, translations from source to target camera
    n_hat: [...] X 1 X 3, plane normal w.r.t source camera frame
    a: [...] X 1 X 1, plane equation displacement
  Returns:
    [...] X H_t X W_t X 1 images corresponding to inverse depth at each pixel
  """
  with tf.name_scope('trg_disp_maps'):
    dmats_t = inv_homography_dmat(k_t, rot, t, n_hat, a)  # size: [...] X 1 X 3
    disp_t = tf.reduce_sum(
        tf.expand_dims(dmats_t, -2)*pixel_coords_trg, axis=-1, keepdims=True)
    return disp_t
