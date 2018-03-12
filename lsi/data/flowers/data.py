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

"""Flowers Data Loader.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import fnmatch
import os

import numpy as np
import tensorflow as tf
from pyglib import log


def process_lf(lf, lfsize):
  """Process lightfield image : reshapes and takes central u,v images.

  Args:
    lf: H X W X C image
    lfsize: [h, w, n_u, n_v]
  Returns:
    lf: h X w X n_u X n_v X C image
  """
  lf = tf.image.adjust_gamma(
      tf.to_float(lf[:lfsize[0]*8, :lfsize[1]*8, :]), gamma=0.4)
  lf = tf.transpose(
      tf.reshape(lf, [lfsize[0], 8, lfsize[1], 8, 3]), [0, 2, 1, 3, 4])
  # u_min = (14//2)-(lfsize[2]//2)
  # u_max = (14//2)+(lfsize[2]//2)
  # v_min = (14//2)-(lfsize[3]//2)
  # v_max = (14//2)+(lfsize[3]//2)
  # lf = lf[:, :, u_min:u_max, v_min:v_max, :]
  return lf


def sample_lf(lf, patch_size):
  """Sample source and trg patch.

  Args:
    lf: h X w X n_u X n_v X C image
    patch_size: [img_h, img_w]
  Returns:
    src_im: img_h X img_w X C image
    trg_im: img_h X img_w X C image
    del_u: camera x difference
    del_v: camera y difference
  """
  im_h = patch_size[0]
  im_w = patch_size[1]
  n_u = tf.shape(lf)[2]
  n_v = tf.shape(lf)[3]

  r = tf.random_uniform(
      shape=[], minval=0, maxval=tf.shape(lf)[0]-im_h, dtype=tf.int32)
  c = tf.random_uniform(
      shape=[], minval=0, maxval=tf.shape(lf)[1]-im_w, dtype=tf.int32)
  # u_src = tf.random_uniform(shape=[], minval=0, maxval=n_u, dtype=tf.int32)
  u_src = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
  v_src = tf.random_uniform(shape=[], minval=0, maxval=n_v, dtype=tf.int32)

  # u_trg = tf.random_uniform(shape=[], minval=0, maxval=n_u, dtype=tf.int32)
  u_trg = tf.random_uniform(shape=[], minval=n_u-2, maxval=n_u, dtype=tf.int32)
  v_trg = tf.random_uniform(shape=[], minval=0, maxval=n_v, dtype=tf.int32)

  src_im = lf[r:r+im_h, c:c+im_w, v_src, u_src, :]
  trg_im = lf[r:r+im_h, c:c+im_w, v_trg, u_trg, :]
  return src_im, trg_im, u_trg-u_src, v_trg-v_src


class DataLoader(object):
  """Flowers Data Loading class.
  """

  def __init__(self, opts):
    """Initialization function.

    Args:
      opts: dictionary containing params required.
    """
    self.opts = opts
    self.batch_size = opts.batch_size
    self.imgs_root_dir = opts.flowers_data_root
    self.lf_size = [375, 540, 8, 8]  # dimensions of Lytro light fields
    self.delta_camera = 0.01  # distance between consecutive cameras
    self.focal = 400  # focal length
    self.h = opts.img_height
    self.w = opts.img_width

    # image list
    self.img_list = []
    for root, _, filenames in os.walk(self.imgs_root_dir):
      for filename in fnmatch.filter(filenames, '*.png'):
        self.img_list.append(os.path.join(root, filename))

    split = opts.data_split
    self.img_list.sort()
    rng = np.random.RandomState(0)
    rng.shuffle(self.img_list)
    n_all = len(self.img_list)
    n_train = int(round(0.7*n_all))
    n_val = int(round(0.15*n_all))
    if split == 'train':
      self.img_list = self.img_list[0:n_train]
    elif split == 'val':
      self.img_list = self.img_list[n_train:(n_train+n_val)]
    elif split == 'test':
      self.img_list = self.img_list[(n_train+n_val):n_all]
    log.info(len(self.img_list))

  def define_queues(self):
    """Defines preloading queues.
    """
    with tf.name_scope('queued_data_loader'):
      self.filename_queue = tf.train.string_input_producer(
          self.img_list, seed=0, shuffle=True)
      image_reader = tf.WholeFileReader()
      _, image_file = image_reader.read(self.filename_queue)
      # image_file = tf.Print(image_file, [image_file_key])
      image = tf.image.decode_image(image_file)
      image = tf.cast(tf.image.decode_image(image_file), 'float32')
      image *= 1.0/255  # since images are loaded in [0, 255]
      image = tf.slice(image, [0, 0, 0], [-1, -1, 3])
      self.lf = process_lf(image, self.lf_size)
      self.src_im, self.trg_im, self.del_u, self.del_v = sample_lf(
          self.lf, (self.h, self.w))
      self.src_im.set_shape((self.h, self.w, 3))
      self.trg_im.set_shape((self.h, self.w, 3))
      # del_u.set_shape((1))
      # del_v.set_shape((1))

    # All queued vars
    self.queue_output = [
        self.src_im, self.trg_im,
        self.del_u, self.del_v]

    # Coordinate the loading of image files.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    self.tf_sess = tf.Session(config=config)

    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(
        coord=self.coord, sess=self.tf_sess)

  def forward_instance(
      self, lf):
    """Single pair loader.

    Args:
      lf: Read lightfield
    Returns:
      img_s: Source frame image
      img_t: Target frame image
      k_s: Source frame intrinsic
      k_t: Target frame intrinsic
      rot: relative rotation from source to target
      trans: relative translation from source to target
    """
    img_src, img_trg, del_u, del_v = self.tf_sess.run(
        self.queue_output, feed_dict={self.lf: lf})
    rot = np.eye(3)
    trans = np.zeros(3)
    trans[0] = -del_u*self.delta_camera
    trans[1] = -del_v*self.delta_camera
    k_mat = np.array([
        self.focal, 0, self.w/2,
        0, self.focal, self.h/2,
        0, 0, 1]).reshape(3, 3)
    return (
        img_src, img_trg,
        np.copy(k_mat), np.copy(k_mat),
        rot, trans.reshape(3, 1)
    )

  def forward(self, bs):
    """Computes bs data instances.

    Args:
      bs: batch_size
    Returns:
      img_s: Source frame images
      img_t: Target frame images
      k_s: Source frame intrinsics
      k_t: Target frame intrinsics
      rot: relative rotations from source to target
      trans: relative translations from source to target
    """
    lf = self.tf_sess.run(self.lf)
    instances_list = [list(self.forward_instance(lf)) for b in range(bs)]
    nvars = len(instances_list[0])
    concat_instances = [np.stack(
        [instances_list[b][ix] for b in range(bs)]
    ) for ix in range(nvars)]

    return concat_instances
