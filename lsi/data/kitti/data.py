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

"""Kitti Data Loader.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import fnmatch
import os

import numpy as np
import tensorflow as tf


def resize_instrinsic(intrinsic, scale_x, scale_y):
  intrinsic_rsz = np.copy(intrinsic)
  intrinsic_rsz[0, :] *= scale_x
  intrinsic_rsz[1, :] *= scale_y
  return intrinsic_rsz


def raw_city_sequences():
  """Sequence names for city sequences in kitti raw data.

  Returns:
    seq_names: list of names
  """
  seq_names = [
      '2011_09_26_drive_0001',
      '2011_09_26_drive_0002',
      '2011_09_26_drive_0005',
      '2011_09_26_drive_0009',
      '2011_09_26_drive_0011',
      '2011_09_26_drive_0013',
      '2011_09_26_drive_0014',
      '2011_09_26_drive_0017',
      '2011_09_26_drive_0018',
      '2011_09_26_drive_0048',
      '2011_09_26_drive_0051',
      '2011_09_26_drive_0056',
      '2011_09_26_drive_0057',
      '2011_09_26_drive_0059',
      '2011_09_26_drive_0060',
      '2011_09_26_drive_0084',
      '2011_09_26_drive_0091',
      '2011_09_26_drive_0093',
      '2011_09_26_drive_0095',
      '2011_09_26_drive_0096',
      '2011_09_26_drive_0104',
      '2011_09_26_drive_0106',
      '2011_09_26_drive_0113',
      '2011_09_26_drive_0117',
      '2011_09_28_drive_0001',
      '2011_09_28_drive_0002',
      '2011_09_29_drive_0026',
      '2011_09_29_drive_0071',
  ]
  return seq_names


class DataLoader(object):
  """Kitti Data Loading class.
  """

  def __init__(self, opts):
    """Initialization function.

    Args:
      opts: dictionary containing params required.
    """
    self.opts = opts
    self.batch_size = opts.batch_size
    # 'mview' or 'odom' or 'raw_city' variant
    self.dataset_variant = opts.kitti_dataset_variant
    self.output_disparities = (self.dataset_variant == 'raw_city') and hasattr(
        opts, 'kitti_dl_disparities'
    ) and opts.kitti_dl_disparities and (not opts.data_split == 'train')
    self.root_dir = opts.kitti_data_root
    if self.dataset_variant == 'odom':
      # Dataset corresponds to the odometry extension here
      # http://www.cvlibs.net/datasets/kitti/eval_odometry.php
      self.root_dir = os.path.join(
          self.root_dir, 'odometry', 'dataset', 'sequences')
    elif self.dataset_variant == 'mview':
      # Dataset corresponds to the multi-view extension here
      # http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow
      # Calibration from http://www.cvlibs.net/datasets/kitti/setup.php
      self.root_dir = os.path.join(self.root_dir, 'stereo_multiview_2015')
      if opts.data_split == 'train':
        self.root_dir += '/training'
      else:
        self.root_dir += '/testing'
    elif self.dataset_variant == 'raw_city':
      # Dataset corresponds to the multi-view extension here
      # http://www.cvlibs.net/datasets/kitti/raw_data.php
      self.root_dir = os.path.join(self.root_dir, 'raw')

    self.h = opts.img_height
    self.w = opts.img_width
    self.init_img_names_seq_list()

  def init_img_names_seq_list(self):
    """Initialize list of image names and corresponding sequence ids.
    """
    # image list
    opts = self.opts
    self.img_list_src = []
    self.img_list_trg = []
    self.seq_id_list = []
    if self.dataset_variant == 'mview':
      for root, _, filenames in os.walk(
          os.path.join(self.root_dir, 'image_2')):
        for filename in fnmatch.filter(filenames, '*.png'):
          self.img_list_src.append(os.path.join(root, filename))
      self.img_list_src.sort()
      for img_name in self.img_list_src:
        seq_id = int(img_name.split('/')[-1].split('_')[0])
        self.seq_id_list.append(seq_id)

    elif self.dataset_variant == 'odom':
      if opts.data_split == 'train':
        data_seq = range(0, 7) + range(12, 21)
      elif opts.data_split == 'val':
        data_seq = range(7, 9)
      elif opts.data_split == 'test':
        data_seq = range(9, 11)
      for seq_id in data_seq:
        seq_dir = os.path.join(self.root_dir, '{:02d}'.format(seq_id))
        for root, _, filenames in os.walk(
            os.path.join(seq_dir, 'image_2')):
          for filename in fnmatch.filter(filenames, '*.png'):
            self.img_list_src.append(os.path.join(root, filename))
            self.seq_id_list.append(seq_id)

    elif self.dataset_variant == 'raw_city':
      exclude_img = '2011_09_26_drive_0117_sync/image_02/data/0000000074.png'
      seq_names = raw_city_sequences()
      rng = np.random.RandomState(0)
      rng.shuffle(seq_names)
      n_all = len(seq_names)
      n_train = int(round(0.7*n_all))
      n_val = int(round(0.15*n_all))
      if opts.data_split == 'train':
        seq_names = seq_names[0:n_train]
      elif opts.data_split == 'val':
        seq_names = seq_names[n_train:(n_train+n_val)]
      elif opts.data_split == 'test':
        seq_names = seq_names[(n_train+n_val):n_all]
      for seq_id in seq_names:
        seq_date = seq_id[0:10]
        seq_dir = os.path.join(
            self.root_dir, seq_date, '{}_sync'.format(seq_id))
        for root, _, filenames in os.walk(
            os.path.join(seq_dir, 'image_02')):
          for filename in fnmatch.filter(filenames, '*.png'):
            src_img_name = os.path.join(root, filename)
            if exclude_img not in src_img_name:
              self.img_list_src.append(os.path.join(src_img_name))
              self.seq_id_list.append(seq_date)

    if self.dataset_variant == 'raw_city':
      self.img_list_trg = [f.replace(
          'image_02', 'image_03') for f in self.img_list_src]

      if self.output_disparities:
        self.img_list_disp_src = []
        for im_name in self.img_list_src:
          im_name_split = im_name.split('/')
          disp_name_src = os.path.join(
              self.root_dir, 'spss_stereo_results', opts.data_split,
              im_name_split[-4],
              im_name_split[-1][:-4] + '_left_initial_disparity.png')
          self.img_list_disp_src.append(disp_name_src)
        self.img_list_disp_trg = [f.replace(
            'left', 'right') for f in self.img_list_disp_src]

    else:
      self.img_list_trg = [f.replace(
          'image_2', 'image_3') for f in self.img_list_src]

  def preload_calib_files(self):
    """Preload calibration files for the sequence."""
    self.cam_calibration = {}
    if self.dataset_variant == 'mview':
      for root, _, filenames in os.walk(
          os.path.join(self.root_dir, 'calib_cam_to_cam')):
        for filename in fnmatch.filter(filenames, '*.txt'):
          calib_file = os.path.join(root, filename)
          seq_id = int(filename.split('.txt')[0])
          self.cam_calibration[seq_id] = self.read_calib_file(calib_file)

    elif self.dataset_variant == 'odom':
      for seq_id in range(22):
        calib_file = os.path.join(
            self.root_dir, '{:02d}'.format(seq_id), 'calib.txt')
        cam_calibration = self.read_calib_file(calib_file)
        for key in ['P_rect_00', 'P_rect_01', 'P_rect_02', 'P_rect_03']:
          cam_calibration[key] = np.copy(
              cam_calibration[key.replace('_rect_0', '')])
        self.cam_calibration[seq_id] = cam_calibration

    elif self.dataset_variant == 'raw_city':
      seq_names = raw_city_sequences()
      for seq_id in seq_names:
        seq_date = seq_id[0:10]
        calib_file = os.path.join(
            self.root_dir, seq_date, 'calib_cam_to_cam.txt')
        self.cam_calibration[seq_date] = self.read_calib_file(calib_file)

  def read_calib_file(self, file_path):
    """Read camera intrinsics."""
    # taken from https://github.com/hunse/kitti
    float_chars = set('0123456789.e+- ')
    data = {}
    with open(file_path, 'r') as f:
      for line in f:
        key, value = line.split(':', 1)
        value = value.strip()
        data[key] = value
        if float_chars.issuperset(value):
          # try to cast to float array
          try:
            data[key] = np.array(map(float, value.split(' ')))
          except ValueError:
            # casting error: data[key] already eq. value, so pass
            pass

    return data

  def img_queue_loader(self, img_list, nc=3):
    with tf.name_scope('queued_data_loader'):
      filename_queue = tf.train.string_input_producer(
          img_list, seed=0, shuffle=True)
      image_reader = tf.WholeFileReader()
      _, image_file = image_reader.read(filename_queue)
      # image_file = tf.Print(image_file, [image_file_key])
      image = tf.image.decode_image(image_file)
      image = tf.cast(tf.image.decode_image(image_file), 'float32')
      image *= 1.0/255  # since images are loaded in [0, 255]
      image = tf.slice(image, [0, 0, 0], [-1, -1, nc])

      orig_shape = tf.shape(image)
      orig_shape.set_shape((3))

      image = tf.image.resize_images(
          image, [self.h, self.w], method=tf.image.ResizeMethod.AREA)
      image.set_shape((self.h, self.w, nc))

      return image, orig_shape

  def define_queues(self):
    """Defines preloading queues.
    """
    self.src_im, self.src_orig_shape = self.img_queue_loader(self.img_list_src)
    self.trg_im, self.trg_orig_shape = self.img_queue_loader(self.img_list_trg)
    if self.output_disparities:
      self.src_disp, _ = self.img_queue_loader(self.img_list_disp_src, nc=1)
      self.trg_disp, _ = self.img_queue_loader(self.img_list_disp_trg, nc=1)

    # All queued vars
    sample_queue = tf.train.range_input_producer(
        len(self.img_list_src), seed=0, shuffle=True)
    sample_index = sample_queue.dequeue()

    queue_tensors = [
        self.src_im, self.src_orig_shape,
        self.trg_im, self.trg_orig_shape, sample_index]
    if self.output_disparities:
      queue_tensors.append(self.src_disp)
      queue_tensors.append(self.trg_disp)

    self.queue_output = tf.train.batch(
        queue_tensors,
        batch_size=self.batch_size)

    # Coordinate the loading of image files.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    self.tf_sess = tf.Session(config=config)

    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(
        coord=self.coord, sess=self.tf_sess)

  def forward_instance(
      self, img_src, img_trg, src_shape, trg_shape, calib_data):
    """Single pair loader.

    Args:
      img_src: source image
      img_trg: trg image
      src_shape: source image original shape
      trg_shape: target image original shape
      calib_data: calibratio data for current sequence
    Returns:
      img_s: Source frame image
      img_t: Target frame image
      k_s: Source frame intrinsic
      k_t: Target frame intrinsic
      rot: relative rotation from source to target
      trans: relative translation from source to target
    """
    rot = np.eye(3)
    trans = np.zeros(3)
    k_s = np.copy(calib_data['P_rect_02'].reshape(3, 4)[:3, :3])

    k_t = np.copy(calib_data['P_rect_03'].reshape(3, 4)[:3, :3])

    trans_src = np.copy(calib_data['P_rect_02'].reshape(3, 4)[:, 3])
    trans_trg = np.copy(calib_data['P_rect_03'].reshape(3, 4)[:, 3])

    # The translation is in homogeneous 2D coords, convert to regular 3d space:
    trans_src[0] = (trans_src[0] - k_s[0, 2]*trans_src[2])/k_s[0, 0]
    trans_src[1] = (trans_src[1] - k_s[1, 2]*trans_src[2])/k_s[1, 1]

    trans_trg[0] = (trans_trg[0] - k_t[0, 2]*trans_trg[2])/k_t[0, 0]
    trans_trg[1] = (trans_trg[1] - k_t[1, 2]*trans_trg[2])/k_t[1, 1]

    trans = trans_trg - trans_src

    k_s = resize_instrinsic(k_s, self.w/src_shape[1], self.h/src_shape[0])
    k_t = resize_instrinsic(k_t, self.w/trg_shape[1], self.h/trg_shape[0])

    return (
        img_src, img_trg,
        k_s, k_t,
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
    batch = self.tf_sess.run(self.queue_output)
    if self.output_disparities:
      img_src, src_shape, img_trg, trg_shape, sample_ids, disp_src, disp_trg = batch
    else:
      img_src, src_shape, img_trg, trg_shape, sample_ids = batch

    instances_list = [list(self.forward_instance(
        img_src[b, :, :, :],
        img_trg[b, :, :, :],
        src_shape[b],
        trg_shape[b],
        self.cam_calibration[self.seq_id_list[sample_ids[b]]]
    )) for b in range(bs)]

    nvars = len(instances_list[0])
    concat_instances = [np.stack(
        [instances_list[b][ix] for b in range(bs)]
    ) for ix in range(nvars)]
    if self.output_disparities:
      concat_instances.append(disp_src)
      concat_instances.append(disp_trg)

    return concat_instances
