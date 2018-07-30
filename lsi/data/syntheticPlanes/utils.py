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

"""Utils for Synthetic Planar Data Generator.
"""

import fnmatch
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import logging as log


def resize_instrinsic(intrinsic, scale_x, scale_y):
  intrinsic_rsz = np.copy(intrinsic)
  intrinsic_rsz[0, :] *= scale_x
  intrinsic_rsz[1, :] *= scale_y
  return intrinsic_rsz


def dims2kmat(w_plane, h_plane, w_tex, h_tex):
  """Computes intrinsic matrix from plane and texture image sizes.

  Args:
    w_plane: real world size of plane in x-dimension
    h_plane: real world size of plane in y-dimension
    w_tex: width of texture map
    h_tex: Height of texture map
  Returns:
    kmat : 3 X 3 instrinsic matrix.
    dz : distance along z
  """
  dz = 1.0  # assume fronto-parallel plane at Z=1
  kmat = np.array([
      [w_tex*dz/w_plane, 0, w_tex/2],
      [0, h_tex*dz/h_plane, h_tex/2],
      [0, 0, 1]
  ])
  return kmat


def get_centre(pt, x_dir, y_dir, w, h, off_x=0.5, off_y=0.5):
  """Compute plane centre given a point with plane coordinates.

  Args:
    pt: known point on plane
    x_dir: x_direction of plane
    y_dir: y_direction of plane
    w: plane width
    h: plane height
    off_x: the point pt is offx*w from top left
    off_y: the point pt is offy*h from top left
  Returns:
    coordinate of centre: top_left + 0.5h*y_dir + 0.5w*x_dir
  """
  x_dir = np.reshape(x_dir, (3, 1))
  x_dir = np.divide(x_dir, np.linalg.norm(x_dir))

  y_dir = np.reshape(y_dir, (3, 1))
  y_dir = np.divide(y_dir, np.linalg.norm(y_dir))

  pt = np.reshape(pt, (3, 1))
  return pt + w*x_dir*(0.5-off_x) + h*y_dir*(0.5-off_y)


def canonical_transform(centre_s, x_dir, y_dir, trans_init=None):
  """Transform s.t canonical plane has x_dir, y_dir, centre_s.

  Args:
    centre_s: desired plane centre after transformation
    x_dir: desired x_direction after transformation
    y_dir: desired y_direction after transformation
    trans_init: initial centre of canonical plane
  Returns:
    rot : 3 X 3 instrinsic matrix.
    trans : 3 X 1 translation matrix.
  """
  x_dir = np.reshape(x_dir, (3, 1))
  x_dir = np.divide(x_dir, np.linalg.norm(x_dir))

  y_dir = np.reshape(y_dir, (3, 1))
  y_dir = np.divide(y_dir, np.linalg.norm(y_dir))

  centre_s = np.reshape(centre_s, (3, 1))

  if trans_init is None:
    trans_init = np.array([0, 0, 1])
  trans_init = np.reshape(trans_init, (3, 1))

  z_dir = np.cross(x_dir, y_dir, axisa=0, axisb=0, axisc=0)

  rot = np.concatenate([x_dir, y_dir, z_dir], axis=1)
  trans = centre_s - np.matmul(rot, trans_init)
  return rot, trans


def box_planes(extent):
  """Parameters for defining planes corresponding to a box.

  Args:
    extent: (x0, y0, z0, x1, y1, z1) for box coords
  Returns:
    list of plane instantiation parameters
  """

  x0, y0, z0, x1, y1, z1 = extent
  planes = []

  x_dir = np.array([1, 0, 0])
  y_dir = np.array([0, 1, 0])
  z_dir = np.array([0, 0, 1])
  front = {
      'pt': np.array([x0, y0, z1]), 'x_dir': x_dir, 'y_dir': y_dir,
      'w': x1-x0, 'h': y1-y0, 'off_x': 0, 'off_y': 0}
  ceil = {
      'pt': np.array([x0, y0, z1]), 'x_dir': x_dir, 'y_dir': -1*z_dir,
      'w': x1-x0, 'h': z1-z0, 'off_x': 0, 'off_y': 0}
  floor = {
      'pt': np.array([x0, y1, z1]), 'x_dir': x_dir, 'y_dir': -1*z_dir,
      'w': x1-x0, 'h': z1-z0, 'off_x': 0, 'off_y': 0}
  wall_l = {
      'pt': np.array([x0, y0, z0]), 'x_dir': z_dir, 'y_dir': y_dir,
      'w': z1-z0, 'h': y1-y0, 'off_x': 0, 'off_y': 0}
  wall_r = {
      'pt': np.array([x1, y0, z0]), 'x_dir': z_dir, 'y_dir': y_dir,
      'w': z1-z0, 'h': y1-y0, 'off_x': 0, 'off_y': 0}

  planes.append(front)
  planes.append(floor)
  planes.append(ceil)
  planes.append(wall_l)
  planes.append(wall_r)

  return planes


def cam_trans_dolly(trans, nshots=10):
  """Camera translations during a dolly shot.

  Args:
    trans: translation from (0, 0, 0) at end of shot
    nshots: number of frames
  Returns:
    list of camera translations
  """
  alphas = np.linspace(0, 1, nshots)
  return [trans*alphas[ix] for ix in range(nshots)]


def _rot_y(theta):
  return np.array([
      [math.cos(theta), 0, math.sin(theta)],
      [0, 1, 0],
      [-math.sin(theta), 0, math.cos(theta)]
  ])


def _rot_x(theta):
  return np.array([
      [1, 0, 0],
      [0, math.cos(theta), -math.sin(theta)],
      [0, math.sin(theta), math.cos(theta)]
  ])


def cam_trans_circle(anchor, theta_start=0, theta_end=30, nshots=10):
  """Camera translations during a z-axis circling shot.

  Args:
    anchor: point to rotate around
    theta_start: initial angle around anchor
    theta_end: final angle around anchor
    nshots: number of frames
  Returns:
    list of camera translations
  """
  thetas = np.linspace(theta_start, theta_end, nshots)
  anchor = np.copy(np.reshape(anchor, (3, 1)))
  rots = [_rot_y(theta*math.pi/180) for theta in thetas]
  return [anchor - np.matmul(r, anchor) for r in rots]


def lookat_rotation(delta):
  """Camera rotation such that a point at delta gets z-aligned.

  Args:
    delta: point to look at
  Returns:
    rotation matrix to transform points s.t. R*delta = (0, 0, z)
  """
  delta = np.reshape(delta, 3)
  theta = np.arctan2(delta[0], delta[2])
  radius = np.linalg.norm(delta)
  phi = np.arcsin(delta[1]/radius)
  return np.matmul(_rot_x(phi), _rot_y(-1*theta))


class QueuedRandomTextureLoader(object):
  """Loads a random image from the base_dir (or its subdirectories).
  """

  def __init__(self, base_dir, ext='.jpg',
               batch_size=1, h=None, w=None, nc=3, split='all'):
    """Initialization function.

    Args:
      base_dir: directory where image files are present
      ext: file extension for images
      batch_size: number of images to be loaded together
      h: image height.
      w: image width.
      nc: number of channels. Default = 3
      split: all/train/val/test. Select all or 70% or 15% or 15% of the files
    """
    self.batch_size = batch_size
    if not base_dir.endswith('/'):
      base_dir += '/'
    self.img_list = []
    self.base_dir = base_dir
    for root, _, filenames in os.walk(base_dir):
      for filename in fnmatch.filter(filenames, '*' + ext):
        self.img_list.append(os.path.join(root, filename))

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
    print self.img_list[0]

    # Tf graph
    log.info('Image directory : %s.', base_dir)
    log.info('Number of Images : %d.', len(self.img_list))
    log.info('Data Split : %s.', split)
    print 'Image directory : %s.' % base_dir
    print 'Number of Images : %d.' % len(self.img_list)
    with tf.name_scope('queued_data_loader'):
      filename_queue = tf.train.string_input_producer(
          self.img_list, seed=0, shuffle=True)
      image_reader = tf.WholeFileReader()
      _, image_file = image_reader.read(filename_queue)
      # image_file = tf.Print(image_file, [image_file_key])
      image = tf.cast(tf.image.decode_image(image_file, channels=nc), 'float32')
      image *= 1.0/255  # since images are loaded in [0, 255]
      image = tf.slice(image, [0, 0, 0], [-1, -1, nc])

      orig_shape = tf.shape(image)
      orig_shape.set_shape((3))

      image = tf.image.resize_images(
          image, [h, w], method=tf.image.ResizeMethod.AREA)
      image.set_shape((h, w, nc))

      self.image_and_shape = tf.train.batch(
          [image, orig_shape], batch_size=batch_size)

      # Coordinate the loading of image files.

      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True

      self.tf_sess = tf.Session(config=config)

      self.coord = tf.train.Coordinator()
      self.threads = tf.train.start_queue_runners(
          coord=self.coord, sess=self.tf_sess)

    return

  def load(self):
    """Load a random image.

    Returns:
      A random image as a numpy array.
    """
    image_and_shape = self.tf_sess.run(self.image_and_shape)
    return image_and_shape[0], image_and_shape[1]


class RandomTextureLoader(object):
  """Loads a random image from the base_dir (or its subdirectories).
  """

  def __init__(self, base_dir, ext='.jpg'):
    """Initialization function.

    Args:
      base_dir: directory where image files are present
      ext: file extension for images
    """
    if not base_dir.endswith('/'):
      base_dir += '/'
    self.normalize_scale = 1.0
    if ext == '.jpg':
      self.normalize_scale = 1.0/255
    self.img_list = []
    self.base_dir = base_dir
    for root, _, filenames in os.walk(base_dir):
      for filename in fnmatch.filter(filenames, '*' + ext):
        self.img_list.append(os.path.join(root, filename)[len(base_dir):])
    return

  def load(self, substr=None):
    """Load a random image.

    Args:
      substr: pattern to filter image names
    Returns:
      A random image as a numpy array.
    """
    img_list = self.img_list
    if substr is not None:
      img_list = [im for im in img_list if substr in im]
    img_ind = np.random.randint(len(img_list))
    img = plt.imread(os.path.join(self.base_dir, img_list[img_ind]))
    if len(np.shape(img)) == 2:
      img = np.expand_dims(img, axis=2)
    if np.shape(img)[2] == 1:
      img = np.multiply(img, np.ones((1, 1, 3)))
    return img*self.normalize_scale
