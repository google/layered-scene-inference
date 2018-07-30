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

"""Synthetic Planar Data Generator.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from absl import logging as log
from lsi.data.syntheticPlanes import utils
from lsi.geometry import homography
from lsi.geometry import layers


def sample_views(nviews, mode='random'):
  """View sampling function.

  Args:
    nviews: Number of views to be sampled
    mode: 'random' or 'circle'
      'random': Generates views by moving the camera in the xy plane at z=0
          and looking at some random point around z=[3.0,3.5]
          and a random x,y
      'circle': circles around a fixed point.
          (useful for consistent visualizations of worlds)
  Returns:
    rot_trans_list: list of (rot, trans) tuples
  """
  rot_trans_list = []
  if mode == 'random':
    for _ in range(nviews):
      x_cam = np.random.uniform(-0.5, 0.5)
      y_cam = np.random.uniform(-0.5, 0.5)
      z_cam = 0

      x_lookat = np.random.uniform(-0.5, 0.5)
      y_lookat = np.random.uniform(-0.5, 0.5)
      z_lookat = np.random.uniform(3.0, 3.5)

      cam_pt = np.reshape(np.array([x_cam, y_cam, z_cam]), (3, 1))
      lookat_pt = np.reshape(np.array([x_lookat, y_lookat, z_lookat]), (3, 1))
      rot_mat = utils.lookat_rotation(lookat_pt - cam_pt)
      rot_trans_list.append((rot_mat, -1*np.matmul(rot_mat, cam_pt)))

  elif mode == 'circle':
    lookat_pt = np.reshape(np.array([0, 0, 2]), (3, 1))
    trans_list = utils.cam_trans_circle(
        lookat_pt, theta_start=30, theta_end=-30, nshots=nviews)
    for cam_pt in trans_list:
      rot_mat = utils.lookat_rotation(lookat_pt - cam_pt)
      rot_trans_list.append((rot_mat, -1*np.matmul(rot_mat, cam_pt)))

  return rot_trans_list


class WorldGenerator(object):
  """Planar World Generator.

  Generates random box-like worlds with random textures on side-planes
  and a few random (almost) fronto-parallel foreground objects.
  """

  def __init__(
      self, bg_tex_dir, obj_tex_dir,
      h=400, w=400, n_obj_max=4, n_obj_min=1,
      n_box_planes=5,
      ext_obj='.png', ext_bg='.jpg', split='all'):
    """Initialization function.

    Args:
      bg_tex_dir: directory for sampling images for wall/floor/ceiling textures
      obj_tex_dir: directory for sampling images for 'pop-up' object textures
      h: height of texture images
      w: width of texture images
      n_obj_max: max number of pop-up objects per random sample
      n_obj_min: min number of pop-up objects per random sample
      n_box_planes: number of planes to actually sample from the box
      ext_obj: extension of object texture images
      ext_bg: extension of bg texture images
      split: all/train/val/test. Select all or 70% or 15% or 15% of the files
    """

    self.h = h
    self.w = w
    self.n_obj_min = n_obj_min
    self.n_obj_max = n_obj_max
    self.n_box_planes = n_box_planes
    self.bs = n_box_planes + n_obj_max
    assert self.bs > 0  # should atleast be sampling box planes or billboards

    ## instantiate obj texture loader (if needed)
    if n_obj_max > 0:
      self.tex_loader_obj = utils.QueuedRandomTextureLoader(
          obj_tex_dir, ext=ext_obj, batch_size=n_obj_max, h=h, w=w, nc=4,
          split=split)

    ## instantiate bg texture loader (if needed)
    if n_box_planes > 0:
      self.tex_loader_bg = utils.QueuedRandomTextureLoader(
          bg_tex_dir, ext=ext_bg, batch_size=n_box_planes, h=h, w=w, nc=3,
          split=split)

    return

  def dummy_obj_plane(self, z_max):
    """Returns a default plane.

    This is a useful function for defining dummy planes which have all pixels
    as transparent as this function just gives some sensible defaults.
    Args:
      z_max: depth of plane
    Returns:
      obj_plane: a plane at z=z_max of a fixed size
    """
    x_dir = np.array([1, 0, 0])
    y_dir = np.array([0, 1, 0])
    # Plane params
    obj_plane = {
        'pt': np.array([0, 0, z_max]), 'x_dir': x_dir, 'y_dir': y_dir,
        'w': 1, 'h': 1, 'off_x': 0.5, 'off_y': 0.5}

    return obj_plane

  def random_obj_plane(self, extent, aspect, fixed_plane=None):
    """Returns a plane in the box according to object aspect ratio.

    Given an aspect ratio, we first slightly peturb it to obtain the aspect
    ratio of the 3D plane. Then, we compute the dimensions s.t. the plane
    stays inside the box. Then, we randomly place it on the floor in an
    almost frontal orientation.
    Args:
      extent: 3D box extent - x_min, y_min, z_min, x_max, y_max, z_max
      aspect: tentative aspect ratio (Y/X)
      fixed_plane: if None, the samples a random plane, otherwise returns
          some predefined params
    Returns:
      obj_plane: the resulting object plane.
    """
    log_aspect = np.log(aspect)
    log_aspect += np.random.uniform(-0.2, 0.2)
    aspect = np.exp(log_aspect)
    h_box = extent[4] - extent[1]
    w_box = extent[3] - extent[0]
    d_box = extent[5] - extent[2]

    # Compute obj height and width
    if aspect < h_box/w_box:  # width is bottleneck
      w_obj = np.random.uniform(0.4, 0.6)*w_box
      h_obj = w_obj*aspect
    else:  # height is bottleneck
      h_obj = np.random.uniform(0.4, 0.6)*h_box
      w_obj = h_obj/aspect

    # Random point where the objext plane is placed
    w_frac = w_obj/w_box
    if fixed_plane is not None:
      centre_x = extent[0] + 0.25*w_box + 0.25*fixed_plane*w_box
      centre_z = extent[2] + 0.2*fixed_plane*d_box
    else:
      centre_x = extent[0]
      centre_x += w_box*np.random.uniform(0.1, 0.9-w_frac) + 0.5*w_obj
      centre_z = extent[2] + 0.5*np.random.uniform(0, extent[5]-extent[2])

    pt_floor = np.array([centre_x, extent[4], centre_z])

    # Random Orientation
    x_dir = np.array([1, 0, 0])
    y_dir = np.array([0, 1, 0])

    # Plane params
    obj_plane = {
        'pt': pt_floor, 'x_dir': x_dir, 'y_dir': y_dir,
        'w': w_obj, 'h': h_obj, 'off_x': 0.5, 'off_y': 1}

    return obj_plane

  def forward(self):
    """Returns the world with a random box and some 'pop up' objects.

    Randomly sample the room box and textures associated with celing
    walls/floor. Then, randomly place a few 'pop up'
    objects on the floor.
    Returns:
      rot_w2s: Rotations from frontal planes to current frame.
      t_w2s: Translations from frontal planes to world frame.
      k_w: Intrinsics for images corresponding to plane textures
      n_hat_w: Normals for frontal-planes (z-direction vectors)
      a_w: Offset for frontal-planes (z=1)
      imgs_w: Texture images
      masks_w: Plane transparencies
    """
    bs = self.bs
    h = self.h
    w = self.w
    n_box_planes = self.n_box_planes
    imgs_w = np.ones((bs, h, w, 3))
    masks_w = np.zeros((bs, h, w, 1))

    # Front facing plane variables
    n_hat_w = np.array([[0, 0, 1]])
    n_hat_w = np.expand_dims(n_hat_w, axis=0)
    n_hat_w = np.repeat(n_hat_w, bs, axis=0)
    n_hat_w = np.reshape(n_hat_w, (bs, 1, 3))
    a_w = np.array([[-1]])
    a_w = np.expand_dims(a_w, axis=0)
    a_w = np.repeat(a_w, bs, axis=0)

    # Box layout sampling
    # xmin = -0.7 + np.random.uniform(-0.2, 0.2)
    # ymin = -0.5 + np.random.uniform(-0.2, 0.2)
    # zmin = 2.0 + np.random.uniform(-0.5, 0.5)

    # xmax = 0.7 + np.random.uniform(-0.2, 0.2)
    # ymax = 0.5 + np.random.uniform(-0.2, 0.2)
    # zmax = 3.5  # keeping a fixed distance should resolve scale ambiguities

    ## Using a fixed box, eaiser for debugging
    xmin = -0.7
    ymin = -0.5
    zmin = 2.0
    xmax = 0.7
    ymax = 0.5
    zmax = 3.5

    extent = [xmin, ymin, zmin, xmax, ymax, zmax]
    planes = utils.box_planes(extent)
    planes = planes[0:self.n_box_planes]

    # Box Texture Sampling
    log.info('Box Texture Sampling.')
    if n_box_planes > 0:
      masks_w[0:len(planes), :, :, :] = 1
      tex_imgs_bg, _ = self.tex_loader_bg.load()
      imgs_w[0:len(planes), :, :, :] = tex_imgs_bg[0:len(planes), :, :, :]

    # Foreground Objects sampling
    log.info('Foreground Objects Sampling.')
    n_obj = np.random.randint(self.n_obj_min, self.n_obj_max+1)
    if n_obj > 0:
      tex_imgs_obj, tex_shape_obj = self.tex_loader_obj.load()
      log.info('max texture value: %f', np.max(tex_imgs_obj))

    for ix in range(self.n_obj_max):
      if ix < n_obj:
        tex_img = tex_imgs_obj[ix, :, :, :]
        tex_h = tex_shape_obj[ix, 0]
        tex_w = tex_shape_obj[ix, 1]

        imgs_w[ix + n_box_planes, :, :, :] = tex_img[:, :, 0:3]
        masks_w[ix + n_box_planes, :, :, 0] = tex_img[:, :, 3]
        aspect_tex = tex_h/tex_w
        obj_plane = self.random_obj_plane(extent, aspect_tex, fixed_plane=ix)
      else:
        obj_plane = self.dummy_obj_plane(extent[5])
      planes.append(obj_plane)

    # Computing plane intrinsincs etc.
    log.info('Plane Params.')

    t_w2s = np.zeros((bs, 3, 1))
    rot_w2s = np.zeros((bs, 3, 3))
    k_w = np.zeros((bs, 3, 3))

    for ix, pl in enumerate(planes):
      centre_pt = utils.get_centre(
          pl['pt'], pl['x_dir'], pl['y_dir'], pl['w'], pl['h'],
          off_x=pl['off_x'], off_y=pl['off_y'])
      rot_w2s[ix, :, :], t_w2s[ix, :, :] = utils.canonical_transform(
          centre_pt, pl['x_dir'], pl['y_dir'])
      k_w[ix, :, :] = utils.dims2kmat(pl['w'], pl['h'], h, w)

    return rot_w2s, t_w2s, k_w, n_hat_w, a_w, imgs_w, masks_w


class Renderer(object):
  """Planar Data Renderer.
  """

  def __init__(self, n_imgs, h=400, w=400, ds_factor=1):
    """Initialization function.

    Generates the computation graphs for rendering a planar layers
    based world.
    Args:
      n_imgs: number of planes/images that the world will be composed of
      h: Height of texture images
      w: Width of texture images
      ds_factor: downsample the renderings by this factor.
          ds_factor = 2 -> output size = h/2, w/2
    """
    with tf.name_scope('synth_planes_renderer'):
      self.n_imgs = n_imgs
      # define the computation graphs below
      k_w = tf.placeholder(tf.float32, [n_imgs, 3, 3], name='k_w')
      k_s_inp = tf.placeholder(tf.float32, [3, 3], name='k_s')
      k_t_inp = tf.placeholder(tf.float32, [3, 3], name='k_t')

      # Transforms from fronto-parallel planes to source image frame
      rot_w2s = tf.placeholder(tf.float32, [n_imgs, 3, 3], name='rot_w2s')
      t_w2s = tf.placeholder(tf.float32, [n_imgs, 3, 1], name='t_w2s')

      # Plane variables in their respective canonical frames
      n_hat_w = tf.placeholder(tf.float32, [n_imgs, 1, 3], name='n_hat_w')
      a_w = tf.placeholder(tf.float32, [n_imgs, 1, 1], name='a_w')
      pixel_coords_inp = tf.placeholder(
          tf.float32, [h, w, 3], name='pixel_coords')
      imgs_w = tf.placeholder(tf.float32, [n_imgs, h, w, 3], name='imgs_w')
      masks_w = tf.placeholder(tf.float32, [n_imgs, h, w, 1], name='masks_w')

      # Transforms from source frame to target frame
      rot_s2t_inp = tf.placeholder(tf.float32, [3, 3], name='rot_s2t_inp')
      t_s2t_inp = tf.placeholder(tf.float32, [3, 1], name='t_s2t_inp')

      # Repeat some tensors along layer dimension
      rot_s2t = tf.tile(tf.expand_dims(rot_s2t_inp, 0), [n_imgs, 1, 1])
      k_s = tf.tile(tf.expand_dims(k_s_inp, 0), [n_imgs, 1, 1])
      k_t = tf.tile(tf.expand_dims(k_t_inp, 0), [n_imgs, 1, 1])
      rot_s2t = tf.tile(tf.expand_dims(rot_s2t_inp, 0), [n_imgs, 1, 1])
      t_s2t = tf.tile(tf.expand_dims(t_s2t_inp, 0), [n_imgs, 1, 1])
      pixel_coords = tf.tile(
          tf.expand_dims(pixel_coords_inp, 0), [n_imgs, 1, 1, 1])

      with tf.name_scope('w2s2t_rendering'):
        # we'll define a compute graph for source_layers_rendering
        # intermediate nodes for layer_images_src, layer_masks_src, n_hat_s, a_s
        # followed by layer transformation code to render target
        imgs_w2s = homography.transform_plane_imgs(
            imgs_w, pixel_coords, k_w, k_s, rot_w2s, t_w2s, n_hat_w, a_w)

        masks_w2s = homography.transform_plane_imgs(
            masks_w, pixel_coords, k_w, k_s, rot_w2s, t_w2s, n_hat_w, a_w)

        n_hat_s, a_s = homography.transform_plane_eqns(
            rot_w2s, t_w2s, n_hat_w, a_w)

        imgs_s2t, masks_s2t, dmaps_s2t = layers.planar_transform(
            imgs_w2s, masks_w2s, pixel_coords_inp,
            k_s_inp, k_t_inp, rot_s2t_inp, t_s2t_inp, n_hat_s, a_s)

        source_layers_rendering = layers.compose(
            imgs_s2t, masks_s2t, dmaps_s2t, soft=False,
            min_disp=2e-1, depth_softmax_temp=0.4)
        if ds_factor != 1:
          source_layers_rendering = tf.image.resize_images(
              source_layers_rendering, [h//ds_factor, w//ds_factor],
              method=tf.image.ResizeMethod.AREA)

      with tf.name_scope('w2t_rendering'):
        # we'll define a compute graph for planar_rendering
        # intermediate nodes for rot_w2t, t_w2t followed by rendering
        rot_w2t = tf.matmul(rot_s2t, rot_w2s)
        t_w2t = t_s2t + tf.matmul(rot_s2t, t_w2s)
        imgs_w2t = homography.transform_plane_imgs(
            imgs_w, pixel_coords, k_w, k_t, rot_w2t, t_w2t, n_hat_w, a_w)
        masks_w2t = homography.transform_plane_imgs(
            masks_w, pixel_coords, k_w, k_t, rot_w2t, t_w2t, n_hat_w, a_w)
        # imgs_w2t = tf.Print(
        #     imgs_w2t, [tf.reduce_max(imgs_w2t), tf.reduce_max(masks_w2t)],
        #     message='max transformed texture value: ')
        dmats_w2t = homography.trg_disp_maps(
            pixel_coords, k_t, rot_w2t, t_w2t, n_hat_w, a_w)
        n_hat_t, a_t = homography.transform_plane_eqns(
            rot_w2t, t_w2t, n_hat_w, a_w)

        # imgs_w2t = tf.Print(
        #     imgs_w2t, [tf.reduce_mean(imgs_w2t)], message='imgs_w2t_mean: '
        # )
        # masks_w2t = tf.Print(
        #     masks_w2t, [tf.reduce_mean(masks_w2t)], message='masks_w2t_mean: '
        # )
        # dmats_w2t = tf.Print(
        #     dmats_w2t, [tf.reduce_mean(dmats_w2t)], message='dmats_w2t_mean: '
        # )

        planar_rendering = layers.compose(
            imgs_w2t, masks_w2t, dmats_w2t, soft=False,
            min_disp=2e-1, depth_softmax_temp=0.4)
        # planar_rendering = tf.Print(
        #     planar_rendering, [tf.reduce_max(planar_rendering)],
        #     message='max composed texture value: ')

        rendering_disp_fg = layers.compose_depth(
            masks_w2t, dmats_w2t, bg_layer=False,
            min_disp=2e-1, depth_softmax_temp=0.4)

        rendering_disp_bg = layers.compose_depth(
            masks_w2t, dmats_w2t, bg_layer=True,
            min_disp=2e-1, depth_softmax_temp=0.4)

        if ds_factor != 1:
          planar_rendering = tf.image.resize_images(
              planar_rendering, [h//ds_factor, w//ds_factor],
              method=tf.image.ResizeMethod.AREA)
          rendering_disp_bg = tf.image.resize_images(
              rendering_disp_bg, [h//ds_factor, w//ds_factor],
              method=tf.image.ResizeMethod.AREA)
          rendering_disp_fg = tf.image.resize_images(
              rendering_disp_fg, [h//ds_factor, w//ds_factor],
              method=tf.image.ResizeMethod.AREA)

    # start tf session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=config)

    # tf placeholders
    self.tf_phs = {
        'k_w': k_w, 'k_s': k_s_inp, 'k_t': k_t_inp,
        'rot_w2s': rot_w2s, 't_w2s': t_w2s,
        'n_hat_w': n_hat_w, 'a_w': a_w,
        'imgs_w': imgs_w, 'masks_w': masks_w,
        'pixel_coords': pixel_coords_inp,
        'rot_s2t': rot_s2t_inp, 't_s2t': t_s2t_inp
    }

    self.tf_intermediates = {
        'imgs_s2t': imgs_s2t, 'masks_s2t': masks_s2t
    }

    self.tf_res = {
        'source_layers_rendering': source_layers_rendering,
        'planar_rendering': planar_rendering,
        'rendering_disp_fg': rendering_disp_fg,
        'rendering_disp_bg': rendering_disp_bg,
        'n_hat_t': n_hat_t,
        'a_t': a_t
    }

    self._feed_dict = {}
    return

  def set_feed_dict(self, **kwargs):
    """Set the world variables for rendering.

    Args:
      **kwargs: values for various place_holders.
        'k_w': n_imgs X 3 X 3 intrinsics for plane texture images
        'k_s': 3 X 3 intrinsic for source frame camera
        'k_t': 3 X 3 intrinsic for rendering frame camera
        'rot_w2s': n_imgs X 3 X 3 rotations for planes
        't_w2s': n_imgs X 3 X 1 translations for planes
        'n_hat_w': n_imgs X 1 X 3 orienations for frontal planes
            (typically just z-axis oriented)
        'a_w': n_imgs X 1 X 1 offsets for frontal planes
            (typically just z=1)
        'imgs_w': n_imgs X H X W X 3 texture images for planes
        'masks_w': n_imgs X H X W X 1 transpareny values for planes
        'pixel_coords': H X W X 3 meshgrid for pixel coords
    """
    phs = self.tf_phs
    for key, val in kwargs.iteritems():
      self._feed_dict[phs[key]] = val

  def render_layers(self, rot, t):
    """Render target frame via layered representation in source frame.

    This function first renders the layered representation of the
    world in the source frame and then computes the target frame image
    by transforming this layered representation.
    Args:
      rot: 3 X 3 rotation martix from source to target
      t: 3 X 1 translation martix from source to target
    Returns:
      rendered H X W X 3 image
    """
    self.set_feed_dict(rot_s2t=rot, t_s2t=t)
    return self._sess.run(
        self.tf_res['source_layers_rendering'], feed_dict=self._feed_dict)

  def render_planes(self, rot, t):
    """Render target frame directly by rendering planes.

    Args:
      rot: 3 X 3 rotation martix from source to target
      t: 3 X 1 translation martix from source to target
    Returns:
      rendered H X W X 3 image
    """
    self.set_feed_dict(rot_s2t=rot, t_s2t=t)
    return self._sess.run(
        self.tf_res['planar_rendering'], feed_dict=self._feed_dict)

  def render_disps(self, rot, t):
    """Render fg and bg layer disps directly by rendering planes.

    Args:
      rot: 3 X 3 rotation martix from source to target
      t: 3 X 1 translation martix from source to target
    Returns:
      rendered [H X W X 1, H X W X 1] fg and bg depth image
    """
    self.set_feed_dict(rot_s2t=rot, t_s2t=t)
    return self._sess.run(
        [self.tf_res['rendering_disp_fg'], self.tf_res['rendering_disp_bg']],
        feed_dict=self._feed_dict)

  def plane_geometry(self, rot, t):
    """Plane equation coordinates w.r.t final rendered image.

    Args:
      rot: 3 X 3 rotation martix from source to target
      t: 3 X 1 translation martix from source to target
    Returns:
      n_hat_t: L X 1 X 3 normals
      a_t: L X 1 X 1 displacements
    """
    self.set_feed_dict(rot_s2t=rot, t_s2t=t)
    return self._sess.run(
        [self.tf_res['n_hat_t'], self.tf_res['a_t']],
        feed_dict=self._feed_dict)


class DataLoader(object):
  """Planar Data Loader (combines Generator and Renderer).
  """

  def __init__(self, opts):
    """Initialization function.

    Args:
      opts: dictionary containing params required for
          WorldGenerator and Renderer
    """
    self.opts = opts
    self.output_gt = opts.synth_dl_eval_data
    ds_factor = opts.synth_ds_factor
    img_width = opts.img_width*ds_factor
    img_height = opts.img_width*ds_factor
    self.generator = WorldGenerator(
        opts.sun_imgs_dir, opts.pascal_objects_dir,
        h=img_height, w=img_width,
        n_obj_max=opts.n_obj_max, n_obj_min=opts.n_obj_min,
        n_box_planes=opts.n_box_planes, split=opts.data_split)

    self.renderer = Renderer(
        opts.n_box_planes+opts.n_obj_max,
        h=opts.img_height*ds_factor, w=opts.img_width*ds_factor,
        ds_factor=ds_factor
    )

    f_x = img_width
    f_y = img_height
    u_x = f_x/2.0
    u_y = f_y/2.0
    pixel_coords = np.meshgrid(
        np.linspace(0.5, img_width-0.5, img_width),
        np.linspace(0.5, img_height-0.5, img_height),
        np.linspace(1, 1, 1)
    )
    self.pixel_coords = np.reshape(
        np.transpose(pixel_coords, (3, 1, 2, 0)),
        (img_height, img_width, 3)
    )
    self.k_s = np.array([
        [f_x, 0, u_x],
        [0, f_y, u_y],
        [0, 0, 1]
    ])
    self.k_t = np.copy(self.k_s)
    self.renderer.set_feed_dict(
        k_s=self.k_s, k_t=self.k_t,
        pixel_coords=self.pixel_coords
    )

  def forward_instance(self):
    """Single pair loader.

    Returns:
      img_s: Source frame image
      img_t: Target frame image
      k_s: Source frame intrinsic
      k_t: Target frame intrinsic
      rot: relative rotation from source to target
      trans: relative translation from source to target
      n_hat: gt plane normals
      a: gt plane displacements
    """
    ds_factor = self.opts.synth_ds_factor
    generator = self.generator
    renderer = self.renderer
    log.info('Generating Layered World.')
    rot_w2s, t_w2s, k_w, n_hat_w, a_w, imgs_w, masks_w = generator.forward()

    masks_w_bg = np.copy(masks_w)
    masks_w_bg[self.opts.n_box_planes:, :, :, :] = 0

    log.info('Rendering Layered World.')
    renderer.set_feed_dict(
        k_w=k_w, rot_w2s=rot_w2s, t_w2s=t_w2s,
        n_hat_w=n_hat_w, a_w=a_w,
        imgs_w=imgs_w, masks_w=masks_w
    )

    # (rot_src, trans_src) = sample_views(1, 'random')[0]
    rot_src = np.eye(3)
    trans_src = np.zeros((3, 1))

    (rot_trg, trans_trg) = sample_views(1, 'random')[0]
    # rot_trg = np.eye(3)
    # trans_trg = np.zeros((3, 1))

    img_s = renderer.render_planes(rot_src, trans_src)
    img_t = renderer.render_planes(rot_trg, trans_trg)

    if self.output_gt:
      disp_s_fg, _ = renderer.render_disps(rot_src, trans_src)
      disp_t_fg, _ = renderer.render_disps(rot_trg, trans_trg)
      # print(rot_trg, trans_trg)
      n_hat, a = renderer.plane_geometry(rot_src, trans_src)

      masks_w_bg = np.copy(masks_w)
      masks_w_bg[self.opts.n_box_planes:, :, :, :] = 0
      renderer.set_feed_dict(
          k_w=k_w, rot_w2s=rot_w2s, t_w2s=t_w2s,
          n_hat_w=n_hat_w, a_w=a_w,
          imgs_w=imgs_w, masks_w=masks_w_bg
      )
      img_s_bg = renderer.render_planes(rot_src, trans_src)
      disp_s_bg, _ = renderer.render_disps(rot_src, trans_src)
      # print(rot_src, trans_src)
      img_t_bg = renderer.render_planes(rot_trg, trans_trg)
      disp_t_bg, _ = renderer.render_disps(rot_trg, trans_trg)

    log.info('Rendering Finished.')

    # p_s = R_s*p_w + t_s
    # p_t = R_t*p_w + t_t
    # So, p_t = R*p_s + t, R = R_t*R_s^(-1), t = t_t - R*t_s
    rot = np.matmul(rot_trg, np.transpose(rot_src, (1, 0)))
    trans = trans_trg - np.matmul(rot, trans_src)

    if self.output_gt:
      return (
          np.copy(img_s), np.copy(img_t),
          utils.resize_instrinsic(self.k_s, 1/ds_factor, 1/ds_factor),
          utils.resize_instrinsic(self.k_t, 1/ds_factor, 1/ds_factor),
          rot, trans,
          n_hat, a,
          # None, None, None, None
          np.copy(disp_s_fg), np.copy(disp_s_bg),
          np.copy(disp_t_fg), np.copy(disp_t_bg),
          np.copy(img_s_bg), np.copy(img_t_bg)
      )
    else:
      return (
          np.copy(img_s), np.copy(img_t),
          utils.resize_instrinsic(self.k_s, 1/ds_factor, 1/ds_factor),
          utils.resize_instrinsic(self.k_t, 1/ds_factor, 1/ds_factor),
          rot, trans
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
      n_hat: gt plane normals
      a: gt plane displacements
    """
    instances_list = [list(self.forward_instance()) for _ in range(bs)]
    nvars = len(instances_list[0])
    concat_instances = [np.stack(
        [instances_list[b][ix] for b in range(bs)]
    ) for ix in range(nvars)]

    return concat_instances
