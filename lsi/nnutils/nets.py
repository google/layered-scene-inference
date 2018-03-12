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

"""CNN definition helpers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import utils
# from pyglib import log
from lsi.nnutils import helpers as nn_helpers


def encoder_simple(inp_img, nz=1000, is_training=True, reuse=False):
  """Creates a simple encoder CNN.

  Args:
    inp_img: TensorFlow node for input with size B X H X W X C
    nz: number of units in last layer, default=1000
    is_training: whether batch_norm should be in train mode
    reuse: Whether to reuse weights from an already defined net
  Returns:
    An encoder CNN which computes a final representation with nz
    units.
  """
  batch_norm_params = {'is_training': is_training}
  with tf.variable_scope('encoder', reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params,
        weights_regularizer=slim.l2_regularizer(0.05),
        activation_fn=tf.nn.relu,
        outputs_collections=end_points_collection):
      cnv1 = slim.conv2d(inp_img, 32, [7, 7], stride=2, scope='cnv1')
      cnv1b = slim.conv2d(cnv1, 32, [7, 7], stride=1, scope='cnv1b')
      cnv2 = slim.conv2d(cnv1b, 64, [5, 5], stride=2, scope='cnv2')
      cnv2b = slim.conv2d(cnv2, 64, [5, 5], stride=1, scope='cnv2b')
      cnv3 = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
      cnv3b = slim.conv2d(cnv3, 128, [3, 3], stride=1, scope='cnv3b')
      cnv4 = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
      cnv4b = slim.conv2d(cnv4, 256, [3, 3], stride=1, scope='cnv4b')
      cnv5 = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
      cnv5b = slim.conv2d(cnv5, 512, [3, 3], stride=1, scope='cnv5b')
      cnv6 = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
      cnv6b = slim.conv2d(cnv6, 512, [3, 3], stride=1, scope='cnv6b')
      cnv7 = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
      cnv7b = slim.conv2d(cnv7, 512, [3, 3], stride=1, scope='cnv7b')
      cnv7b_flat = slim.flatten(cnv7b, scope='cnv7b_flat')
      enc = slim.stack(
          cnv7b_flat, slim.fully_connected, [2*nz, nz, nz], scope='fc')

    end_points = utils.convert_collection_to_dict(end_points_collection)
    return enc, end_points


def plane_predictor(feat, n_layers=1, reuse=False):
  """Predicts plane equations : normals and displacements.

  Args:
    feat: B X nz feature vectors
    n_layers: number of plane equations to predict (denoted as L)
    reuse: Whether to reuse weights from an already defined net
  Returns:
    normals : L X B X 1 X 3, displacements : L X B X 1 X 1.
  """
  with tf.variable_scope('plane_pred', reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope(
        [slim.fully_connected],
        normalizer_fn=None,
        weights_regularizer=slim.l2_regularizer(0.05),
        activation_fn=None,
        outputs_collections=end_points_collection):
      normals = []
      displacements = []
      for l in range(n_layers):
        norm_pred = slim.fully_connected(feat, 3, scope='normal_' + str(l))
        # norm_pred = tf.Print(norm_pred, [norm_pred], message='Pre-unit Norm')

        norm_pred = slim.unit_norm(norm_pred, 1, scope='unit_normal_' + str(l))
        # norm_pred = tf.Print(norm_pred, [norm_pred], message='Post-unit Norm')

        norm_pred = tf.reshape(norm_pred, (-1, 1, 3))

        disp_pred = slim.fully_connected(feat, 1, scope='disp_' + str(l))
        disp_pred = tf.reshape(disp_pred, (-1, 1, 1))

        normals.append(norm_pred)
        displacements.append(disp_pred)

      end_points = utils.convert_collection_to_dict(end_points_collection)
      normals = tf.stack(normals, axis=0)
      displacements = tf.stack(displacements, axis=0)

      return [normals, displacements], end_points


def depthsweep_planes(bs, n_layers=1, min_disp=0.1, max_disp=1):
  """Outputs fixed plane equations.

  Args:
    bs: batchsize
    n_layers: number of plane equations to output (denoted as L)
    min_disp: inverse depth of last plane
    max_disp: inverse depth of first plane
  Returns:
    normals : L X B X 1 X 3, displacements : L X B X 1 X 1.
  """
  with tf.variable_scope('depthsweep_planes'):
    eps = 1e-10
    if n_layers > 1:
      delta = (min_disp - max_disp)/(n_layers-1)
      plane_disparities = tf.range(max_disp, min_disp+delta, delta-eps)
    else:
      plane_disparities = max_disp

    plane_disparities = tf.reshape(plane_disparities, [n_layers, 1, 1, 1])
    plane_displacements = tf.divide(1, plane_disparities)
    plane_displacements *= tf.ones([1, bs, 1, 1])
    x_normal = tf.zeros([n_layers, bs, 1, 1])
    y_normal = tf.zeros([n_layers, bs, 1, 1])
    z_normal = -1*tf.ones([n_layers, bs, 1, 1])
    normals = tf.concat([x_normal, y_normal, z_normal], 3)
    return [normals, plane_displacements]


def decoder_simple(
    feat, nconv=7, is_training=True, skip_feat=None, reuse=False):
  """Creates a simple encoder CNN.

  Args:
    feat: Input geatures with size B X nz or B X H X W X nz
    nconv: number of deconv layers
    is_training: whether batch_norm should be in train mode
    skip_feat: additional skip-features per upconv layer
    reuse: Whether to reuse weights from an already defined net
  Returns:
    A decoder CNN which adds nconv upsampling layers
    units.
  """
  batch_norm_params = {'is_training': is_training}
  n_filters = [32, 64, 128, 256]
  if nconv > 4:
    for _ in range(nconv-4):
      n_filters.append(512)

  with tf.variable_scope('decoder', reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose],
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params,
        weights_regularizer=slim.l2_regularizer(0.05),
        activation_fn=tf.nn.relu,
        outputs_collections=end_points_collection):
      if feat.get_shape().ndims == 2:
        feat = tf.expand_dims(tf.expand_dims(feat, 1), 1)
      for nc in range(nconv, 0, -1):
        n_filt = n_filters[nc-1]
        feat = slim.conv2d_transpose(
            feat, n_filt, [4, 4], stride=2, scope='upcnv'+str(nc))
        if (nc > 1) and (skip_feat is not None):
          feat = tf.concat([feat, skip_feat[-nc+1]], axis=3)
        feat = slim.conv2d(
            feat, n_filt, [3, 3], stride=1, scope='upcnv'+str(nc)+'b')

    end_points = utils.convert_collection_to_dict(end_points_collection)
    return feat, end_points


def pixelwise_predictor_recurrent(
    feat, nc=3, n_layers=1, reuse=False, lstm_size=32):
  """Predicts layerwise outputs, one layer a a time via an RNN.

  Args:
    feat: B X H X W X C feature vectors
    nc: number of output channels
    n_layers: number of plane equations to predict (denoted as L)
    reuse: Whether to reuse weights from an already defined net
    lstm_size: number of units in the LSTM cell
  Returns:
    preds : L X B X H X W X nc.
  """
  feat_shape = feat.get_shape().as_list()
  nc_inp = feat_shape[-1]
  with tf.variable_scope('pixelwise_pred_rec', reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'

    feat = tf.reshape(feat, [-1, nc_inp])
    bs_lstm = feat.get_shape().as_list()[0]
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(lstm_size)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(2)])

    lstm_outs = []
    state = stacked_lstm.zero_state(bs_lstm, tf.float32)
    zero_feat = tf.zeros(feat.get_shape())
    for l in range(n_layers):
      if l > 1:
        output, state = stacked_lstm(zero_feat, state)
      else:
        output, state = stacked_lstm(feat, state)
      lstm_outs.append(output)

    lstm_outs = tf.reshape(tf.stack(lstm_outs, axis=0), [-1, lstm_size])
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        normalizer_fn=None,
        weights_regularizer=slim.l2_regularizer(0.05),
        activation_fn=tf.nn.sigmoid,
        outputs_collections=end_points_collection):
      preds = tf.reshape(
          slim.fully_connected(lstm_outs, nc),
          [n_layers] + feat_shape[:-1] + [nc])

    end_points = utils.convert_collection_to_dict(end_points_collection)

    return preds, end_points


def pixelwise_predictor(
    feat, nc=3, n_layers=1, n_layerwise_steps=0,
    skip_feat=None, reuse=False, is_training=True):
  """Predicts texture images and probilistic masks.

  Args:
    feat: B X H X W X C feature vectors
    nc: number of output channels
    n_layers: number of plane equations to predict (denoted as L)
    n_layerwise_steps: Number of independent per-layer up-conv steps
    skip_feat: List of features useful for skip connections. Used if lws>0.
    reuse: Whether to reuse weights from an already defined net
    is_training: whether batch_norm should be in train mode
  Returns:
    textures : L X B X H X W X nc.
  """
  with tf.variable_scope('pixelwise_pred', reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope(
        [slim.conv2d],
        normalizer_fn=None,
        weights_regularizer=slim.l2_regularizer(0.05),
        activation_fn=tf.nn.sigmoid,
        outputs_collections=end_points_collection):
      preds = []
      for l in range(n_layers):
        with tf.variable_scope('upsample_' +  str(l), reuse=reuse):
          feat_l, _ = decoder_simple(
              feat, nconv=n_layerwise_steps, skip_feat=skip_feat,
              reuse=reuse, is_training=is_training)
          pred = slim.conv2d(
              feat_l, nc, [3, 3], stride=1, scope='pred_' + str(l))
          preds.append(pred)

      end_points = utils.convert_collection_to_dict(end_points_collection)
      preds = tf.stack(preds, axis=0)

      return preds, end_points


def ldi_predictor(
    feat, n_layers=1, reuse=False,
    n_layerwise_steps=0, skip_feat=None, pred_masks=False, is_training=True):
  """Predicts ldi : [textures, masks, disps].

  Args:
    feat: B X H X W X C feature vectors
    n_layers: number of layers to predict (denoted as L)
    reuse: Whether to reuse weights from an already defined net
    n_layerwise_steps: Number of independent per-layer up-conv steps
    skip_feat: List of features useful for skip connections. Used if lws>0.
    pred_masks: Whether to predict masks or use all 1s
    is_training: whether batch_norm should be in train mode
  Returns:
    ldi : [textures, masks, disps]
        textures : L X B X H X W X nc.
        masks : L X B X H X W X 1 (all ones)
        textures : L X B X H X W X 1
  """
  with tf.variable_scope('ldi_tex_disp', reuse=reuse):
    nc = 3+1
    if pred_masks:
      nc += 1
    tex_disp_pred, _ = pixelwise_predictor(
        feat, nc=nc, n_layers=n_layers,
        n_layerwise_steps=n_layerwise_steps, skip_feat=skip_feat,
        reuse=reuse, is_training=is_training)
    if pred_masks:
      tex_pred, masks_ldi, disps_pred = tf.split(
          tex_disp_pred, [3, 1, 1], axis=4)
      masks_ldi = nn_helpers.enforce_bg_occupied(tf.nn.sigmoid(masks_ldi))
    else:
      tex_pred, disps_pred = tf.split(tex_disp_pred, [3, 1], axis=4)
      masks_ldi = tf.ones(disps_pred.get_shape())

    ldi = [tex_pred, masks_ldi, disps_pred]
    return ldi


def resize_like(inputs, ref):
  h, w = inputs.get_shape()[1], inputs.get_shape()[2]
  rh, rw = ref.get_shape()[1], ref.get_shape()[2]
  if h == rh and w == rw:
    return inputs
  return tf.image.resize_nearest_neighbor(inputs, [rh.value, rw.value])


def encoder_decoder_simple(
    inp_img, nz=1000, nupconv=8, is_training=True,
    reuse=False, nl_diff_enc_dec=0):
  """Creates a simple encoder-decoder CNN.

  Args:
    inp_img: TensorFlow node for input with size B X H X W X C
    nz: number of units in last layer, default=1000
    nupconv: number of upconv layers in the deocder
    is_training: whether batch_norm should be in train mode
    reuse: Whether to reuse weights from an already defined net
    nl_diff_enc_dec: Number of dec layers are nupconv - nl_diff_enc_dec
  Returns:
    feat: A bottleneck representation with nz units.
    feat_dec: features of the same size as the image.
    skip_feat: initial layer features useful for layerwise steps
    end_points: intermediate activations
  """
  feat, enc_intermediate = encoder_simple(
      inp_img, is_training=is_training, nz=nz, reuse=reuse)
  feat_dec, dec_intermediate = decoder_simple(
      feat, nconv=nupconv-nl_diff_enc_dec, is_training=is_training, reuse=reuse)
  enc_dec_int = dict(enc_intermediate, **dec_intermediate)
  skip_feat = None
  return feat, feat_dec, skip_feat, enc_dec_int


def encoder_decoder_unet(
    inp_img, nz=1000, is_training=True, reuse=False, nl_diff_enc_dec=0):
  """Creates a Unet-like CNN with + features extracted from bottleneck.

  Args:
    inp_img: TensorFlow node for input with size B X H X W X C
    nz: number of units in last layer, default=1000
    is_training: whether batch_norm should be in train mode
    reuse: Whether to reuse weights from an already defined net
    nl_diff_enc_dec: Number of dec layers are num_enc_layers - nl_diff_enc_dec
  Returns:
    feat: A bottleneck representation with nz units.
    icnv1: features of the same size as the image / 2^(nl_diff_enc_dec).
    skip_feat: initial layer features useful for layerwise steps
    end_points: intermediate activations
  """
  batch_norm_params = {'is_training': is_training}
  with tf.variable_scope('encoder_decoder_unet', reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope(
        [slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params,
        weights_regularizer=slim.l2_regularizer(0.05),
        activation_fn=tf.nn.relu,
        outputs_collections=end_points_collection):
      cnv1 = slim.conv2d(inp_img, 32, [7, 7], stride=2, scope='cnv1')
      cnv1b = slim.conv2d(cnv1, 32, [7, 7], stride=1, scope='cnv1b')
      cnv2 = slim.conv2d(cnv1b, 64, [5, 5], stride=2, scope='cnv2')
      cnv2b = slim.conv2d(cnv2, 64, [5, 5], stride=1, scope='cnv2b')
      cnv3 = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
      cnv3b = slim.conv2d(cnv3, 128, [3, 3], stride=1, scope='cnv3b')
      cnv4 = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
      cnv4b = slim.conv2d(cnv4, 256, [3, 3], stride=1, scope='cnv4b')
      cnv5 = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
      cnv5b = slim.conv2d(cnv5, 512, [3, 3], stride=1, scope='cnv5b')
      cnv6 = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
      cnv6b = slim.conv2d(cnv6, 512, [3, 3], stride=1, scope='cnv6b')
      cnv7 = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
      cnv7b = slim.conv2d(cnv7, 512, [3, 3], stride=1, scope='cnv7b')

      ## features via fc layers on bottleneck
      cnv7b_flat = slim.flatten(cnv7b, scope='cnv7b_flat')
      feat = slim.stack(
          cnv7b_flat, slim.fully_connected, [2*nz, nz, nz], scope='fc')

      feats_dec = []  # decoded features at different layers
      skip_feat = []  # initial layer features useful for layerwise steps

      upcnv7 = slim.conv2d_transpose(
          cnv7b, 512, [4, 4], stride=2, scope='upcnv7')
      # There might be dimension mismatch due to uneven down/up-sampling
      # upcnv7 = resize_like(upcnv7, cnv6b)
      i7_in = tf.concat([upcnv7, cnv6b], axis=3)
      icnv7 = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')
      feats_dec.append(icnv7)
      skip_feat.append(cnv6b)

      upcnv6 = slim.conv2d_transpose(
          icnv7, 512, [4, 4], stride=2, scope='upcnv6')
      # upcnv6 = resize_like(upcnv6, cnv5b)
      i6_in = tf.concat([upcnv6, cnv5b], axis=3)
      icnv6 = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')
      feats_dec.append(icnv6)
      skip_feat.append(cnv5b)

      upcnv5 = slim.conv2d_transpose(
          icnv6, 256, [4, 4], stride=2, scope='upcnv5')
      # upcnv5 = resize_like(upcnv5, cnv4b)
      i5_in = tf.concat([upcnv5, cnv4b], axis=3)
      icnv5 = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')
      feats_dec.append(icnv5)
      skip_feat.append(cnv4b)

      upcnv4 = slim.conv2d_transpose(
          icnv5, 128, [4, 4], stride=2, scope='upcnv4')
      i4_in = tf.concat([upcnv4, cnv3b], axis=3)
      icnv4 = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
      feats_dec.append(icnv4)
      skip_feat.append(cnv3b)

      upcnv3 = slim.conv2d_transpose(
          icnv4, 64, [4, 4], stride=2, scope='upcnv3')
      i3_in = tf.concat([upcnv3, cnv2b], axis=3)
      icnv3 = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
      feats_dec.append(icnv3)
      skip_feat.append(cnv2b)

      upcnv2 = slim.conv2d_transpose(
          icnv3, 32, [4, 4], stride=2, scope='upcnv2')
      i2_in = tf.concat([upcnv2, cnv1b], axis=3)
      icnv2 = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
      feats_dec.append(icnv2)
      skip_feat.append(cnv1b)

      upcnv1 = slim.conv2d_transpose(
          icnv2, 32, [4, 4], stride=2, scope='upcnv1')
      icnv1 = slim.conv2d(upcnv1, 32, [3, 3], stride=1, scope='icnv1')
      feats_dec.append(icnv1)

      end_points = utils.convert_collection_to_dict(end_points_collection)
      return feat, feats_dec[-1-nl_diff_enc_dec], skip_feat, end_points


def warp_predictor(
    inp_img0, inp_img1, nparams, nz=100, is_training=True, reuse=False):
  """Creates a simple encoder CNN.

  Args:
    inp_img0: TensorFlow node for input with size B X H X W X C1
    inp_img1: TensorFlow node for input with size B X H X W X C2
    nparams: number of warp parameters
    nz: number of units in last layer, default=100
    is_training: whether batch_norm should be in train mode
    reuse: Whether to reuse weights from an already defined net
  Returns:
    An encoder CNN which computes a final representation with nz
    units.
    As a design principle, we want this to be a lightweight CNN since it'll be
    used often.
  """
  batch_norm_params = {'is_training': is_training}
  with tf.variable_scope('warp_encoder', reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params,
        weights_regularizer=slim.l2_regularizer(0.05),
        activation_fn=tf.nn.relu,
        outputs_collections=end_points_collection):
      imgs_list = [inp_img0, inp_img1]
      conv_list = []
      for ix in range(2):
        with tf.variable_scope('enc_' + str(ix), reuse=reuse) as sc:
          cnv1 = slim.conv2d(imgs_list[ix], 8, [7, 7], stride=2, scope='cnv1')
          cnv1b = slim.conv2d(cnv1, 8, [7, 7], stride=1, scope='cnv1b')
          cnv2 = slim.conv2d(cnv1b, 16, [5, 5], stride=2, scope='cnv2')
          cnv2b = slim.conv2d(cnv2, 16, [5, 5], stride=1, scope='cnv2b')
          cnv3 = slim.conv2d(cnv2b, 32, [3, 3], stride=2, scope='cnv3')
          cnv3b = slim.conv2d(cnv3, 32, [3, 3], stride=1, scope='cnv3b')
          cnv4 = slim.conv2d(cnv3b, 64, [3, 3], stride=2, scope='cnv4')
          cnv4b = slim.conv2d(cnv4, 64, [3, 3], stride=1, scope='cnv4b')
          cnv5 = slim.conv2d(cnv4b, 128, [3, 3], stride=2, scope='cnv5')
          cnv5b = slim.conv2d(cnv5, 128, [3, 3], stride=1, scope='cnv5b')
          conv_list.append(cnv5b)

      cnv5b_concat = tf.concat(conv_list, axis=3)
      cnv6 = slim.conv2d(cnv5b_concat, 128, [3, 3], stride=2, scope='cnv6')
      cnv6b = slim.conv2d(cnv6, 128, [3, 3], stride=1, scope='cnv6b')
      cnv7 = slim.conv2d(cnv6b, 128, [3, 3], stride=2, scope='cnv7')
      cnv7b = slim.conv2d(cnv7, 128, [3, 3], stride=1, scope='cnv7b')
      cnv7b_flat = slim.flatten(cnv7b, scope='cnv7b_flat')
      enc = slim.stack(
          cnv7b_flat, slim.fully_connected, [2*nz, nz, nz], scope='fc')
      warp_params = slim.fully_connected(
          enc, nparams, activation_fn=None, normalizer_fn=None)

    end_points = utils.convert_collection_to_dict(end_points_collection)
    return warp_params, end_points


def scale_predictor(
    inp_img_src, inp_img_trg, rot_s2t, trans_s2t,
    nz=100, is_training=True, reuse=False,
    use_exp_activation=True, exp_range=1):
  """Predicts absolute scale.

  Args:
    inp_img_src: TensorFlow node for input with size B X H X W X C1
    inp_img_trg: TensorFlow node for input with size B X H X W X C2
    rot_s2t: B X 3 X 3 rotation
    trans_s2t: B X 3 X 1 translation
    nz: number of units in last layer, default=100
    is_training: whether batch_norm should be in train mode
    reuse: Whether to reuse weights from an already defined net
    use_exp_activation: Whether to use exp(exp_range*tanh(x)) to restrict scale
    exp_range: Used in activation function above
  Returns:
    scale: Predicted absolute scale
  """
  batch_norm_params = {'is_training': is_training}
  with tf.variable_scope('scale_encoder', reuse=reuse) as sc:
    rot_s2t = tf.reshape(rot_s2t, [-1, 9])
    trans_s2t = tf.reshape(trans_s2t, [-1, 3])
    rot_trans = tf.concat([rot_s2t, trans_s2t], axis=1)
    # rot_trans = tf.Print(
    #     rot_trans, [rot_trans], message='Inp Rot+Trans', summarize=48)
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params,
        weights_regularizer=slim.l2_regularizer(0.05),
        activation_fn=tf.nn.relu,
        outputs_collections=end_points_collection):
      imgs_list = [inp_img_src, inp_img_trg]
      conv_list = []
      for ix in range(2):
        # with tf.variable_scope('enc_' + str(ix), reuse=reuse):
        with tf.variable_scope('enc_conv', reuse=(reuse or ix >= 1)):
          cnv1 = slim.conv2d(imgs_list[ix], 8, [7, 7], stride=2, scope='cnv1')
          cnv1b = slim.conv2d(cnv1, 8, [7, 7], stride=1, scope='cnv1b')
          cnv2 = slim.conv2d(cnv1b, 16, [5, 5], stride=2, scope='cnv2')
          cnv2b = slim.conv2d(cnv2, 16, [5, 5], stride=1, scope='cnv2b')
          cnv3 = slim.conv2d(cnv2b, 32, [3, 3], stride=2, scope='cnv3')
          cnv3b = slim.conv2d(cnv3, 32, [3, 3], stride=1, scope='cnv3b')
          cnv4 = slim.conv2d(cnv3b, 64, [3, 3], stride=2, scope='cnv4')
          cnv4b = slim.conv2d(cnv4, 64, [3, 3], stride=1, scope='cnv4b')
          cnv5 = slim.conv2d(cnv4b, 128, [3, 3], stride=2, scope='cnv5')
          cnv5b = slim.conv2d(cnv5, 128, [3, 3], stride=1, scope='cnv5b')
          conv_list.append(cnv5b)

      cnv5b_concat = tf.concat(conv_list, axis=3)
      cnv6 = slim.conv2d(cnv5b_concat, 128, [3, 3], stride=2, scope='cnv6')
      cnv6b = slim.conv2d(cnv6, 128, [3, 3], stride=1, scope='cnv6b')
      cnv7 = slim.conv2d(cnv6b, 128, [3, 3], stride=2, scope='cnv7')
      cnv7b = slim.conv2d(cnv7, 128, [3, 3], stride=1, scope='cnv7b')
      cnv7b_flat = slim.flatten(cnv7b, scope='cnv7b_flat')
      enc_imgs = slim.stack(
          cnv7b_flat, slim.fully_connected, [2*nz, nz], scope='fc_imgs')
      enc_trans_rot = slim.stack(
          rot_trans, slim.fully_connected, [nz, nz], scope='fc_rot_trans')
      # enc_trans_rot = tf.Print(
      #     enc_trans_rot, [enc_trans_rot], message='enc_trans_rot')
      # enc_imgs = tf.Print(
      #     enc_imgs, [enc_imgs], message='enc_imgs')
      enc_feat = tf.concat([enc_imgs, enc_trans_rot], axis=1)
      scale_params = slim.fully_connected(
          enc_feat, 1, activation_fn=None, normalizer_fn=None)
      if use_exp_activation:
        scale_params = tf.exp(exp_range*tf.tanh(scale_params))

      # scale_params = tf.sigmoid(scale_params)
      # scale_params = tf.Print(
      #     scale_params, [scale_params], message='Scale', summarize=4)

    end_points = utils.convert_collection_to_dict(end_points_collection)
    return scale_params, end_points
