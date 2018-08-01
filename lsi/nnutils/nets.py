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
# from absl import logging as log
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
