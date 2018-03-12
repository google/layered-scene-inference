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

"""Refinement helpers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lsi.geometry import layers
from lsi.nnutils import helpers as nn_helpers
from lsi.nnutils import nets
reload(nets)


def refine_params(src_imgs, trg_imgs, nparams):
  """Outputs params for warping.

  Args:
    src_imgs: are L X [...] X H X W X C
    trg_imgs: [...] X H X W X C targets
    nparams: number of transformation params
  Returns:
    pred_params: are L X [...] X nparams, transformation parameters
  """
  with tf.name_scope('refine_params'):
    init_dims = src_imgs.get_shape().as_list()[:-3:]
    n_layers = init_dims[0]
    end_dims_src = src_imgs.get_shape().as_list()[-3::]

    trg_rep_dims = [n_layers]
    trg_rep_dims += [1 for _ in range(len(trg_imgs.get_shape()))]
    trg_imgs = tf.tile(tf.expand_dims(trg_imgs, axis=0), trg_rep_dims)

    end_dims_trg = trg_imgs.get_shape().as_list()[-3::]

    prod_init_dims = init_dims[0]
    for ix in range(1, len(init_dims)):
      prod_init_dims *= init_dims[ix]
    src_imgs = tf.reshape(src_imgs, [prod_init_dims] + end_dims_src)
    trg_imgs = tf.reshape(trg_imgs, [prod_init_dims] + end_dims_trg)
    pred_params, _ = nets.warp_predictor(src_imgs, trg_imgs, nparams)
    pred_params = tf.reshape(pred_params, init_dims + [nparams])
    return pred_params


def corner_refine(layer_imgs, params):
  """Outputs transformations to each src_layer.

  Args:
    layer_imgs: are L X B X H X W X C
    params: L X B X 8 warp parameters
  Returns:
    layer_imgs_trg: are L X B X H X W X C, after transformation
  """
  with tf.name_scope('corner_refine'):
    imgs_shape = layer_imgs.get_shape().as_list()
    init_dims = params.get_shape().as_list()[:-1:]
    params = tf.reshape(params, init_dims + [4, 2])
    return layers.corner_transform(
        layer_imgs,
        nn_helpers.pixel_coords(imgs_shape[1], imgs_shape[2], imgs_shape[3]),
        params)
