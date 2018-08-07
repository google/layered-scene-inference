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
"""Script for dumping tensorflow events to a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import re

from absl import app
from absl import gflags
from absl import logging as log
from lsi.visualization import html_utils
import scipy.misc
import scipy.misc.pilutil
import tensorflow as tf

FLAGS = gflags.FLAGS
flags.DEFINE_string('exp_name', 'synth_ldi_pred_encdec',
                    'Name of the experiment')
flags.DEFINE_string('checkpoint_dir', '/tmp/experiments/snapshots/',
                    'Root directory for tensorflow output files')
flags.DEFINE_integer('num_trial', -1,
                     'If >=, we append "_trial<num_trial>" dir name')
flags.DEFINE_integer('img_height', 256, 'Image height to be saved')
flags.DEFINE_integer('img_width', 256, 'Image Width to be saved')


def write_iter_page(keys, height, width):
  """Create summary page.

  Args:
    keys: image names to be saved
    height: image height
    width: image width
  Returns:
    html: page text
  """
  table_rows = []
  for inst in range(3):
    table_cols = [
        html_utils.image('inst{}/{}.jpg'.format(inst, k), k, height, width)
        for k in keys
    ]
    table_rows.append(html_utils.table_row(table_cols))

  html = html_utils.html_page(html_utils.table(table_rows))
  return html


def parse_event(sess, event, image_dict, summary_keys, out_dir):
  """Save visualizations for an event.

  Args:
    sess: tf session
    event: tf event
    image_dict: keys to look for
    summary_keys: keys to save in summary page
    out_dir: directory to save
  """
  opts = FLAGS
  os.makedirs(out_dir)
  summary_path = osp.join(out_dir, 'summary.html')
  with open(summary_path, mode='w') as f:
    f.write(write_iter_page(summary_keys, opts.img_height, opts.img_width))

  for inst_id in range(3):
    os.makedirs(osp.join(out_dir, 'inst{}'.format(inst_id)))
  img_keys = image_dict.keys()

  for v in event.summary.value:
    if v.tag.split('/')[0] in img_keys:
      inst_id = v.tag.split('/')[-1]
      img_id = v.tag.split('/')[0]
      img_name = 'inst{}/{}.jpg'.format(inst_id, image_dict[img_id])

      im_tensor = tf.image.decode_image(v.image.encoded_image_string)
      img = sess.run(im_tensor)
      if img.shape[2] == 1:
        img = img[:, :, 0]
      img = scipy.misc.imresize(img, (opts.img_height, opts.img_width))
      with open(osp.join(out_dir, img_name), mode='w') as f:
        scipy.misc.imsave(f, img)


def parse_events_file(sess,
                      log_file,
                      image_dict,
                      summary_keys,
                      out_dir,
                      freq=5000):
  """Parse an events file."""
  for event in tf.train.summary_iterator(log_file):
    global_step = event.step
    if global_step > 0 and global_step % freq == 0:
      log.info('Step: %d', global_step)
      parse_event(sess, event, image_dict, summary_keys,
                  osp.join(out_dir, 'iter_{}'.format(global_step)))


def main(_):
  image_dict = {
      'src_imgs': 'src_im',
      'trg_imgs': 'trg_im',
      'src_disp_pred_layer_0': 'src_l0_disp',
      'src_tex_pred_layer_0': 'src_l0_tex',
      'src_disp_pred_layer_1': 'src_l1_disp',
      'src_tex_pred_layer_1': 'src_l1_tex',
      'trg_splat_0': 'src2trg_l0_splat',
      'trg_splat_1': 'src2trg_l1_splat',
      'trg_disp_pred_layer_0': 'trg_l0_disp',
      'trg_tex_pred_layer_0': 'trg_l0_tex',
      'trg_disp_pred_layer_1': 'trg_l1_disp',
      'trg_tex_pred_layer_1': 'trg_l1_tex',
      'src_splat_0': 'trg2src_l0_splat',
      'src_splat_1': 'trg2src_l1_splat',
  }
  summary_keys = [
      'src_im', 'src_l0_disp', 'src_l0_tex', 'src_l1_disp', 'src_l1_tex'
  ]

  FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.exp_name)

  out_dir = FLAGS.checkpoint_dir + '_tf_events'
  if FLAGS.num_trial >= 0:
    FLAGS.checkpoint_dir += '_trial{}'.format(FLAGS.num_trial)
  sess = tf.Session()

  re_init = re.compile(r'^events')
  event_files = os.listdir(FLAGS.checkpoint_dir)
  event_files = [f for f in event_files if re_init.search(f)]

  for event_file in event_files:
    try:
      parse_events_file(sess, osp.join(FLAGS.checkpoint_dir, event_file),
                        image_dict, summary_keys, out_dir)
    except tf.errors.DataLossError:
      continue


if __name__ == '__main__':
  app.run(main)
