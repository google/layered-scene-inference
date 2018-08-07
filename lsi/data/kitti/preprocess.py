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
"""Code for preprocessing KITTI data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import os

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'kitti_data_root', '/datasets/kitti',
    'Directory where flowers data images are cameras are stored')

flags.DEFINE_string(
    'spss_exec',
    '/code/lsi/external/spsstereo_git_patch/spsstereo',
    'Directory where flowers data images are cameras are stored')


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


def main(_):
  root_dir = os.path.join(FLAGS.kitti_data_root, 'kitti_raw')
  exclude_img = '2011_09_26_drive_0117_sync/image_02/data/0000000074.png'
  seq_names = raw_city_sequences()
  img_list_src = []
  folder_list_spss = []

  for seq_id in seq_names:
    seq_date = seq_id[0:10]
    seq_dir = os.path.join(root_dir, seq_date, '{}_sync'.format(seq_id))
    for root, _, filenames in os.walk(os.path.join(seq_dir, 'image_02')):
      for filename in fnmatch.filter(filenames, '*.png'):
        src_img_name = os.path.join(root, filename)
        if exclude_img not in src_img_name:
          img_list_src.append(src_img_name)
          folder_list_spss.append(
              os.path.join(root_dir, 'spss_stereo_results',
                           src_img_name.split('/')[-4]))

  img_list_trg = [f.replace('image_02', 'image_03') for f in img_list_src]
  for ix, (src_im, trg_im, dst) in enumerate(
      zip(img_list_src, img_list_trg, folder_list_spss)):
    if ix % 50 == 0:
      print('{}/{}'.format(ix, len(img_list_src)))
    if not os.path.exists(dst):
      os.makedirs(dst)
    os.system('{} {} {}'.format(FLAGS.spss_exec, src_im, trg_im))
    os.system('mv ./*.png {}/'.format(dst))
    os.system('mv ./*.txt {}/'.format(dst))


if __name__ == '__main__':
  app.run(main)
