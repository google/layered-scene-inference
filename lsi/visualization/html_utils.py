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


def html_page(content):
  init_text = '<!DOCTYPE html>\n<html>\n<head>\n{}\n</head>\n<body>\n'.format(
      page_style()
  )
  end_text = '\n</html>\n</body>\n'

  return init_text + content + end_text


def page_style():
  """Html style string.

  Args:
  Returns:
    style_str: HTML sytle string
  """

  style_str = '''<style>
table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
}
th, td {
    padding: 10px;
}
</style>'''
  return style_str


def image(rel_path, caption, height, width):
  img_str = '<img src="{}" alt="{}" style="width:{}px;height:{}px;">'.format(
      rel_path, caption, width, height)
  return img_str


def table(table_rows):
  init_text = '<table style="width:100%">\n'
  end_text = '</table>\n'
  table_str = init_text
  for tr in table_rows:
    table_str += '<tr>\n{}</tr>\n'.format(tr)
  table_str += end_text
  return table_str


def table_row(table_cols):
  row_str = ''
  for tc in table_cols:
    row_str += '<td>\n{}\n</td>\n'.format(tc)
  return row_str
