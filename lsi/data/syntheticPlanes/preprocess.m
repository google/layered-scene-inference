% Copyright 2018 Google LLC
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%      http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.


% Directory where processed images of pascal objects should be stored
pascal_objects_dir = '/data0/shubhtuls/code/lsi/cachedir/sbd/objects';

% Directory where images of pascal objects are stored
voc_dir = '/data1/shubhtuls/cachedir/Datasets/VOCdevkit';

% Directory where pascal segmentation mask annotations are stored
pascal_anno_dir =  '/data0/shubhtuls/code/lsi/cachedir/segkps';

if(~exist(pascal_objects_dir,'dir'))
    mkdir(pascal_objects_dir);
end

voc_code_path = fullfile(voc_dir, 'VOCcode');
addpath(voc_code_path);

voc_classes = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'};

for c = 1:length(voc_classes)
  cls = voc_classes{c};
  segs = load(fullfile(pascal_anno_dir, cls)); segs = segs.segmentations;
  for o = 1:length(segs.voc_image_id)
    voc_id = segs.voc_image_id{o};
    rec_id = segs.voc_rec_id(o);
    rec = VOCreadrecxml(fullfile(voc_dir, 'VOC2012', 'Annotations', [voc_id '.xml']));
    obj_rec = rec.objects(rec_id);
    if (obj_rec.truncated || obj_rec.occluded || obj_rec.difficult)
      continue;
    end
    bbox = obj_rec.bbox;
    poly_x = segs.poly_x{o};
    poly_y = segs.poly_y{o};

    img = imread(fullfile(voc_dir, 'VOC2012', 'JPEGImages', [voc_id '.jpg']));
    img = double(img)/255.0;
    mask = roipoly(img, poly_x, poly_y);
    mask = double(mask);
    for col = 1:3
      img(:,:,col) = img(:,:,col).*mask;
    end
    %img = cat(3, img, mask);
    img = img(bbox(2):bbox(4), bbox(1):bbox(3),:);
    mask = mask(bbox(2):bbox(4), bbox(1):bbox(3));

    img_path = fullfile(pascal_objects_dir, [voc_id '_' num2str(rec_id) '.png']);
    imwrite(img, img_path, 'Alpha', mask);
    
  end
end