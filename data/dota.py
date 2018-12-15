"""dota dataset classes

original author: francisco massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

updated by: ellis brown, max degroot
for dota file list is:
|data/
|data/trainvalsplit/
|data/testsplit/
|data|trainvalsplit/images/
|data|trainvalsplit/labeltxt/
|data|trainvalsplit/train.txt

the image folder is xxx.jpg
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np


import zipfile
import codecs

DOTA_CLASSES = (  # always index 0
                        'plane', 'baseball-diamond',
                        'bridge', 'ground-track-field',
                        'small-vehicle', 'large-vehicle',
                        'ship', 'tennis-court',
                        'basketball-court', 'storage-tank',
                        'soccer-ball-field', 'roundabout',
                        'harbor', 'swimming-pool',
                        'helicopter')

# note: if you used our download scripts, this should be right
#DOTA_ROOT = osp.join(HOME, '/data/dota-split-300/')
DOTA_ROOT = '/home/lzp/data/dota-split-300/'

class DOTAAnnotationTransform(object):
    """transforms a dota annotation into a tensor of bbox coords and label index
    initilized with a dictionary lookup of classnames to indexes

    arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of voc's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: false)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind =  class_to_ind or dict(
            zip(DOTA_CLASSES, range(len(DOTA_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, targets, width, height):
        """
        arguments:
            target (annotation) : the target annotation to be made usable
                will be an et.element
        returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for target in targets:
            difficult = (int(target[9]) != 0)  #0 -> easy
            if not self.keep_difficult and difficult:
                continue
            name = target[8].lower().strip()
            pts = ['x1', 'y1', 'x2', 'y2','x3', 'y3', 'x4', 'y4']
            bndbox = []
            # for i,pt in enumerate(pts):
            #     if (i == 0 or i == 1 or i == 6 or i == 7):
            #
            #         cur_pt = max(float(target[i]) - 1, 0)
            #         cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
            #         bndbox.append(cur_pt)
            xmin, xmax, ymin, ymax = min(target[0], min(target[2], min(target[4], target[6]))), \
                                     max(target[0], max(target[2], max(target[4], target[6]))), \
                                     min(target[1], min(target[3], min(target[5], target[7]))), \
                                     max(target[1], max(target[3], max(target[5], target[7])))

            cur_pt_1 = max(float(xmin) - 1.0, 0.0)
            cur_pt_1 = 1.0*cur_pt_1 / width
            bndbox.append(cur_pt_1)
            cur_pt_2 = max(float(ymin) - 1.0, 0.0)
            cur_pt_2 = 1.0*cur_pt_2 / height
            bndbox.append(cur_pt_2)
            cur_pt_3 = max(float(xmax) - 1.0, 0.0)
            cur_pt_3 = 1.0*cur_pt_3 / width
            bndbox.append(cur_pt_3)
            cur_pt_4 = max(float(ymax) - 1.0, 0.0)
            cur_pt_4 = 1.0*cur_pt_4 / height
            bndbox.append(cur_pt_4)

            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        return res


class DOTADetection(data.Dataset):
    """voc detection dataset object

    input is image, target is annotation

    arguments:
        root (string): filepath to vocdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'voc2007')
    """

    def __init__(self, root,
                 image_sets=['trainvalsplit'],
                 transform=None, target_transform=DOTAAnnotationTransform(),
                 dataset_name='DOTA',
                 train_test="train"):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        train_rootpath = osp.join(self.root, 'trainvalsplit')
        self._annopath = osp.join(train_rootpath, 'labelTxt', '%s.txt')
        self._imgpath = osp.join(train_rootpath, 'images', '%s.png')
        self.ids = list()
        if train_test == "train":
            for line in open(osp.join(train_rootpath, 'train11.txt')):
                self.ids.append((train_rootpath, line.strip()))
        else:
            for line in open(osp.join(train_rootpath, 'test.txt')):
                print(line)
                self.ids.append((train_rootpath, line.strip()))

        # for line in open(osp.join(train_rootpath,  'train11.txt')):
        #     self.ids.append((train_rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        # print(img_id)
        anno_file_path  = (self._annopath % img_id[1])
        # print(anno_file_path)
        img = cv2.imread(self._imgpath % img_id[1])

        # print("Image Path")
        # print(self._imgpath % img_id[1])

        height, width, channels = img.shape
        # print("Image Channel:")
        # print(height,width,channels)
        #target = et.parse(self._annopath % img_id).getroot()
        f = codecs.open(anno_file_path,"r")
        targets = f.readlines()
        f.close()
        # print("GT is:")
        # print(targets)
        targets = [target.strip().strip('\n').split(' ') for target in targets]
        # print("targets11111111.size")
        # print(np.shape(targets))

        if self.target_transform is not None:
            targets = self.target_transform(targets, width, height)
        # print("After target_transform GT is:")
        # print(targets)
        if self.transform is not None:
            targets = np.array(targets)

            # print("After transform GT is:")
            # print(targets)
            # if targets.size < 8:   # avoide the null .txt
            #     # return torch.from_numpy(img).permute(2, 0, 1)., [], height, width
            #     return torch.from_numpy(img).permute(2, 0, 1).float(), [], height, width

            img, boxes, labels = self.transform(img, targets[:, 0:4], targets[:,4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            targets = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # return torch.from_numpy(img).permute(2, 0, 1), targets, height, width
        return torch.from_numpy(img).permute(2, 0, 1), targets, height, width #    .float()


        # return torch.from_numpy(img), targets, height, width

    def pull_image(self, index):
        '''returns the original image object at index in pil form

        note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        argument:
            index (int): index of img to show
        return:
            pil img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id[1], cv2.IMREAD_COLOR) #.float()

    def pull_anno(self, index):
        '''returns the original annotation of image at index

        note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        argument:
            index (int): index of img to get annotation of
        return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]


        #anno = et.parse(self._annopath % img_id).getroot()
        f = codecs.open(self._annopath % img_id[1],"r")
        anno = f.readlines()
        f.close()
        anno = [target.strip().split(' ') for target in anno]
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''returns the original image at an index in tensor form

        note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        argument:
            index (int): index of img to show
        return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

