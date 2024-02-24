# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

import pdb


@DATASET_REGISTRY.register()
class GroupVC_SC(ImageDataset):
    """GroupVC Group verison.

    """
    _junk_pids = [-1]

    dataset_dir = '/media/data3/zhangquan/documents/CCReID/HHL/data/'
    dataset_name = "GroupVC_SC"

    def __init__(self, root='datasets', **kwargs):

        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = osp.join(self.dataset_dir, 'GroupVC-multicc')
        self.train_list = osp.join(self.data_dir, 'train.txt')
        self.query_list = osp.join(self.data_dir, 'query_SCS.txt')
        self.gallery_list = osp.join(self.data_dir, 'gallery_SCS.txt')

        train = lambda: self.process_dir(self.train_list)
        query = lambda: self.process_dir(self.query_list, is_train=False)
        gallery = lambda: self.process_dir(self.gallery_list, is_train=False)

        super(GroupVC_SC, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        data = []
        for line in open(dir_path):
            img_item = line.strip()
            img_path = img_item.split(' ')[0]
            img_ids = img_item.split(' ')[1].split(',')[0]
            cam_id = int(img_item.split(' ')[1].split(',')[1])
            img_num_ps = int(img_item.split(' ')[2].split(',')[0])

            assert 1 <= cam_id <= 4
            cam_id = cam_id - 1

            img_path = osp.join(self.data_dir, img_path)
            img_gid = int(img_ids)
            img_num_ps = int(img_num_ps)

            if is_train:
                img_gid = self.dataset_name + "_" + str(img_gid)
                cam_id = self.dataset_name + "_" + str(cam_id)

            data.append((img_path, img_gid, cam_id, img_num_ps))
        return data

@DATASET_REGISTRY.register()
class GroupVC_CC(ImageDataset):
    """GroupVC Group verison.

    """
    _junk_pids = [-1]

    dataset_dir = '/media/data3/zhangquan/documents/CCReID/HHL/data/'
    dataset_name = "GroupVC_CC"

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = osp.join(self.dataset_dir, 'GroupVC-multicc')
        self.train_list = osp.join(self.data_dir, 'train.txt')
        self.query_list = osp.join(self.data_dir, 'query_CCS.txt')
        self.gallery_list = osp.join(self.data_dir, 'gallery_CCS.txt')

        train = lambda: self.process_dir(self.train_list)
        query = lambda: self.process_dir(self.query_list, is_train=False)
        gallery = lambda: self.process_dir(self.gallery_list, is_train=False)

        super(GroupVC_CC, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        data = []
        for line in open(dir_path):
            img_item = line.strip()
            img_path = img_item.split(' ')[0]
            img_ids = img_item.split(' ')[1].split(',')[0]
            cam_id = int(img_item.split(' ')[1].split(',')[1])
            img_num_ps = int(img_item.split(' ')[2].split(',')[0])

            cam_id = cam_id - 1

            img_path = osp.join(self.data_dir, img_path)
            img_gid = int(img_ids)
            img_num_ps = int(img_num_ps)

            if is_train:
                img_gid = self.dataset_name + "_" + str(img_gid)
                cam_id = self.dataset_name + "_" + str(cam_id)

            data.append((img_path, img_gid, cam_id, img_num_ps))
        return data
