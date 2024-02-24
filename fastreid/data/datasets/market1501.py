# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings
import pdb

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Market1501(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = '/media/data3/zhangquan/documents/CCReID/HHL/data'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    dataset_name = "market1501"

    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)
        
        self.train_list = osp.join(self.data_dir, 'bounding_box_train.txt')
        self.query_list = osp.join(self.data_dir, 'query.txt')
        self.gallery_list = osp.join(self.data_dir, 'bounding_box_test.txt')

        train = lambda: self.process_dir(self.train_list)
        query = lambda: self.process_dir(self.query_list, is_train=False)
        gallery = lambda: self.process_dir(self.gallery_list, is_train=False)

        super(Market1501, self).__init__(train, query, gallery, **kwargs)
    
    def process_dir(self, dir_path, is_train=True):
        data = []
        for line in open(dir_path):
            img_item = line.strip()
            
            img_path, img_gid, img_num_ps = img_item.split(' ')
            img_gid = int(img_gid)
            img_num_ps = int(img_num_ps)
            
            if img_gid == -1:
                continue
            
            cam_id = img_path.split('/')[-1].split('_')[1][1]
            cam_id = int(cam_id) - 1

            img_path = osp.join(self.data_dir, img_path)

            if is_train:
                img_gid = self.dataset_name + "_" + str(img_gid)
                cam_id = self.dataset_name + "_" + str(cam_id)
  
            data.append((img_path, img_gid, cam_id, img_num_ps))

        return data
