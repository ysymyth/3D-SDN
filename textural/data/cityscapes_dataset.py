# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
import torch
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
from torchvision import transforms
import json
import data.cityscapes_labels as cityscapes_labels


class CustomDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.phase = 'train' if opt.isTrain else 'val'

        # Obtain A_paths, B_paths, inst_paths
        self.getlists(opt)

        random.Random(20).shuffle(self.A_paths)
        random.Random(20).shuffle(self.B_paths)
        random.Random(20).shuffle(self.inst_paths)
        random.Random(20).shuffle(self.pose_paths)
        random.Random(20).shuffle(self.normal_paths)
        self.dataset_size = len(self.A_paths)
        print('{}_set size: {}'.format(self.phase, self.dataset_size))

    def __getitem__(self, index):
        # input A (label maps)
        A_path = self.A_paths[index]
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = pose_tensor = normal_tensor = depth_tensor = 0

        # input B (real images)
        if self.opt.isTrain or True:
            B_path = self.B_paths[index]
            B = Image.open(B_path).convert('RGB')
            #if self.use_augmentation: B = self.img_jitter(B)
            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B)

        # if using instance maps
        if not self.opt.no_instance:
            try:
                inst_path = self.inst_paths[index]
                inst = Image.open(inst_path)
                inst_tensor = transform_A(inst)
                if self.opt.inst_precomputed_path:
                    inst_tensor = inst_tensor * 255.0
                    inst_tensor *= 1000
                    inst_tensor[inst_tensor == 0] = A_tensor[inst_tensor == 0]
            except FileNotFoundError:
                inst_tensor = A_tensor

        # if using pose info
        if self.opt.feat_pose:  # add pose info to features
            if self.opt.feat_pose_num_bins > 0:
                pose_tensor = np.zeros((1, A_tensor.size(1), A_tensor.size(2)))
            else:
                pose_tensor = np.zeros((2, A_tensor.size(1), A_tensor.size(2)))
            try:
                d = json.load(open(self.pose_paths[index]))
                inst_map = Image.open(self.pose_paths[index].replace('.json', '.png'))
                inst_map = transform_A(inst_map) * 255.0
                inst_map = inst_map.numpy()[0]
                from math import pi, cos, sin
                if self.opt.feat_pose_num_bins:
                    bins = np.array(list(range(-180, 181, 360 // self.opt.feat_pose_num_bins))) / 180
                for inst in np.unique(inst_map):
                    if inst == 0 or (inst_map == inst).sum() < 256:
                        continue
                    alpha = d[str(int(inst))]["alpha"]
                    if self.opt.feat_pose_num_bins > 0:
                        pose_tensor[0, inst_map == inst] = np.digitize(alpha / pi, bins)
                    else:
                        pose_tensor[0, inst_map == inst] = cos(alpha)
                        pose_tensor[1, inst_map == inst] = sin(alpha)
            except FileNotFoundError:  # no cars
                pass
            pose_tensor = torch.from_numpy(pose_tensor)
            pose_tensor = pose_tensor.int() if self.opt.feat_pose_num_bins else pose_tensor.float()

        if self.opt.feat_normal:  # add normal info to features
            try:
                normal_map = Image.open(self.normal_paths[index])
                normal_tensor = transform_B(normal_map) + 1 / 255  # bias caused by 0..256 instead 0..255
            except FileNotFoundError:  # no cars
                normal_tensor = torch.zeros(B_tensor.size())

        if not self.opt.segm_precomputed_path:
            _A_tensor = A_tensor.clone()
            for label in self.labels:
                A_tensor[_A_tensor == label.id] = label.trainId + 1 if label.trainId != 255 else 0

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'path': A_path,
                      'pose': pose_tensor, 'normal': normal_tensor, 'depth': depth_tensor}

        return input_dict

    def getlists(self, opt):
        subset = "train" if opt.isTrain else "val"

        annotations = json.load(open("{}/annotations/instancesonly_gtFine_{}.json".format(self.root, subset)))['images']
        self.A_paths, self.B_paths, self.inst_paths = [], [], []
        self.pose_paths, self.normal_paths = [], []
        for item in annotations:
            city = item['file_name'].split('_')[0]
            name = item['file_name']  # example: darmstadt_000035_000019_leftImg8bit.png
            if opt.segm_precomputed_path:
                self.A_paths.append(os.path.join(opt.segm_precomputed_path, city, name))
            else:
                self.A_paths.append(os.path.join(self.root, "gtFine", subset, city, item['seg_file_name'].replace('instance', 'label')))
            self.B_paths.append(os.path.join(self.root, "images", name))
            if opt.inst_precomputed_path:
                self.inst_paths.append(os.path.join(opt.inst_precomputed_path, city, name.replace('_leftImg8bit', '')))
            else:
                self.inst_paths.append(os.path.join(self.root, "gtFine", subset, city, item['seg_file_name']))
            if opt.feat_pose:
                self.pose_paths.append(os.path.join(opt.feat_pose, city, name.replace('_leftImg8bit.png', '.json')))
            if opt.feat_normal:
                self.normal_paths.append(os.path.join(opt.feat_normal, city, name.replace('_leftImg8bit.png', '-normal.png')))
        self.labels = cityscapes_labels.labels

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'Cityscapes CustomDataset'
