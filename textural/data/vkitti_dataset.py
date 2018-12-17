import os.path
import random
import torch
import numpy as np
import pickle
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import json
from torchvision import transforms
import time


class CustomDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.phase = 'train' if opt.isTrain else 'test'
        self.root = opt.dataroot
        self.root_img = os.path.join(self.root, "vkitti_1.3.1_rgb")
        self.root_segm = os.path.join(self.root, "vkitti_1.3.1_myscenegt") \
            if not opt.segm_precomputed_path else opt.segm_precomputed_path
        self.root_inst = os.path.join(self.root, "vkitti_1.3.1_inst") \
            if not opt.inst_precomputed_path else opt.inst_precomputed_path

        worldIds = ['0001', '0002', '0006', '0018', '0020']
        sceneIds = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']
        worldSizes = [446, 232, 269, 338, 836]
        splitRanges = {'train': [range(0, 356), range(0, 185), range(69, 270), range(0, 270), range(167, 837)],
                       'test': [range(356, 447), range(185, 233), range(0, 69), range(270, 339), range(0, 167)]}
        self.list = []
        for worldId in worldIds:
            for sceneId in sceneIds:
                for imgId in splitRanges[self.phase][worldIds.index(worldId)]:
                    self.list += ['%s/%s/%05d.png' % (worldId, sceneId, imgId)]

        self.use_augmentation = opt.isTrain and opt.use_augmentation
        if self.use_augmentation:
            self.img_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)

        random.Random().shuffle(self.list)
        self.dataset_size = len(self.list)
        print('{}_set size: {}'.format(self.phase, self.dataset_size))

    def __getitem__(self, index):
        # input A (label maps)
        A_path = os.path.join(self.root_segm, self.list[index])
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0
        if self.opt.segm_precomputed_path:  # NOTE: due to the model, have to add 1
            A_tensor = A_tensor + 1
        B_tensor = inst_tensor = feat_tensor = pose_tensor = normal_tensor = depth_tensor = 0

        # input B (real images)
        B_path = os.path.join(self.root_img, self.list[index])
        B = Image.open(B_path).convert('RGB')
        if self.use_augmentation:
            B = self.img_jitter(B)
        transform_B = get_transform(self.opt, params)
        B_tensor = transform_B(B)

        # if using instance maps
        if not self.opt.no_instance:
            try:
                inst_path = os.path.join(self.root_inst, self.list[index])
                inst = Image.open(inst_path)
                inst_tensor = transform_A(inst)
                if self.opt.inst_precomputed_path:  # need segm to fill in background segments
                    inst_tensor = inst_tensor * 255.0
                    inst_tensor *= 1000  # different from semantic labels
                    if self.opt.segm_precomputed_path:
                        # use car semantic information from inst map; now only remove
                        A_tensor[(inst_tensor == 0) & (A_tensor == 2)] = 5
                        A_tensor[(inst_tensor == 0) & (A_tensor == 12)] = 5
                    inst_tensor[inst_tensor == 0] = A_tensor[inst_tensor == 0]

                if self.opt.load_features:  # TODO: add load_feature
                    feat_path = self.feat_paths[index]
                    feat = Image.open(feat_path).convert('RGB')
                    norm = normalize()
                    feat_tensor = norm(transform_A(feat))
            except FileNotFoundError:
                inst_tensor = A_tensor
                #inst_tensor = torch.zeros(A_tensor.size())

        if self.opt.feat_pose:  # add pose info to features
            if self.opt.feat_pose_num_bins > 0:
                pose_tensor = np.zeros((1, A_tensor.size(1), A_tensor.size(2)))
            else:
                pose_tensor = np.zeros((2, A_tensor.size(1), A_tensor.size(2)))
            try:
                dict_path = os.path.join(self.opt.feat_pose, self.list[index])
                d = json.load(open(dict_path.replace('png', 'json')))
                inst_map = Image.open(dict_path)
                inst_map = transform_A(inst_map) * 255.0
                inst_map = inst_map.numpy()[0]
                from math import pi, cos, sin
                if self.opt.feat_pose_num_bins:
                    bins = np.array(list(range(-180, 181, 360 // self.opt.feat_pose_num_bins))) / 180
                for inst in np.unique(inst_map):
                    if inst == 0:
                        continue
                    if not str(int(inst)) in d:
                        continue
                    #assert str(int(inst)) in d, (self.list[index], d.keys(), np.unique(inst_map))
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
                normal_map = Image.open(os.path.join(self.opt.feat_normal, self.list[index].replace('.png', '-normal.png')))
                normal_tensor = transform_B(normal_map) + 1 / 255  # bias caused by 0..256 instead of 0..255
            except FileNotFoundError:  # no cars
                #print(self.list[index], 'has no car')
                normal_tensor = torch.zeros(B_tensor.size())

        if self.opt.feat_depth:  # add depth info to features
            try:
                depth_map = Image.open(os.path.join(self.opt.feat_depth, self.list[index].replace('.png', '-depth.png')))
                depth_tensor = transform_A(depth_map)
                depth_tensor = 1.0 - depth_tensor.float() / 65535.0
            except FileNotFoundError:
                depth_tensor = torch.zeros(A_tensor.size())

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'path': self.list[index],
                      'pose': pose_tensor, 'normal': normal_tensor, 'depth': depth_tensor}

        return input_dict

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'VKitti CustomDataset'
