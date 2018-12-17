import os
import json
import sys
import torch
import lib.utils.data as torchdata
import cv2
from torchvision import transforms
from scipy.misc import imread, imresize
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from datasets.vkitti_utils import get_tables, get_lists


# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p


class TrainDataset(torchdata.Dataset):
    def __init__(self, opt, max_sample=-1, batch_per_gpu=1):
        self.root_dataset = opt.root_dataset
        self.root_img = os.path.join(self.root_dataset, 'vkitti_1.3.1_rgb')
        self.root_seg = os.path.join(self.root_dataset, 'vkitti_1.3.1_scenegt')
        self.table_segm = get_tables('segm', opt.root_dataset)

        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        self.random_flip = opt.random_flip
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        self.batch_record_list = []

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229, 0.224, 0.225])])

        self.img_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

        self.list_sample = get_lists('train')

        self.if_shuffled = False
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def _get_sub_batch(self):
        while True:
            # get a sample record
            self.batch_record_list.append(self.list_sample[self.cur_idx])

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list) == self.batch_per_gpu:
                batch_records = self.batch_record_list
                self.batch_record_list = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSize, list):
            this_short_size = np.random.choice(self.imgSize)
        else:
            this_short_size = self.imgSize

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_resized_size = np.zeros((self.batch_per_gpu, 2), np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = 375, 1242  # fixed ...
            this_scale = min(this_short_size / min(img_height, img_width),
                             self.imgMaxSize / max(img_height, img_width))
            img_resized_height, img_resized_width = img_height * this_scale, img_width * this_scale
            batch_resized_size[i, :] = img_resized_height, img_resized_width
        batch_resized_height = np.max(batch_resized_size[:, 0])
        batch_resized_width = np.max(batch_resized_size[:, 1])

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_resized_height = int(round2nearest_multiple(batch_resized_height, self.padding_constant))
        batch_resized_width = int(round2nearest_multiple(batch_resized_width, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate,\
            'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(self.batch_per_gpu, 3, batch_resized_height, batch_resized_width)
        batch_segms = torch.zeros(self.batch_per_gpu, batch_resized_height // self.segm_downsampling_rate,
                                  batch_resized_width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_img, this_record)
            segm_path = os.path.join(self.root_seg, this_record)
            img = Image.open(image_path)
            segm = imread(segm_path, mode='RGB')
            worldId, sceneId = this_record.split('/')[:2]
            segm = np.apply_along_axis(lambda a: self.table_segm[(worldId, sceneId, a[0], a[1], a[2])], 2, segm)
            segm = segm.astype(np.uint8)

            # add data augmentation: color jittering
            img = self.img_jitter(img)
            img = np.array(img)

            assert(img.ndim == 3)
            assert(segm.ndim == 2)
            assert(img.shape[0] == segm.shape[0])
            assert(img.shape[1] == segm.shape[1])

            if self.random_flip:
                random_flip = np.random.choice([0, 1])
                if random_flip == 1:
                    img = cv2.flip(img, 1)
                    segm = cv2.flip(segm, 1)

            # note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_resized_size[i, 0], batch_resized_size[i, 1]), interp='bilinear')
            segm = imresize(segm, (batch_resized_size[i, 0], batch_resized_size[i, 1]), interp='nearest')

            # to avoid seg label misalignment
            segm_rounded_height = round2nearest_multiple(segm.shape[0], self.segm_downsampling_rate)
            segm_rounded_width = round2nearest_multiple(segm.shape[1], self.segm_downsampling_rate)
            segm_rounded = np.zeros((segm_rounded_height, segm_rounded_width), dtype='uint8')
            segm_rounded[:segm.shape[0], :segm.shape[1]] = segm

            segm = imresize(segm_rounded, (segm_rounded.shape[0] // self.segm_downsampling_rate,
                                           segm_rounded.shape[1] // self.segm_downsampling_rate),
                            interp='nearest')
            # image to float
            img = img.astype(np.float32)[:, :, ::-1]  # RGB to BGR!!!
            img = img.transpose((2, 0, 1))
            img = self.img_transform(torch.from_numpy(img.copy()))

            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = torch.from_numpy(segm.astype(np.int)).long()

        batch_segms = batch_segms - 1  # label from -1 to 149
        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return int(1e6)  # It's a fake length due to the trick that every loader maintains its own list
        # return self.num_sampleclass


class ValDataset(torchdata.Dataset):
    def __init__(self, opt, max_sample=-1):
        self.root_dataset = opt.root_dataset
        self.root_img = os.path.join(self.root_dataset, 'vkitti_1.3.1_rgb')
        self.root_seg = os.path.join(self.root_dataset, 'vkitti_1.3.1_scenegt')
        self.table_segm = get_tables('segm', opt.root_dataset)

        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229, 0.224, 0.225])])

        self.list_sample = get_lists(opt.split)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_img, this_record)
        segm_path = os.path.join(self.root_seg, this_record)
        img = imread(image_path, mode='RGB')
        img = img[:, :, ::-1]  # BGR to RGB!!!
        segm = imread(segm_path)
        worldId, sceneId = this_record.split('/')[:2]
        segm = np.apply_along_axis(lambda a: self.table_segm[(worldId, sceneId, a[0], a[1], a[2])], 2, segm)
        segm = segm.astype(np.uint8)
        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in self.imgSize:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = round2nearest_multiple(target_height, self.padding_constant)
            target_width = round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image to float
            img_resized = img_resized.astype(np.float32)
            img_resized = img_resized.transpose((2, 0, 1))
            img_resized = self.img_transform(torch.from_numpy(img_resized))

            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        segm = torch.from_numpy(segm.astype(np.int)).long()

        batch_segms = torch.unsqueeze(segm, 0)

        batch_segms = batch_segms - 1  # label from -1 to 149
        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record
        return output

    def __len__(self):
        return self.num_sample


class TestDataset(torchdata.Dataset):
    def __init__(self, opt, max_sample=-1):
        self.root_img = opt.root_dataset
        if opt.test_img in ['train', 'test', 'all', 'benchmark']:
            self.root_img = os.path.join(self.root_img, 'vkitti_1.3.1_rgb')
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        # down sampling rate of segm label
        self.segm_downsampling_rate = opt.segm_downsampling_rate

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229, 0.224, 0.225])])

        if opt.test_img in ['train', 'test', 'all']:
            self.list_sample = get_lists(opt.test_img)
        elif opt.test_img == 'benchmark':
            self.list_sample = []
            benchmark_json = json.load(open(opt.benchmark_json))
            for pair in benchmark_json[:len(benchmark_json) // 2]:
                world, topic, source = pair['world'], pair['topic'], pair['source']
                self.list_sample += [os.path.join(world, topic, source + '.png')]
        else:
            self.list_sample = [opt.test_img]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_img, this_record)
        img = imread(image_path, mode='RGB')
        img = img[:, :, ::-1]  # BGR to RGB!!!
        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in self.imgSize:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = round2nearest_multiple(target_height, self.padding_constant)
            target_width = round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image to float
            img_resized = img_resized.astype(np.float32)
            img_resized = img_resized.transpose((2, 0, 1))
            img_resized = self.img_transform(torch.from_numpy(img_resized))

            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['info'] = this_record
        return output

    def __len__(self):
        return self.num_sample
