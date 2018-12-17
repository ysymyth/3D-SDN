import json
import matplotlib.cm
import numpy as np
import os
import pandas as pd
import PIL.Image
import PIL.ImageDraw
import torch
import torch.nn.functional as F
import torchvision
import random
import scipy.ndimage
import scipy.special

from derender3d import TargetType


class Transforms(object):
    pad = torchvision.transforms.functional.pad
    crop = torchvision.transforms.functional.crop
    resize = torchvision.transforms.functional.resize
    normalize = torchvision.transforms.functional.normalize
    to_tensor = torchvision.transforms.functional.to_tensor
    to_pil_image = torchvision.transforms.functional.to_pil_image
    color_jitter = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

    # torch.Tensor -> torch.Tensor
    @staticmethod
    def pad_like(image, _image, mode='constant', value=0):
        pad_size_2 = _image.shape[2] - image.shape[2]
        pad_size_3 = _image.shape[3] - image.shape[3]

        return F.pad(image, (pad_size_3 // 2, pad_size_3 // 2, pad_size_2 // 2, pad_size_2 // 2), mode=mode, value=value)

    # list -> list
    @staticmethod
    def roi_jitter(roi, ratio=0.1):
        droi_y = int(ratio * (roi[2] - roi[0]))
        droi_x = int(ratio * (roi[3] - roi[1]))

        return [
            roi[0] + random.randint(- droi_y, droi_y),
            roi[1] + random.randint(- droi_x, droi_x),
            roi[2] + random.randint(- droi_y, droi_y),
            roi[3] + random.randint(- droi_x, droi_x),
        ]

    # PIL.Image -> PIL.Image
    @staticmethod
    def crop_square(image, roi, fill):
        h = roi[2] - roi[0]
        w = roi[3] - roi[1]

        s = max(h, w)
        dh = (s - h) // 2
        dw = (s - w) // 2

        padding = [
            - min(0, roi[1] - dw),
            - min(0, roi[0] - dh),
            max(0, roi[3] + dw - image.width),
            max(0, roi[2] + dh - image.height),
        ]

        _l = roi[1] - dw + padding[0]
        _t = roi[0] - dh + padding[1]

        image = Transforms.pad(image, padding=tuple(padding), fill=fill)
        image = Transforms.crop(image, _t, _l, s, s)

        return image

    # np.ndarray -> np.ndarray
    @staticmethod
    def scene_to_mask(image_scene, code):
        return np.all(image_scene == code, axis=2, keepdims=True).astype(np.float32)

    # np.ndarray -> np.ndarray
    @staticmethod
    def depth_to_normal(image_depth):
        delta_v = scipy.ndimage.correlate1d(image_depth, weights=[-0.5, 0, 0.5], axis=0, mode='nearest')
        delta_u = scipy.ndimage.correlate1d(image_depth, weights=[-0.5, 0, 0.5], axis=1, mode='nearest')

        image_normal = np.stack([
            delta_u,
            - delta_v,
            np.ones_like(image_depth),
        ], axis=2).astype(np.float32)
        image_normal /= np.sqrt(np.sum(np.square(image_normal), axis=2, keepdims=True))

        return image_normal

    # np.ndarray -> list
    @staticmethod
    def mask_to_roi(image_mask):
        index_row = np.where(np.any(image_mask, axis=0))[0]
        index_col = np.where(np.any(image_mask, axis=1))[0]
        return [
            int(index_col[0]),
            int(index_row[0]),
            int(index_col[-1] + 1),
            int(index_row[-1] + 1),
        ]

    # np.ndarray -> np.ndarray
    @staticmethod
    def map_to_cm(image_map):
        image_map = image_map / np.max(image_map)
        image_cm = matplotlib.cm.jet(image_map).astype(np.float32)
        image_cm[image_map == 0, :3] = 1.0

        return image_cm

    # np.ndarray -> PIL.Image
    @staticmethod
    def visualize(image_rgb, image_map, rois, interests=None, alpha=0.5):
        image_map_pil = Transforms.to_pil_image(np.uint8(np.transpose(image_map, (1, 2, 0))))

        image_cm = Transforms.map_to_cm(image_map[0])
        image_cm[..., 3] = alpha * (image_map[0] > 0)
        image_cm_pil = Transforms.to_pil_image(np.uint8(image_cm * 255))

        image_rgb_pil = Transforms.to_pil_image(image_rgb).convert(mode='RGBA')
        image_rgb_pil.paste(image_cm_pil.convert('RGB'), (0, 0), image_cm_pil)

        draw = PIL.ImageDraw.Draw(image_rgb_pil)
        for num in range(len(rois)):
            roi = rois[num]

            if (interests is not None) and interests[num]:
                color = 'green'
            else:
                color = 'red'

            draw.rectangle([roi[1], roi[0], roi[3], roi[2]], outline=color)

        return (image_map_pil, image_rgb_pil)


class BaseDataset(torch.utils.data.dataset.Dataset):
    def transform_ignore(self, image_ignore, roi):
        image_ignore_pil = Transforms.to_pil_image(np.uint8(image_ignore * 255))

        ignore_pil = Transforms.crop_square(image_ignore_pil, roi, fill=255)
        ignore_pil = Transforms.resize(ignore_pil, (256, 256))
        ignore = Transforms.to_tensor(ignore_pil)
        return ignore

    def transform_mask(self, image_mask, roi):
        image_mask_pil = Transforms.to_pil_image(np.uint8(image_mask * 255))

        mask_pil = Transforms.crop_square(image_mask_pil, roi, fill=0)
        mask_pil = Transforms.resize(mask_pil, (256, 256))
        mask = Transforms.to_tensor(mask_pil)
        return mask

    def transform_rgb(self, image_rgb, roi):
        image_rgb_pil = Transforms.to_pil_image(image_rgb)

        rgb_pil = Transforms.crop_square(image_rgb_pil, roi, fill=(127, 127, 127))
        if self.is_train:
            rgb_pil = Transforms.color_jitter(rgb_pil)
        rgb_pil = Transforms.resize(rgb_pil, (224, 224))
        rgb = Transforms.to_tensor(rgb_pil)

        rgb = Transforms.normalize(
            rgb,
            mean=self.mean,
            std=self.std,
        )

        return rgb


class HybridDataset(torch.utils.data.dataset.ConcatDataset):
    def __init__(self, datasets, weights=None):
        super(HybridDataset, self).__init__(datasets)

        if weights is None:
            weights = [1.0] * len(datasets)

        self.weights = weights

    def get_weights(self):
        weights = np.concatenate([
            weight * np.ones(len(dataset)) / len(dataset)
            for (dataset, weight) in zip(self.datasets, self.weights)
        ], axis=0)

        return weights


class VKitti(BaseDataset):
    root_dir = os.getenv('VKITTI_ROOT_DIR')

    worlds = ['0001', '0002', '0006', '0018', '0020']
    train_frames = [range(0, 356), range(0, 185), range(69, 270), range(0, 270), range(167, 837)]
    test_frames = [range(356, 447), range(185, 233), range(0, 69), range(270, 339), range(0, 167)]
    topics = ['15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']

    motgt_df = None
    scenegt_df = None

    mean = [0.5, 0.5, 0.5]
    std = [0.25, 0.25, 0.25]

    class Camera:
        width = 1242
        height = 375

        focal = 725.0
        u0 = 620.5
        v0 = 187.0

    @staticmethod
    def read_scene(world, topic, frame):
        path = os.path.join(VKitti.root_dir, 'vkitti_1.3.1_scenegt', world, topic, '{:05d}.png'.format(frame))
        image_scene_pil = PIL.Image.open(path)
        return np.asarray(image_scene_pil)

    @staticmethod
    def read_rgb(world, topic, frame):
        path = os.path.join(VKitti.root_dir, 'vkitti_1.3.1_rgb', world, topic, '{:05d}.png'.format(frame))
        image_rgb_pil = PIL.Image.open(path)
        return np.asarray(image_rgb_pil)

    @staticmethod
    def read_depth(world, topic, frame):
        path = os.path.join(VKitti.root_dir, 'vkitti_1.3.1_depthgt', world, topic, '{:05d}.png'.format(frame))
        image_depth_pil = PIL.Image.open(path)
        image_depth = np.asarray(image_depth_pil).astype(np.float32) / 100 * VKitti.Camera.focal

        return image_depth

    @staticmethod
    def _read_motgt(debug=False):
        motgt_dfs = []
        for world in VKitti.worlds:
            for topic in VKitti.topics:
                motgt_path = os.path.join(VKitti.root_dir, 'vkitti_1.3.1_motgt', '{:s}_{:s}.txt'.format(world, topic))
                if not os.path.isfile(motgt_path):
                    continue

                print('Reading {:s}'.format(motgt_path))

                df = pd.read_csv(motgt_path, sep=' ', index_col=False)
                df['world'] = world
                df['topic'] = topic

                motgt_dfs.append(df)

        if motgt_dfs:
            VKitti.motgt_df = pd.concat(motgt_dfs).set_index(['world', 'topic', 'frame'])
        else:
            VKitti.motgt_df = pd.DataFrame([], index=['world', 'topic', 'frame'])

    @staticmethod
    def _read_scenegt(debug=False):
        scenegt_dfs = []
        for world in VKitti.worlds:
            for topic in VKitti.topics:
                scenegt_path = os.path.join(VKitti.root_dir, 'vkitti_1.3.1_scenegt', '{:s}_{:s}_scenegt_rgb_encoding.txt'.format(world, topic))
                if not os.path.isfile(scenegt_path):
                    continue

                print('Reading {:s}'.format(scenegt_path))

                df = pd.read_csv(scenegt_path, sep=' ', index_col=False)
                df['world'] = world
                df['topic'] = topic

                scenegt_dfs.append(df)

        if scenegt_dfs:
            VKitti.scenegt_df = pd.concat(scenegt_dfs).set_index(['world', 'topic', 'Category(:id)'])
        else:
            VKitti.scenegt_df = pd.DataFrame([], index=['world', 'topic', 'Category(:id)'])

    def __init__(self, is_train=False, is_evaluate=False, debug=False):
        self.is_train = is_train
        self.is_evaluate = is_evaluate

        if VKitti.motgt_df is None:
            VKitti._read_motgt(debug=debug)

        if VKitti.scenegt_df is None:
            VKitti._read_scenegt(debug=debug)

        if VKitti.motgt_df.size == 0:
            self.df = None
            return

        subsample = []
        for (num_world, world) in enumerate(VKitti.worlds):
            if is_train:
                _frames = VKitti.train_frames[num_world]
            else:
                _frames = VKitti.test_frames[num_world]

            for topic in VKitti.topics:
                frames = VKitti.motgt_df.loc[(world, topic)].index.unique()
                for frame in frames:
                    if frame in _frames:
                        subsample.append((
                            world,
                            topic,
                            frame,
                        ))

        self.df = VKitti.motgt_df.loc[subsample]

        if not is_evaluate:
            roi = np.stack([
                self.df.t,
                self.df.l,
                self.df.b,
                self.df.r,
            ], axis=1)
            droi_y = roi[:, 2] - roi[:, 0]
            droi_x = roi[:, 3] - roi[:, 1]
            sel = (
                (droi_y * droi_x > 16 * 16) &
                (self.df.truncr < 0.7) &
                (self.df.occupr > 0.3)
            )

            self.df = self.df[sel]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        (world, topic, frame) = item.name

        motgt_df = VKitti.motgt_df.loc[(world, topic, frame)]
        scenegt_df = VKitti.scenegt_df.loc[(world, topic)]

        image_scene = VKitti.read_scene(world, topic, frame)
        if not self.is_evaluate:
            image_rgb = VKitti.read_rgb(world, topic, frame)

        name = item.orig_label + ':' + item.tid.astype(np.str)
        code = scenegt_df.loc[name].values
        image_mask = np.all(image_scene[:, :, None, :] == code, axis=3)

        roi = Transforms.mask_to_roi(image_mask)
        if self.is_train:
            roi = Transforms.roi_jitter(roi)

        roi_norm = [
            (roi[0] - VKitti.Camera.v0) / VKitti.Camera.focal,
            (roi[1] - VKitti.Camera.u0) / VKitti.Camera.focal,
            (roi[2] - VKitti.Camera.v0) / VKitti.Camera.focal,
            (roi[3] - VKitti.Camera.u0) / VKitti.Camera.focal,
        ]
        mroi_norm = [
            (roi_norm[2] + roi_norm[0]) / 2.0,
            (roi_norm[3] + roi_norm[1]) / 2.0,
        ]
        droi_norm = [
            (roi_norm[2] - roi_norm[0]),
            (roi_norm[3] - roi_norm[1]),
        ]

        theta = [- item.ry]
        rotation = [np.cos(item.ry / 2), 0, - np.sin(item.ry / 2), 0]
        scale = [item.l3d, item.h3d, 1.2206 * item.w3d]
        xyz = [item.x3d, - (item.y3d - item.h3d / 2), - item.z3d]

        translation2d = [
            (xyz[1] / xyz[2] - mroi_norm[0]) / droi_norm[0],
            (- xyz[0] / xyz[2] - mroi_norm[1]) / droi_norm[1],
        ]
        translation2d = np.clip(translation2d, -6, 6)
        log_scale = np.log(scale)

        depth = np.sum(np.square(xyz), axis=0, keepdims=True)
        log_depth = (
            np.log(depth) +
            np.log(droi_norm[0]) +
            np.log(droi_norm[1])
        )

        xyzs = np.stack([motgt_df.x3d, - (motgt_df.y3d - motgt_df.h3d / 2), - motgt_df.z3d], axis=1)
        depths = np.sum(xyzs ** 2, axis=1)

        names = motgt_df.orig_label + ':' + motgt_df.tid.astype(np.str)
        codes = scenegt_df.loc[names.values].values
        image_masks = np.all(image_scene[:, :, None, :] == codes, axis=3)
        image_ignore = np.sum(image_masks * (depths < depth), axis=2, keepdims=True)

        res = {
            'targets': TargetType.pretrain | TargetType.finetune,
            'image_masks': Transforms.to_tensor(255 * image_mask),
            'image_ignores': Transforms.to_tensor(255 * image_ignore),
            'widths': np.float32([VKitti.Camera.width]),
            'heights': np.float32([VKitti.Camera.height]),
            'focals': np.float32([VKitti.Camera.focal]),
            'u0s': np.float32([VKitti.Camera.u0]),
            'v0s': np.float32([VKitti.Camera.v0]),
            'rois': np.float32(roi),
            'roi_norms': np.float32(roi_norm),
            'thetas': np.float32(theta),
            'rotations': np.float32(rotation),
            'translations': np.float32(xyz),
            'translation2ds': np.float32(translation2d),
            'scales': np.float32(scale),
            'log_scales': np.float32(log_scale),
            'log_depths': np.float32(log_depth),
        }

        if not self.is_evaluate:
            res.update({
                'images': self.transform_rgb(image_rgb, roi),
                'masks': self.transform_mask(image_mask, roi),
                'ignores': self.transform_ignore(image_ignore, roi),
            })

        return res


class KittiBaseDataset(BaseDataset):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    class Camera:
        focal = 725.0
        u0 = 610.0
        v0 = 185.0


class KittiObject(KittiBaseDataset):
    root_dir = os.getenv('KITTI_OBJECT_ROOT_DIR')

    train_frames = range(0, 6733)
    validation_frames = range(6733, 7481)
    debug_train_frames = range(0, 10)
    debug_validation_frames = range(10, 20)
    test_frames = range(0, 7518)

    motgt_names = [
        'type',
        'truncated',
        'occluded',
        'alpha',
        'left', 'top', 'right', 'bottom',
        'h', 'w', 'l',
        'x', 'y', 'z',
        'ry',
        'score',
    ]

    motgt_df = None
    camera_df = None

    @staticmethod
    def read_rgb(region, frame):
        if (region == 'training') or (region == 'validation'):
            name = 'training'

        path = os.path.join(KittiObject.root_dir, name, 'image_2', '{:06d}.png'.format(frame))
        image_rgb_pil = PIL.Image.open(path)
        return np.asarray(image_rgb_pil)

    @staticmethod
    def _read_motgt(debug=False):
        motgt_dfs = []

        for region in ['training', 'validation']:
            if region == 'training':
                if debug:
                    frames = KittiObject.debug_train_frames
                else:
                    frames = KittiObject.train_frames
            elif region == 'validation':
                if debug:
                    frames = KittiObject.debug_validation_frames
                else:
                    frames = KittiObject.validation_frames

            for frame in frames:
                motgt_path = os.path.join(KittiObject.root_dir, 'training', 'label_2', '{:06d}.txt'.format(frame))
                print('Reading {:s}'.format(motgt_path))

                df = pd.read_csv(motgt_path, sep=' ', names=KittiObject.motgt_names, header=None)
                df['region'] = region
                df['frame'] = frame

                motgt_dfs.append(df)

        KittiObject.motgt_df = pd.concat(motgt_dfs).set_index(['region', 'frame'])

    @staticmethod
    def _read_camera(debug=False):
        camera_dfs = []

        for region in ['training', 'validation']:
            if region == 'training':
                if debug:
                    frames = KittiObject.debug_train_frames
                else:
                    frames = KittiObject.train_frames
            elif region == 'validation':
                if debug:
                    frames = KittiObject.debug_validation_frames
                else:
                    frames = KittiObject.validation_frames

            for frame in frames:
                camera_path = os.path.join(KittiObject.root_dir, 'training', 'calib', '{:06d}.txt'.format(frame))
                print('Reading {:s}'.format(camera_path))

                df = pd.read_csv(
                    camera_path,
                    sep=' ',
                    header=None,
                    names=['name', 'focal', 'u0', 'v0'],
                    index_col=0,
                    usecols=[0, 1, 3, 7],
                )
                df = df.loc[['P2:']]
                df['region'] = region
                df['frame'] = frame

                camera_dfs.append(df)

        KittiObject.camera_df = pd.concat(camera_dfs).set_index(['region', 'frame'])

    def __init__(self, is_train=False, debug=False):
        self.is_train = is_train

        if KittiObject.motgt_df is None:
            KittiObject._read_motgt(debug=debug)

        if KittiObject.camera_df is None:
            KittiObject._read_camera(debug=debug)

        if is_train:
            self.df = KittiObject.motgt_df.loc[['training']]
            self.df = self.df[self.df.type.isin(['Car', 'Van', 'Truck'])]
        else:
            self.df = KittiObject.motgt_df.loc[['validation']]
            self.df = self.df[self.df.type.isin(['Car'])]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        (region, frame) = item.name

        camera_item = KittiObject.camera_df.loc[(region, frame)]

        image_rgb = KittiObject.read_rgb(region, frame)

        roi_norm = [
            (item.top - camera_item.v0) / camera_item.focal,
            (item.left - camera_item.u0) / camera_item.focal,
            (item.bottom - camera_item.v0) / camera_item.focal,
            (item.right - camera_item.u0) / camera_item.focal,
        ]
        mroi_norm = [
            (roi_norm[2] + roi_norm[0]) / 2.0,
            (roi_norm[3] + roi_norm[1]) / 2.0,
        ]
        droi_norm = [
            (roi_norm[2] - roi_norm[0]),
            (roi_norm[3] - roi_norm[1]),
        ]

        image_rgb = self.transform_rgb(image_rgb, [
            int(item.top),
            int(item.left),
            int(item.bottom),
            int(item.right),
        ])

        focal = [camera_item.focal]
        theta = [- item.ry]
        scale = [item.l, item.h, item.w]
        xyz = [item.x, - (item.y - item.h / 2), - item.z]

        translation2d = [
            (xyz[1] / xyz[2] - mroi_norm[0]) / droi_norm[0],
            (- xyz[0] / xyz[2] - mroi_norm[1]) / droi_norm[1],
        ]
        translation2d = np.clip(translation2d, -6, 6)
        log_scale = np.log(scale)
        depth = np.sum(np.square(xyz), axis=0, keepdims=True)
        log_depth = (
            np.log(depth) +
            np.log(droi_norm[0]) +
            np.log(droi_norm[1])
        )

        return {
            'targets': TargetType.pretrain,
            'images': image_rgb,
            'focals': np.float32(focal),
            'roi_norms': np.float32(roi_norm),
            'thetas': np.float32(theta),
            'translation2ds': np.float32(translation2d),
            'log_scales': np.float32(log_scale),
            'log_depths': np.float32(log_depth),
        }


class KittiSemantics(KittiBaseDataset):
    root_dir = os.getenv('KITTI_SEMANTICS_ROOT_DIR')
    cache_dir = os.getenv('KITTI_SEMANTICS_CACHE_DIR')

    train_frames = range(0, 180)
    validation_frames = range(180, 200)
    debug_train_frames = range(0, 10)
    debug_validation_frames = range(10, 20)

    scenegt_df = None

    class Category:
        car = 66

    @staticmethod
    def index2cat(index):
        return index // 100

    @staticmethod
    def read_rgb(region, frame):
        if (region == 'training') or (region == 'validation'):
            name = 'training'

        path = os.path.join(KittiSemantics.root_dir, name, 'image_2', '{:06d}_10.png'.format(frame))
        image_rgb_pil = PIL.Image.open(path)
        return np.asarray(image_rgb_pil)

    @staticmethod
    def read_scene(region, frame):
        if (region == 'training') or (region == 'validation'):
            name = 'training'

        path = os.path.join(KittiSemantics.root_dir, name, 'instance', '{:06d}_10.png'.format(frame))
        image_scene_pil = PIL.Image.open(path)
        return np.asarray(image_scene_pil)

    @staticmethod
    def _read_scenegt(debug=False):
        scenegt_dfs = []

        if not os.path.isdir(KittiSemantics.cache_dir):
            os.makedirs(KittiSemantics.cache_dir)

        for region in ['training', 'validation']:
            if region == 'training':
                if debug:
                    frames = KittiSemantics.debug_train_frames
                else:
                    frames = KittiSemantics.train_frames
            elif region == 'validation':
                if debug:
                    frames = KittiSemantics.debug_validation_frames
                else:
                    frames = KittiSemantics.validation_frames

            for frame in frames:
                json_path = os.path.join(KittiSemantics.cache_dir, '_{:06d}.json'.format(frame))
                print('Reading {:s}'.format(json_path))

                if os.path.isfile(json_path):
                    with open(json_path, 'r') as f:
                        json_objs = json.load(f)
                else:
                    image_scene = KittiSemantics.read_scene(region, frame)
                    obj_indices = np.unique(image_scene)

                    json_objs = []
                    for obj_index in obj_indices:
                        image_mask_array = (image_scene == obj_index)
                        index_row = np.where(np.any(image_mask_array, axis=0))[0]
                        index_col = np.where(np.any(image_mask_array, axis=1))[0]
                        roi = np.asarray([
                            index_col[0],
                            index_row[0],
                            index_col[-1] + 1,
                            index_row[-1] + 1,
                        ])

                        json_objs.append({
                            'obj_index': int(obj_index),
                            'roi': roi.tolist(),
                        })

                    with open(json_path, 'w') as f:
                        json.dump(json_objs, f)

                for json_obj in json_objs:
                    obj_index = json_obj['obj_index']
                    roi = json_obj['roi']

                    if not KittiSemantics.index2cat(obj_index) == KittiSemantics.Category.car:
                        continue

                    scenegt_dfs.append({
                        'region': region,
                        'frame': frame,
                        'obj_index': obj_index,
                        'roi': roi,
                    })

        KittiSemantics.scenegt_df = pd.DataFrame(scenegt_dfs).set_index(['region', 'frame'])

    def __init__(self, is_train=False, debug=False):
        self.is_train = is_train

        if KittiSemantics.scenegt_df is None:
            KittiSemantics._read_scenegt(debug=debug)

        if is_train:
            self.df = KittiSemantics.scenegt_df.loc[['training']]
        else:
            self.df = KittiSemantics.scenegt_df.loc[['validation']]

        roi = np.asarray(self.df.roi.values.tolist())

        droi_y = roi[:, 2] - roi[:, 0]
        droi_x = roi[:, 3] - roi[:, 1]
        sel = (
            (droi_y * droi_x > 32 * 32) *
            (droi_x / droi_y < 4) *
            (droi_y / droi_x < 4)
        )

        self.df = self.df[sel]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        (region, frame) = item.name

        image_scene = KittiSemantics.read_scene(region, frame)
        image_rgb = KittiSemantics.read_rgb(region, frame)

        u0 = (image_rgb.shape[1] - 1) / 2
        v0 = (image_rgb.shape[0] - 1) / 2

        obj_index = item.obj_index
        roi = item.roi
        if self.is_train:
            roi = Transforms.roi_jitter(roi)

        roi_norm = [
            (roi[0] - v0) / KittiSemantics.Camera.focal,
            (roi[1] - u0) / KittiSemantics.Camera.focal,
            (roi[2] - v0) / KittiSemantics.Camera.focal,
            (roi[3] - u0) / KittiSemantics.Camera.focal,
        ]
        focal = [KittiSemantics.Camera.focal]

        image_mask = (image_scene == obj_index)[:, :, None]

        return {
            'targets': TargetType.finetune,
            'images': self.transform_rgb(image_rgb, roi),
            'focals': np.float32(focal),
            'masks': self.transform_mask(image_mask, roi),
            'ignores': torch.zeros(1, 256, 256),
            'roi_norms': np.float32(roi_norm),
        }


class KittiSemanticsHybrid(HybridDataset):
    def __init__(self, is_train, debug=False):
        super(KittiSemanticsHybrid, self).__init__([
            KittiObject(is_train=is_train, debug=debug),
            KittiSemantics(is_train=is_train, debug=debug),
        ])


class CityscapesBaseDataset(BaseDataset):
    root_dir = os.getenv('CITYSCAPES_ROOT_DIR')

    camera_df = None

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    class Camera:
        focal = 2250.0
        u0 = 925.0
        v0 = 460.0

    @staticmethod
    def read_rgb(split, city, seq, frame):
        path = os.path.join(CityscapesBaseDataset.root_dir, 'images', 'leftImg8bit', split, city, '{:s}_{:s}_{:s}_leftImg8bit.png'.format(city, seq, frame))
        image_pil = PIL.Image.open(path)
        return np.asarray(image_pil)

    @staticmethod
    def _read_camera(splits, debug=False):
        camera_dfs = []

        for split in splits:
            split_dir = os.path.join(CityscapesBaseDataset.root_dir, 'camera', split)

            for city in os.listdir(split_dir):
                city_dir = os.path.join(split_dir, city)

                for name in sorted(os.listdir(city_dir)):
                    if not name.endswith('_camera.json'):
                        continue

                    (_, seq, frame, _) = name.split('_')
                    path = os.path.join(city_dir, name)

                    print('Reading {:s}'.format(path))

                    with open(path, 'r') as f:
                        json_objs = json.load(f)

                    camera_dfs.append({
                        'split': split,
                        'city': city,
                        'seq': seq,
                        'frame': frame,
                        'f': json_objs['intrinsic']['fx'],
                        'u0': json_objs['intrinsic']['u0'],
                        'v0': json_objs['intrinsic']['v0'],
                    })

                    if debug:
                        break

        CityscapesBaseDataset.camera_df = pd.DataFrame(camera_dfs).set_index(['split', 'city', 'seq', 'frame'])


class CityscapesSemantics(CityscapesBaseDataset):
    cache_dir = os.getenv('CITYSCAPES_SEMANTICS_CACHE_DIR')

    splits = ['train', 'val']

    scenegt_df = None

    class Category:
        car = 26

    @staticmethod
    def index2cat(index):
        return index // 1000

    @staticmethod
    def read_scene(split, city, seq, frame):
        path = os.path.join(CityscapesBaseDataset.root_dir, 'gtFine', split, city, '{:s}_{:s}_{:s}_gtFine_instanceIds.png'.format(city, seq, frame))
        image_scene_pil = PIL.Image.open(path)
        return np.asarray(image_scene_pil)[..., None]

    @staticmethod
    def read_disparity(split, city, seq, frame):
        path = os.path.join(CityscapesBaseDataset.root_dir, 'disparity', split, city, '{:s}_{:s}_{:s}_disparity.png'.format(city, seq, frame))
        image_disparity_pil = PIL.Image.open(path)
        return np.asarray(image_disparity_pil)[..., None]

    @staticmethod
    def _read_scenegt(debug=False):
        scenege_dfs = []

        if not os.path.isdir(CityscapesSemantics.cache_dir):
            os.makedirs(CityscapesSemantics.cache_dir)

        for split in CityscapesSemantics.splits:
            split_dir = os.path.join(CityscapesBaseDataset.root_dir, 'gtFine', split)

            for city in sorted(os.listdir(split_dir)):
                city_dir = os.path.join(split_dir, city)

                for name in sorted(os.listdir(city_dir)):
                    if not name.endswith('gtFine_instanceIds.png'):
                        continue

                    (seq, frame) = name.split('_')[1:3]
                    json_path = os.path.join(CityscapesSemantics.cache_dir, '{:s}_{:s}_{:s}_gtFine.json'.format(city, seq, frame))

                    print('Reading {:s}'.format(json_path))

                    if os.path.isfile(json_path):
                        with open(json_path, 'r') as f:
                            json_objs = json.load(f)
                    else:
                        image_scene = CityscapesSemantics.read_scene(split, city, seq, frame)
                        obj_indices = np.unique(image_scene)

                        json_objs = []
                        for obj_index in obj_indices:
                            if CityscapesSemantics.index2cat(obj_index) == CityscapesSemantics.Category.car:
                                json_objs.append({'obj_index': int(obj_index)})

                        with open(json_path, 'w') as f:
                            json.dump(json_objs, f)

                    for json_obj in json_objs:
                        obj_index = json_obj['obj_index']

                        scenege_dfs.append({
                            'split': split,
                            'city': city,
                            'seq': seq,
                            'frame': frame,
                            'obj_index': obj_index,
                        })

        CityscapesSemantics.scenegt_df = pd.DataFrame(scenege_dfs).set_index(['split', 'city', 'seq', 'frame'])

    def __init__(self, is_train=False, debug=False):
        self.is_train = is_train

        if CityscapesBaseDataset.camera_df is None:
            CityscapesBaseDataset._read_camera(CityscapesSemantics.splits, debug=debug)

        if CityscapesSemantics.scenegt_df is None:
            CityscapesSemantics._read_scenegt(debug=debug)

        if is_train:
            self.df = CityscapesSemantics.scenegt_df.loc[['train']]
        else:
            self.df = CityscapesSemantics.scenegt_df.loc[['val']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        (split, city, seq, frame) = item.name

        image_rgb = CityscapesBaseDataset.read_rgb(split, city, seq, frame)
        image_scene = CityscapesSemantics.read_scene(split, city, seq, frame)
        image_disparity = CityscapesSemantics.read_disparity(split, city, seq, frame)

        image_mask = (image_scene == item.obj_index)
        roi = Transforms.mask_to_roi(image_mask)
        if self.is_train:
            roi = Transforms.roi_jitter(roi)

        roi_norm = [
            (roi[0] - CityscapesBaseDataset.Camera.v0) / CityscapesBaseDataset.Camera.focal,
            (roi[1] - CityscapesBaseDataset.Camera.u0) / CityscapesBaseDataset.Camera.focal,
            (roi[2] - CityscapesBaseDataset.Camera.v0) / CityscapesBaseDataset.Camera.focal,
            (roi[3] - CityscapesBaseDataset.Camera.u0) / CityscapesBaseDataset.Camera.focal,
        ]

        disparity = image_disparity[image_mask]
        disparity = disparity[disparity != 0]
        if disparity.size:
            disparity = np.percentile(disparity, 95)
        else:
            disparity = 0
        image_ignore = (image_disparity > disparity)

        res = {
            'targets': TargetType.finetune,
            'images': self.transform_rgb(image_rgb, roi),
            'masks': self.transform_mask(image_mask, roi),
            'ignores': self.transform_ignore(image_ignore, roi),
            'widths': np.float32([image_rgb.shape[1]]),
            'heights': np.float32([image_rgb.shape[0]]),
            'focals': np.float32([CityscapesBaseDataset.Camera.focal]),
            'u0s': np.float32([CityscapesBaseDataset.Camera.u0]),
            'v0s': np.float32([CityscapesBaseDataset.Camera.v0]),
            'rois': np.float32(roi),
            'roi_norms': np.float32(roi_norm),
        }
        return res


class CityscapesMaskRCNN(CityscapesBaseDataset):
    root_dir = os.getenv('CITYSCAPES_MASKRCNN_ROOT_DIR')
    cache_dir = os.getenv('CITYSCAPES_MASKRCNN_CACHE_DIR')

    splits = ['train', 'train_extra', 'test']

    scenegt_df = None

    @staticmethod
    def read_scene(split, city, seq, frame):
        path = os.path.join(CityscapesMaskRCNN.root_dir, split, city, '{:s}_{:s}_{:s}_leftImg8bit.png'.format(city, seq, frame))
        image_pil = PIL.Image.open(path)
        return np.asarray(image_pil)

    @staticmethod
    def _read_scenegt(debug=False):
        scenegt_dfs = []

        if not os.path.isdir(CityscapesMaskRCNN.cache_dir):
            os.makedirs(CityscapesMaskRCNN.cache_dir)

        for split in CityscapesMaskRCNN.splits:
            split_dir = os.path.join(CityscapesMaskRCNN.root_dir, split)

            for city in sorted(os.listdir(split_dir)):
                city_dir = os.path.join(split_dir, city)

                for name in sorted(os.listdir(city_dir)):
                    if not name.endswith('_leftImg8bit.png'):
                        continue

                    (_, seq, frame, _) = name.split('_')
                    json_path = os.path.join(CityscapesMaskRCNN.cache_dir, '_{:s}_{:s}_{:s}_gtFine.json'.format(city, seq, frame))

                    print('Reading {:s}'.format(json_path))

                    if os.path.isfile(json_path):
                        with open(json_path, 'r') as f:
                            json_objs = json.load(f)
                    else:
                        image_scene = CityscapesMaskRCNN.read_scene(split, city, seq, frame)
                        obj_indices = np.unique(image_scene)

                        json_objs = []
                        for obj_index in obj_indices:
                            image_mask = (image_scene == obj_index)[:, :, None]
                            roi = Transforms.mask_to_roi(image_mask)

                            json_objs.append({
                                'obj_index': int(obj_index),
                                'roi': roi,
                            })

                        with open(json_path, 'w') as f:
                            json.dump(json_objs, f)

                    for json_obj in json_objs:
                        obj_index = json_obj['obj_index']
                        roi = json_obj['roi']

                        if obj_index == 0:
                            continue

                        scenegt_dfs.append({
                            'split': split,
                            'city': city,
                            'seq': seq,
                            'frame': frame,
                            'obj_index': obj_index,
                            'roi': roi,
                        })

                    if debug:
                        break

        CityscapesMaskRCNN.scenegt_df = pd.DataFrame(scenegt_dfs).set_index(['split', 'city', 'seq', 'frame'])

    def __init__(self, is_train=False, debug=False):
        self.is_train = is_train

        if CityscapesMaskRCNN.scenegt_df is None:
            CityscapesMaskRCNN._read_scenegt(debug=debug)

        if is_train:
            self.df = CityscapesMaskRCNN.scenegt_df.loc[['train']]
        else:
            self.df = CityscapesMaskRCNN.scenegt_df.loc[['test']]

        roi = np.asarray(self.df.roi.values.tolist())

        droi_y = roi[:, 2] - roi[:, 0]
        droi_x = roi[:, 3] - roi[:, 1]
        sel = (
            (droi_y * droi_x > 32 * 32) *
            (droi_x / droi_y < 4) *
            (droi_y / droi_x < 4)
        )

        self.df = self.df[sel]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        (split, city, seq, frame) = item.name

        image_scene = CityscapesMaskRCNN.read_scene(split, city, seq, frame)
        image_rgb = CityscapesMaskRCNN.read_rgb(split, city, seq, frame)

        camera_item = CityscapesMaskRCNN.camera_df.loc[(split, city, seq, frame)]

        obj_index = item.obj_index
        roi = item.roi
        if self.is_train:
            roi = Transforms.roi_jitter(roi)

        roi_norm = [
            (roi[0] - camera_item.v0) / camera_item.f,
            (roi[1] - camera_item.u0) / camera_item.f,
            (roi[2] - camera_item.v0) / camera_item.f,
            (roi[3] - camera_item.u0) / camera_item.f,
        ]
        # focal = [camera_item.f]

        image_mask = (image_scene == obj_index)[:, :, None]

        return {
            'targets': TargetType.finetune,
            'images': self.transform_rgb(image_rgb, roi),
            'widths': np.float32([image_rgb.width]),
            'heights': np.float32([image_rgb.height]),
            'focals': np.float32([camera_item.f]),
            'u0s': np.float32([camera_item.u0]),
            'v0s': np.float32([camera_item.v0]),
            'masks': self.transform_mask(image_mask, roi),
            'ignores': torch.zeros(1, 256, 256),
            'roi_norms': np.float32(roi_norm),
        }


class CityscapesSemanticsHybrid(HybridDataset):
    def __init__(self, is_train, debug=False):
        super(CityscapesSemanticsHybrid, self).__init__(
            datasets=[
                VKitti(is_train=is_train, debug=debug),
                CityscapesSemantics(is_train=is_train, debug=debug),
            ],
            weights=[0.75, 0.25],
        )
