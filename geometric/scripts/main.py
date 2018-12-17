#!/usr/bin/env python

import chainer
import functools
import json
import torch
import torchvision
import numpy as np
import os
import PIL.Image
import sys

from absl import flags
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.nn.parallel import DataParallel

from bulb.net import Net, TrainMixin, TestMixin
from bulb.saver import Saver
from bulb.utils import new_working_dir

from derender3d import TargetType
from derender3d.datasets import Transforms
from derender3d.data_loader import DataLoader
from derender3d.models import Derenderer3d
from derender3d.utils import to_numpy

from maskrcnn.model import MaskRCNN
from maskrcnn.config import Config

flags.DEFINE_string('do', None, 'do')
flags.DEFINE_string('_do', '_test', '_do')
flags.DEFINE_string('input_file', None, 'input_file')
flags.DEFINE_enum('dataset', None, ['vkitti', 'cityscapes'], 'dataset')
flags.DEFINE_enum('mode', None, ['pretrain', 'full', 'finetune', 'extend'], 'mode')
flags.DEFINE_enum('source', 'gt', ['gt', 'maskrcnn'], 'source')
flags.DEFINE_integer('num_opts', 0, 'num_opts')
flags.DEFINE_integer('num_epochs', 256, 'num_epochs')
flags.DEFINE_integer('batch_size', 64, 'batch_size')
flags.DEFINE_integer('num_grids', 4, 'num_grids')
flags.DEFINE_float('mask_weight', 0.1, 'mask_weight')
flags.DEFINE_float('ffd_coeff_reg', 1.0, 'ffd_coeff_reg')
flags.DEFINE_integer('image_size', 256, 'image_size')
flags.DEFINE_integer('render_size', 384, 'render_size')
flags.DEFINE_float('lr', 1e-3, 'lr')
flags.DEFINE_integer('lr_decay_epochs', 16, 'lr_decay_epochs')
flags.DEFINE_float('lr_decay_rate', 0.5, 'lr_decay_rate')
flags.DEFINE_float('weight_decay', 1e-3, 'weight_decay')
flags.DEFINE_integer('summarize_steps', 1, 'summarize_steps')
flags.DEFINE_integer('image_steps', 100, 'image_steps')
flags.DEFINE_integer('save_steps', 5000, 'save_steps')
flags.DEFINE_string('ckpt_dir', None, 'ckpt_dir')
flags.DEFINE_string('maskrcnn_path', None, 'maskrcnn_path')
flags.DEFINE_string('output_dir', None, 'output_dir')
flags.DEFINE_string('edit_json', None, 'edit_json')
flags.DEFINE_string('working_dir_root', './models', 'working_dir_root')
flags.DEFINE_string('name', None, 'name')
flags.DEFINE_integer('num_workers', 8, 'num_workers')
flags.DEFINE_bool('debug', False, 'debug')
FLAGS = flags.FLAGS


class TrainLoader(DataLoader):
    def __init__(self):
        super(TrainLoader, self).__init__(
            dataset=FLAGS.dataset,
            mode=FLAGS.mode,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers,
            is_train=True,
            debug=FLAGS.debug,
        )


class TestLoader(DataLoader):
    def __init__(self):
        super(TestLoader, self).__init__(
            dataset=FLAGS.dataset,
            mode=FLAGS.mode,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers,
            is_train=False,
            debug=FLAGS.debug,
        )


class Model(Derenderer3d):
    def __init__(self):
        super(Model, self).__init__(
            mode=FLAGS.mode,
            image_size=FLAGS.image_size,
            render_size=FLAGS.render_size,
        )


class BaseNet(Net):
    @staticmethod
    def partial(f, m):
        def _f(*args, **kwargs):
            index = torch.nonzero(m)

            if index.numel():
                index = index.squeeze(dim=1)
                v = f(*[arg[index] for arg in args], **kwargs)
                if torch.isnan(v).any():
                    import pdb
                    pdb.set_trace()
                return v
            else:
                return torch.tensor(0.0).cuda()

        return _f

    def step_batch(self):
        _blob = self.model(self.images, self.roi_norms, self.focals)
        self._register_vars(_blob)

        loss_dict = {}
        if FLAGS.mode & TargetType.geometry:
            is_geometry = self.targets & TargetType.pretrain

            mse_loss = BaseNet.partial(F.mse_loss, is_geometry)

            self.theta_deltas = torch.cat([
                torch.cos(self.thetas),
                torch.sin(self.thetas),
            ], dim=1)

            loss_dict.update({
                'theta_delta_loss': mse_loss(self._theta_deltas, self.theta_deltas),
                'translation2d_loss': mse_loss(self._translation2ds, self.translation2ds),
                'scale_loss': mse_loss(self._log_scales, self.log_scales),
                'depth_loss': mse_loss(self._log_depths, self.log_depths),
            })

        if FLAGS.mode & TargetType.reproject:
            is_reproject = self.targets & TargetType.finetune

            mean = BaseNet.partial(torch.mean, is_reproject)
            mse_loss = BaseNet.partial(F.mse_loss, is_reproject)

            self.masks = Transforms.pad_like(self.masks, self._masks)
            self.ignores = Transforms.pad_like(self.ignores, self._masks, mode='replicate')

            mask_losses = (1 - self.ignores) * F.mse_loss(self._masks, self.masks, reduce=False)
            mask_losses = FLAGS.mask_weight * mask_losses.mean(dim=3).mean(dim=2).mean(dim=1)

            loss_dict.update({
                'class_reward': mean(self._class_log_probs * mask_losses.detach()),
                'mask_loss': mean(mask_losses),
                'ffd_coeff_reg': FLAGS.ffd_coeff_reg * torch.mean(self._ffd_coeffs ** 2),
            })

        return loss_dict

    def post_batch(self):
        super(BaseNet, self).post_batch()

        if FLAGS.mode & TargetType.reproject:
            if (self.num_step % FLAGS.image_steps == 0):
                masks = torchvision.utils.make_grid(self.masks)
                self.writer.add_image('{:s}/mask'.format(self.name), masks, self.num_step)

                _masks = torchvision.utils.make_grid(self._masks)
                self.writer.add_image('{:s}/_mask'.format(self.name), _masks, self.num_step)

                ignores = torchvision.utils.make_grid(self.ignores)
                self.writer.add_image('{:s}/ignore'.format(self.name), ignores, self.num_step)


class TrainNet(BaseNet, TrainMixin):
    pass


class TestNet(BaseNet, TestMixin):
    pass


def train():
    working_dir = new_working_dir(FLAGS.working_dir_root, FLAGS.name)

    model = DataParallel(Model()).cuda()
    writer = SummaryWriter(log_dir=working_dir)
    saver = Saver(working_dir=working_dir)
    saver.save_meta(FLAGS.flag_values_dict())

    train_net = TrainNet(
        optimizer=torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay),
        model=model,
        writer=writer,
        data_loader=TrainLoader(),
        lr_decay_epochs=FLAGS.lr_decay_epochs,
        lr_decay_rate=FLAGS.lr_decay_rate,
        summarize_steps=FLAGS.summarize_steps,
        save_steps=FLAGS.save_steps,
        saver=saver,
    )
    test_net = TestNet(
        model=model,
        writer=writer,
        data_loader=TestLoader(),
    )

    if FLAGS.ckpt_dir is not None:
        train_net.load(ckpt_dir=FLAGS.ckpt_dir)

    for num_epoch in range(FLAGS.num_epochs):
        train_net.step_epoch()
        test_net.step_epoch(num_step=train_net.num_step)

    train_net.save()
    writer.close()


def _test_2d(
    dataset,
    image_dir,
    name,
    image_rgb,
    class_ids,
    image_masks,
    rois,
    operations=None,
    use_ry=False,
):

    (height, width, _) = image_rgb.shape

    num_objs = len(class_ids)
    class_ids = torch.tensor(class_ids, dtype=torch.int32, requires_grad=False).cuda()
    image_masks = torch.tensor(image_masks, dtype=torch.float32, requires_grad=False).cuda()
    rois = torch.tensor(rois, dtype=torch.int32, requires_grad=False).cuda()

    interests = torch.ones(num_objs).byte().cuda()

    image_instance_map = torch.zeros(1, height, width).cuda()
    for num_obj in range(num_objs):
        image_instance_map = (1 - image_masks[num_obj]) * image_instance_map + image_masks[num_obj] * (1 + num_obj)

    (image_instance_map_pil, image_vis_pil) = Transforms.visualize(image_rgb, to_numpy(image_instance_map), to_numpy(rois), to_numpy(interests))
    image_instance_map_pil.save(os.path.join(image_dir, '{:s}-ref.png'.format(name)))
    image_vis_pil.save(os.path.join(image_dir, '{:s}-ref-visualize.png'.format(name)))

    mrois = torch.stack([
        rois[:, 2] + rois[:, 0],
        rois[:, 3] + rois[:, 1],
    ], dim=1).float() / 2.0
    drois = torch.stack([
        rois[:, 2] - rois[:, 0],
        rois[:, 3] - rois[:, 1],
    ], dim=1).float()

    _mrois = mrois.clone()
    _drois = drois.clone()

    if operations is not None and operations:
        _mroi_ops = torch.tensor([[
            float(operation['from']['v']),
            float(operation['from']['u']),
        ] for operation in operations]).cuda()

        _mroi_diffs = torch.sum((_mrois[:, None, :] - _mroi_ops[None, :, :]) ** 2, dim=2)
        if len(_mrois) < len(_mroi_ops):
            index_ops = torch.argmin(_mroi_diffs, dim=1)
            indices = [(index_obj, index_op) for (index_obj, index_op) in enumerate(index_ops)]
        else:
            index_objs = torch.argmin(_mroi_diffs, dim=0)
            indices = [(index_obj, index_op) for (index_op, index_obj) in enumerate(index_objs)]

        for (index_obj, index_op) in indices:
            operation = operations[index_op]
            print('Object #{:d} and operation {!s}'.format(index_obj, operation))

            u = float(operation['from']['u'])
            v = float(operation['from']['v'])

            if operation['type'] == 'delete':
                interests[index_obj] = torch.tensor(0.0).type(torch.ByteTensor).cuda()

            elif operation['type'] == 'modify':
                _u = float(operation['to']['u'])
                _v = float(operation['to']['v'])
                zoom = float(operation['zoom'])
                ry = float(operation['ry'])

                _mrois[index_obj] = _mrois[index_obj] + torch.tensor([_v - v, _u - u]).cuda()

                if use_ry:
                    _drois[index_obj] = torch.tensor([zoom * _drois[index_obj, 0], zoom * float(np.cos(ry)) * _drois[index_obj, 1]]).cuda()
                else:
                    _drois[index_obj] = zoom * _drois[index_obj]

    json_obj = {}
    _image_instance_map = torch.zeros(1, height, width).cuda()
    for index_obj in range(num_objs):
        if to_numpy(interests[index_obj]):  # interesting obj
            json_obj[index_obj + 1] = {
                'class_id': int(class_ids[index_obj]),
            }

            _mask = image_masks[index_obj, :, rois[index_obj, 0]:rois[index_obj, 2], rois[index_obj, 1]:rois[index_obj, 3]]
            _mask_pil = Transforms.to_pil_image(_mask.detach().cpu())
            _mask_pil = Transforms.resize(_mask_pil, (int(_drois[index_obj, 0]), int(_drois[index_obj, 1])))
            _image_mask_pil = PIL.Image.new(mode='L', size=(width, height))
            _image_mask_pil.paste(_mask_pil, box=(
                int(_mrois[index_obj, 1] - _drois[index_obj, 1] / 2),
                int(_mrois[index_obj, 0] - _drois[index_obj, 0] / 2),
            ))
            _image_mask = Transforms.to_tensor(_image_mask_pil).cuda()
            _image_mask = torch.round(_image_mask)

            _image_instance_map = (1 - _image_mask) * _image_instance_map + _image_mask * (1 + index_obj)

    with open(os.path.join(image_dir, '{:s}.json'.format(name)), 'w') as f:
        json.dump(json_obj, f, indent=4)

    (_image_instance_map_pil, _image_vis_pil) = Transforms.visualize(image_rgb, to_numpy(_image_instance_map), to_numpy(rois), to_numpy(interests))
    _image_instance_map_pil.save(os.path.join(image_dir, '{:s}.png'.format(name)))
    _image_vis_pil.save(os.path.join(image_dir, '{:s}-visualize.png'.format(name)))


_test_2d_plus = functools.partial(_test_2d, use_ry=True)


def _test(
    dataset,
    model,
    image_dir,
    name,
    image_rgb,
    class_ids,
    image_masks,
    image_ignores,
    rois,
    metas=None,
    operations=None,
    all_interested=False,
):

    (height, width, _) = image_rgb.shape

    num_objs = len(class_ids)
    class_ids = torch.tensor(class_ids, dtype=torch.int32, requires_grad=False).cuda()
    image_masks = torch.tensor(image_masks, dtype=torch.float32, requires_grad=False).cuda()
    rois = torch.tensor(rois, dtype=torch.int32, requires_grad=False).cuda()

    mask_sums = image_masks.sum(dim=3).sum(dim=2).squeeze(dim=1)
    class_sels = torch.tensor([1, 2], dtype=torch.int32, requires_grad=False).cuda()

    if all_interested:
        interests = torch.ones(num_objs).byte().cuda()
    else:
        interests = (
            torch.sum(class_ids[:, None] == class_sels, dim=1).byte() * (mask_sums > 16 * 16)
        )

    image_instance_map = torch.zeros(1, height, width).cuda()
    for num_obj in range(num_objs):
        image_instance_map = (1 - image_masks[num_obj]) * image_instance_map + image_masks[num_obj] * (1 + num_obj)

    (image_instance_map_pil, image_vis_pil) = Transforms.visualize(image_rgb, to_numpy(image_instance_map), to_numpy(rois), to_numpy(interests))
    image_instance_map_pil.save(os.path.join(image_dir, '{:s}-ref.png'.format(name)))
    image_vis_pil.save(os.path.join(image_dir, '{:s}-ref-visualize.png'.format(name)))

    rgbs = []
    for roi in rois:
        rgbs.append(dataset.transform_rgb(image_rgb, to_numpy(roi)))
    rgbs = torch.stack(rgbs, dim=0).cuda()

    masks = []
    for (image_mask, roi) in zip(image_masks, rois):
        masks.append(dataset.transform_mask(np.transpose(to_numpy(image_mask), (1, 2, 0)), to_numpy(roi)))
    masks = torch.stack(masks, dim=0).cuda()

    roi_norms = (rois.float() - torch.tensor([
        dataset.Camera.v0,
        dataset.Camera.u0,
        dataset.Camera.v0,
        dataset.Camera.u0,
    ], requires_grad=False).cuda()) / dataset.Camera.focal

    focals = dataset.Camera.focal * torch.ones(num_objs, 1, requires_grad=False).cuda()

    _zoom_tos = FLAGS.render_size / (2.0 * dataset.Camera.focal) * torch.ones(num_objs, 1).cuda()

    _mroi_norms = torch.stack([
        roi_norms[:, 2] + roi_norms[:, 0],
        roi_norms[:, 3] + roi_norms[:, 1],
    ], dim=1) / 2.0
    _droi_norms = torch.stack([
        roi_norms[:, 2] - roi_norms[:, 0],
        roi_norms[:, 3] - roi_norms[:, 1],
    ], dim=1)

    _blob = {
        '_roi_norms': roi_norms,
        '_mroi_norms': _mroi_norms,
        '_droi_norms': _droi_norms,
        '_focals': focals,
    }

    _blob_derendered = model.module.derenderer(rgbs, _mroi_norms, _droi_norms)
    _blob.update(_blob_derendered)

    if FLAGS.num_opts:
        if image_ignores is None:
            _depths = _blob['_log_depths'] - torch.sum(torch.log(_droi_norms), dim=1, keepdim=True)
            index = torch.sort(_depths, dim=0)[1].squeeze()
            image_masks_sorted = torch.cat([
                torch.zeros_like(image_masks[0:1]),
                torch.index_select(image_masks, dim=0, index=index),
            ], dim=0)
            image_masks_sorted = image_masks_sorted[:-1]
            image_ignores = torch.clamp(torch.cumsum(image_masks_sorted, dim=0), min=0, max=1)
        else:
            image_ignores = torch.tensor(image_ignores, dtype=torch.float32, requires_grad=False).cuda()

        ignores = []
        for (image_ignore, roi) in zip(image_ignores, rois):
            ignores.append(dataset.transform_ignore(np.transpose(to_numpy(image_ignore), (1, 2, 0)), to_numpy(roi)))
        ignores = torch.stack(ignores, dim=0).cuda()

        model.train()
        model.module._force_no_sample = True

        for (key, value) in _blob_derendered.items():
            value = value.clone().detach()
            _blob[key] = value

        _blob_derendered_optimize = {}
        for key in [
            '_theta_deltas',
            '_translation2ds',
            '_log_scales',
            '_ffd_coeffs',
        ]:
            _blob_derendered_optimize[key] = _blob[key].requires_grad_()

        optimizer = torch.optim.Adam(_blob_derendered_optimize.values(), lr=3e-2)
        for num_opt in range(FLAGS.num_opts):
            optimizer.zero_grad()

            _blob.update(model.module.render(_blob))

            _masks = _blob['_masks']
            masks_padded = Transforms.pad_like(masks, _masks)
            loss = torch.nn.functional.mse_loss(_masks, masks_padded, reduce=False) + 100 * torch.mean(_blob['_ffd_coeffs'] ** 2)
            if image_ignores is not None:
                ignores_padded = Transforms.pad_like(ignores, _masks, mode='replicate')
                loss = loss * (1 - ignores_padded)
            loss = torch.mean(loss)

            loss.backward()
            optimizer.step()

            print('Optimizing {:d}/{:d}: loss={:.4e}'.format(num_opt + 1, FLAGS.num_opts, loss.item()))

        model.eval()
        model.module._force_no_sample = False

    if operations is not None and operations:
        _mroi_norms = _blob['_mroi_norms']
        _droi_norms = _blob['_droi_norms']
        _theta_deltas = _blob['_theta_deltas'].detach()
        _translation2ds = _blob['_translation2ds'].detach()
        _log_depths = _blob['_log_depths'].detach()

        _mroi_op_norms = torch.tensor([[
            (float(operation['from']['v']) - dataset.Camera.v0) / dataset.Camera.focal,
            (float(operation['from']['u']) - dataset.Camera.u0) / dataset.Camera.focal,
        ] for operation in operations]).cuda()

        _mroi_norm_diffs = torch.sum((_mroi_norms[:, None, :] - _mroi_op_norms[None, :, :]) ** 2, dim=2)
        if len(_mroi_norms) < len(_mroi_op_norms):
            index_ops = torch.argmin(_mroi_norm_diffs, dim=1)
            indices = [(index_obj, index_op) for (index_obj, index_op) in enumerate(index_ops)]
        else:
            index_objs = torch.argmin(_mroi_norm_diffs, dim=0)
            indices = [(index_obj, index_op) for (index_op, index_obj) in enumerate(index_objs)]

        for (index_obj, index_op) in indices:
            operation = operations[index_op]
            print('Object #{:d} and operation {!s}'.format(index_obj, operation))

            u = float(operation['from']['u'])
            v = float(operation['from']['v'])

            if operation['type'] == 'delete':
                interests[index_obj] = torch.tensor(0.0).type(torch.ByteTensor).cuda()

            elif operation['type'] == 'modify':
                u = float(operation['to'].get('u', u))
                v = float(operation['to'].get('v', v))
                zoom = float(operation['zoom'])
                ry = float(operation['ry'])

                _center2d = torch.tensor([
                    (v - dataset.Camera.v0) / dataset.Camera.focal,
                    (u - dataset.Camera.u0) / dataset.Camera.focal,
                ]).cuda()
                _translation2d = (_center2d - _mroi_norms[index_obj]) / _droi_norms[index_obj]
                _log_depth = _log_depths[index_obj] - 2 * torch.log(torch.tensor(zoom)).cuda()

                (_theta_cos, _theta_sin) = _theta_deltas[index_obj]
                _cos = torch.cos(torch.tensor(-ry)).cuda()
                _sin = torch.sin(torch.tensor(-ry)).cuda()
                _theta_delta = torch.stack([
                    _theta_cos * _cos - _theta_sin * _sin,
                    _theta_sin * _cos + _theta_cos * _sin,
                ])

                _theta_deltas[index_obj] = _theta_delta
                _translation2ds[index_obj] = _translation2d
                _log_depths[index_obj] = _log_depth

    _blob.update(model.module.render(_blob))

    _rotations = _blob['_rotations']
    # _thetas = _blob['_thetas']
    _alphas = _blob['_alphas']
    _scales = _blob['_scales']
    _depths = _blob['_depths']
    _center2ds = _blob['_center2ds']
    _translations = _blob['_translations']
    _zooms = _blob['_zooms']
    _masks = _blob['_masks']
    _normals = _blob['_normals']
    _depth_maps = _blob['_depth_maps']

    torch.save({
        'num_objs': num_objs,
        'image_masks': image_masks,
        'rois': rois,
        'interests': interests,
        '_scales': _scales,
        '_rotations': _rotations,
        '_translations': _translations,
        '_zoom_tos': _zoom_tos,
    }, os.path.join(image_dir, '{:s}.pkl'.format(name)))

    index_objs = to_numpy(torch.sort(_depths[:, 0], dim=0, descending=True)[1])

    json_obj = {}
    _image_instance_map = torch.zeros(1, height, width).cuda()
    _image_normal_map = torch.full((3, height, width), 0.5).cuda()
    _image_depth_map = torch.full((1, height, width), 1.0).cuda()
    for index_obj in index_objs.tolist():
        print('Object #{:d} with depth {:.2f}'.format(index_obj, to_numpy(_depths[index_obj])[0]))

        if to_numpy(interests[index_obj]):  # interesting obj
            json_obj[index_obj + 1] = {
                'class_id': int(class_ids[index_obj]),
                'depth': float(to_numpy(_depths[index_obj])),
                'alpha': float(to_numpy(_alphas[index_obj])),
            }
            if metas is not None:
                for (key, value) in metas[index_obj].items():
                    json_obj[index_obj + 1][key] = value

            _image_size = int(FLAGS.render_size / _zooms[index_obj])

            _mask = _masks[index_obj]
            _mask_pil = Transforms.to_pil_image(_mask.detach().cpu())
            _mask_pil = Transforms.resize(_mask_pil, (_image_size, _image_size))
            _image_mask_pil = PIL.Image.new(mode='L', size=(width, height))
            _image_mask_pil.paste(_mask_pil, box=(
                int(_center2ds[index_obj, 1] * dataset.Camera.focal + dataset.Camera.u0 - _image_size // 2),
                int(_center2ds[index_obj, 0] * dataset.Camera.focal + dataset.Camera.v0 - _image_size // 2),
            ))
            _image_mask = Transforms.to_tensor(_image_mask_pil).cuda()
            _image_mask = torch.round(_image_mask)

            _image_instance_map = (1 - _image_mask) * _image_instance_map + _image_mask * (1 + index_obj)

            #

            _normal = _normals[index_obj] / 2 + 0.5
            _normal_pil = Transforms.to_pil_image(_normal.detach().cpu())
            _normal_pil = Transforms.resize(_normal_pil, (_image_size, _image_size))
            _image_normal_pil = PIL.Image.new(mode='RGB', size=(width, height))
            _image_normal_pil.paste(_normal_pil, box=(
                int(_center2ds[index_obj, 1] * dataset.Camera.focal + dataset.Camera.u0 - _image_size // 2),
                int(_center2ds[index_obj, 0] * dataset.Camera.focal + dataset.Camera.v0 - _image_size // 2),
            ))
            _image_normal = Transforms.to_tensor(_image_normal_pil).cuda()

            _image_normal_map = (1 - _image_mask) * _image_normal_map + _image_mask * _image_normal

            #

            _depth_map = torch.min(_depth_maps[index_obj] * _zooms[index_obj] / 100.0, torch.tensor(1.0).cuda())
            _depth_map = _depth_map.detach().cpu().numpy().transpose(1, 2, 0)
            _depth_pil = Transforms.to_pil_image(_depth_map)
            _depth_pil = Transforms.resize(_depth_pil, (_image_size, _image_size))
            _image_depth_pil = PIL.Image.new(mode='F', size=(width, height))
            _image_depth_pil.paste(_depth_pil, box=(
                int(_center2ds[index_obj, 1] * dataset.Camera.focal + dataset.Camera.u0 - _image_size // 2),
                int(_center2ds[index_obj, 0] * dataset.Camera.focal + dataset.Camera.v0 - _image_size // 2),
            ))
            _image_depth = Transforms.to_tensor(_image_depth_pil).cuda()

            _image_depth_map = (1 - _image_mask) * _image_depth_map + _image_mask * _image_depth

        elif operations is None:
            image_mask = image_masks[index_obj]

            _image_instance_map = (1 - image_mask) * _image_instance_map + image_mask * (1 + index_obj)

    with open(os.path.join(image_dir, '{:s}.json'.format(name)), 'w') as f:
        json.dump(json_obj, f, indent=4)

    (_image_instance_map_pil, _image_vis_pil) = Transforms.visualize(image_rgb, to_numpy(_image_instance_map), to_numpy(rois), to_numpy(interests))
    _image_instance_map_pil.save(os.path.join(image_dir, '{:s}.png'.format(name)))
    _image_vis_pil.save(os.path.join(image_dir, '{:s}-visualize.png'.format(name)))

    _image_normal_map_pil = Transforms.to_pil_image(_image_normal_map.detach().cpu())
    _image_normal_map_pil.save(os.path.join(image_dir, '{:s}-normal.png'.format(name)))

    _image_depth_map = np.uint16(_image_depth_map.detach().cpu().numpy().transpose(1, 2, 0) * 65535)
    _image_depth_map_pil = PIL.Image.new('I', _image_depth_map.T.shape[1:])
    _image_depth_map_pil.frombytes(_image_depth_map.tobytes(), 'raw', 'I;16')
    _image_depth_map_pil.save(os.path.join(image_dir, '{:s}-depth.png'.format(name)))


def test():
    data_loader = TestLoader()
    dataset = data_loader.dataset
    df = dataset.df

    if FLAGS._do == '_test':
        model = DataParallel(Model().cuda())
        model.eval()
        net = TestNet(model=model)
        net.load(ckpt_dir=FLAGS.ckpt_dir)

    elif FLAGS._do == '_test_2d':
        model = None

    if FLAGS.source == 'gt':
        maskrcnn = None

    elif FLAGS.source == 'maskrcnn':
        if FLAGS.dataset == 'vkitti':
            num_classes = 3
        elif FLAGS.dataset == 'cityscapes':
            num_classes = 2

        class InferenceConfig(Config):
            NAME = FLAGS.dataset
            IMAGES_PER_GPU = 1
            GPU_COUNT = 1
            NUM_CLASSES = num_classes

        state_dict = torch.load(FLAGS.maskrcnn_path)

        maskrcnn = MaskRCNN(
            config=InferenceConfig(),
            model_dir='/tmp',
        )
        maskrcnn = maskrcnn.cuda()
        maskrcnn.load_state_dict(state_dict)

    if FLAGS.edit_json is None:
        assert FLAGS.input_file is None

        items = np.random.permutation(df.index.unique())
        operations_list = [None] * len(items)
        names = [None] * len(items)
    else:
        with open(FLAGS.edit_json, 'r') as f:
            edit_json_objs = json.load(f)

        items = []
        operations_list = []
        names = []
        for edit_json_obj in edit_json_objs:
            if FLAGS.dataset == 'vkitti':
                item = (edit_json_obj['world'], edit_json_obj['topic'], int(edit_json_obj['source']))
                name = edit_json_obj['target']
            elif FLAGS.dataset == 'cityscapes':
                item = (edit_json_obj['split'], edit_json_obj['city'], edit_json_obj['seq'], edit_json_obj['source'])
                name = edit_json_obj['target']
            else:
                raise Exception

            items.append(item)
            operations_list.append(edit_json_obj['operations'])
            names.append(name)

    for (item, operations, name) in zip(items, operations_list, names):
        print(item)

        if FLAGS.dataset == 'vkitti':
            (world, topic, frame) = item

            image_dir = os.path.join(FLAGS.output_dir, FLAGS.dataset, FLAGS.source, world, topic)
            _name = '{:05d}'.format(frame)

        elif FLAGS.dataset == 'cityscapes':
            (split, city, seq, frame) = item

            image_dir = os.path.join(FLAGS.output_dir, FLAGS.dataset, FLAGS.source, split, city)
            _name = '{:s}_{:s}_{:s}'.format(city, seq, frame)

        name = name or _name

        lock_path = os.path.join(image_dir, '{:s}.lock'.format(name))
        if os.path.isfile(lock_path):
            print('Skipped')
            continue
        else:
            if not os.path.isdir(image_dir):
                os.makedirs(image_dir)

            with open(lock_path, 'w') as f:
                pass

        metas = None
        if FLAGS.input_file is None:
            image_rgb = dataset.read_rgb(*item)
        else:
            image_rgb = np.asarray(PIL.Image.open(FLAGS.input_file))

        if FLAGS.source == 'gt':
            class_ids = []
            image_masks = []
            image_ignores = []
            rois = []

            if FLAGS.dataset == 'vkitti':
                image_scene = dataset.read_scene(world, topic, frame)

                metas = []
                slice_obj = dataset.motgt_df.index.get_loc(item)
                indices = range(len(dataset.motgt_df))[slice_obj]
                item_objs = list(map(dataset.motgt_df.iloc.__getitem__, indices))

                for item_obj in item_objs:
                    code = dataset.scenegt_df.loc[(
                        world,
                        topic,
                        '{:s}:{:d}'.format(item_obj.orig_label, item_obj.tid),
                    )].values.astype(np.uint8)

                    image_mask = Transforms.scene_to_mask(image_scene, code)
                    roi = Transforms.mask_to_roi(image_mask)

                    image_mask = np.transpose(image_mask, (2, 0, 1))

                    if item_obj.orig_label == 'Car':
                        class_id = 1
                    elif item_obj.orig_label == 'Van':
                        class_id = 2

                    class_ids.append(class_id)
                    image_masks.append(image_mask)
                    image_ignores = None
                    rois.append(roi)
                    metas.append({
                        'tid': int(item_obj.tid),
                    })

            elif FLAGS.dataset == 'cityscapes':
                image_scene = dataset.read_scene(split, city, seq, frame)
                image_disparity = dataset.read_disparity(split, city, seq, frame)

                obj_indices = np.unique(image_scene)
                for obj_index in obj_indices:
                    if not dataset.index2cat(obj_index) == dataset.Category.car:
                        continue

                    image_mask = Transforms.scene_to_mask(image_scene, [obj_index])
                    roi = Transforms.mask_to_roi(image_mask)

                    disparity = image_disparity[image_mask.astype(np.bool)]
                    disparity = disparity[disparity != 0]
                    if disparity.size:
                        disparity = np.percentile(disparity, 95)
                    else:
                        disparity = 0
                    image_ignore = Transforms.scene_to_mask(image_disparity > disparity, 1.0)

                    image_mask = np.transpose(image_mask, (2, 0, 1))
                    image_ignore = np.transpose(image_ignore, (2, 0, 1))

                    class_ids.append(1)
                    image_masks.append(image_mask)
                    image_ignores.append(image_ignore)
                    rois.append(roi)

            class_ids = np.stack(class_ids, axis=0)
            image_masks = np.stack(image_masks, axis=0)
            if image_ignores is not None:
                image_ignores = np.stack(image_ignores, axis=0)
            rois = np.stack(rois, axis=0)

        elif FLAGS.source == 'maskrcnn':
            try:
                detections = maskrcnn.detect([image_rgb])[0]

                class_ids = detections['class_ids']
                image_masks = detections['masks']
                rois = detections['rois']

                image_masks = np.transpose(image_masks, (2, 0, 1))
                image_masks = np.expand_dims(image_masks, axis=1)
                image_ignores = None

            except Exception:
                continue

        sels = np.flipud(np.argsort(np.sum(image_masks, axis=(1, 2, 3))))[:min(len(class_ids), 16)]

        class_ids = class_ids[sels]
        image_masks = image_masks[sels]
        if image_ignores is not None:
            image_ignores = image_ignores[sels]
        rois = rois[sels]

        if FLAGS._do == '_test':
            _test(
                dataset,
                model,
                image_dir,
                name,
                image_rgb,
                class_ids,
                image_masks,
                image_ignores,
                rois,
                metas,
                operations,
            )
        elif (FLAGS._do == '_test_2d') or (FLAGS._do == '_test_2d_plus'):
            globals()[FLAGS._do](
                dataset,
                image_dir,
                name,
                image_rgb,
                class_ids,
                image_masks,
                rois,
                operations,
            )


if __name__ == '__main__':
    argv = FLAGS(sys.argv)

    if FLAGS.mode is not None:
        FLAGS.mode = TargetType.__dict__[FLAGS.mode]

    locals()[FLAGS.do]()
