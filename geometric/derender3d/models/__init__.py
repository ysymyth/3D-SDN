import numpy as np
import os
import torch
import sys

from torch import Tensor, IntTensor
from torch.distributions import Categorical
from torch.nn.modules import Module

import neural_renderer as nr

from derender3d import TargetType
from derender3d.models.derenderer import Derenderer
from derender3d.models.renderer import RenderType, Renderer
from derender3d.models.transforms import FFD, PerspectiveTransform


class ShapenetObj(object):
    root_dir = os.getenv('SHAPENET_ROOT_DIR')

    def __init__(self,
                 class_id,
                 obj_id,
                 root_dir=root_dir):

        path = os.path.join(root_dir, class_id, obj_id, 'models', 'model_normalized.obj')
        print('Reading {:s}'.format(path))

        (vertices, faces) = nr.load_obj(path)
        vertices = vertices / np.ptp(vertices, axis=0)
        vertices = vertices[:, [2, 1, 0]] * np.asarray([-1, 1, 1], dtype=np.float32)

        self.vertices = Tensor(vertices)
        self.faces = IntTensor(faces)


class Derenderer3d(Module):
    def __init__(self, mode, image_size, render_size):
        super(Derenderer3d, self).__init__()

        self.mode = mode
        self.image_size = image_size
        self.render_size = render_size

        self.derenderer = Derenderer()

        self._force_no_sample = False

        if mode & TargetType.reproject:
            self.objs = [
                ShapenetObj(class_id='02958343', obj_id='137f67657cdc9da5f985cd98f7d73e9a'),
                ShapenetObj(class_id='02958343', obj_id='5343e944a7753108aa69dfdc5532bb13'),
                ShapenetObj(class_id='02958343', obj_id='3776e4d1e2587fd3253c03b7df20edd5'),
                ShapenetObj(class_id='02958343', obj_id='3ba5bce1b29f0be725f689444c7effe2'),
                ShapenetObj(class_id='02958343', obj_id='53a031dd120e81dc3aa562f24645e326'),
                ShapenetObj(class_id='02924116', obj_id='7905d83af08a0ca6dafc1d33c05cbcf8'),
                ShapenetObj(class_id='02958343', obj_id='a0fe4aac120d5f8a5145cad7315443b3'),
                ShapenetObj(class_id='02958343', obj_id='cd7feedd6041209131ac5fb37e6c8324'),
            ]
            self.ffds = [FFD(obj.vertices, constraints=[
                FFD.Constraint.symmetry(axis=FFD.Constraint.Axis.z),
                FFD.Constraint.homogeneity(axis=FFD.Constraint.Axis.y, index=[0, 1]),
            ]) for obj in self.objs]
            self.perspective_transform = PerspectiveTransform()
            self.renderer = Renderer(image_size=render_size)

    def forward(self, images, roi_norms, focals):
        # batch_size = len(images)

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

        _blob.update(self.derenderer(images, _mroi_norms, _droi_norms))

        if not (self.mode & TargetType.reproject):
            return _blob

        _blob.update(self.render(_blob))
        return _blob

    def render(self, blob):
        _mroi_norms = blob['_mroi_norms']
        _droi_norms = blob['_droi_norms']
        _focals = blob['_focals']
        _theta_deltas = blob['_theta_deltas']
        _translation2ds = blob['_translation2ds']
        _log_scales = blob['_log_scales']
        _log_depths = blob['_log_depths']
        _class_probs = blob['_class_probs']
        _ffd_coeffs = blob['_ffd_coeffs']

        batch_size = len(_focals)

        _thetas = torch.unsqueeze(torch.atan2(_theta_deltas[:, 1], _theta_deltas[:, 0]), dim=1)
        _rotations = torch.cat([
            torch.cos(_thetas / 2),
            torch.zeros(batch_size, 1).cuda(),
            torch.sin(_thetas / 2),
            torch.zeros(batch_size, 1).cuda(),
        ], dim=1)
        _areas = torch.unsqueeze(_droi_norms[:, 0] * _droi_norms[:, 1], dim=1)

        _scales = torch.exp(_log_scales)
        _depths = torch.sqrt(torch.exp(_log_depths) / _areas)

        _center2ds = _mroi_norms + _translation2ds * _droi_norms
        _translation_units = torch.stack([
            _center2ds[:, 1],
            - _center2ds[:, 0],
            - torch.ones(batch_size).cuda(),
        ], dim=1)
        _translation_units = _translation_units / torch.norm(_translation_units, p=2, dim=1, keepdim=True)
        _translations = _depths * _translation_units

        _alphas = - (_thetas - torch.atan(_translations[:, 0:1] / _translations[:, 2:3]))
        _alphas = torch.remainder(_alphas + np.pi, 2 * np.pi) - np.pi

        if self.training and not self._force_no_sample:
            _class_dists = Categorical(_class_probs)
            _class_samples = _class_dists.sample()
            _class_log_probs = _class_dists.log_prob(_class_samples)

        else:
            (_class_max_probs, _class_samples) = torch.max(_class_probs, dim=1)
            # print(_class_max_probs)
            # print(_class_samples)
            _class_log_probs = torch.log(_class_max_probs)

        if self.training:
            _perspective_translation_units = torch.stack([
                _mroi_norms[:, 1],
                - _mroi_norms[:, 0],
                - torch.ones(batch_size).cuda(),
            ], dim=1)
            _perspective_translation_units = _perspective_translation_units / torch.norm(_perspective_translation_units, p=2, dim=1, keepdim=True)
            _perspective_translations = _depths * _perspective_translation_units
            _zooms = (self.image_size / _focals) / torch.max(_droi_norms, dim=1, keepdim=True)[0]

        else:
            _zoom_tos = self.render_size / (2.0 * _focals)
            _zooms = []

        device = torch.cuda.current_device()

        _masks = []
        _normals = []
        _depth_maps = []
        for size in range(batch_size):
            print(device, end='')
            sys.stdout.flush()

            _class_sample = int(_class_samples[size])
            _ffd_coeff = _ffd_coeffs[size][_class_sample]

            vertices = self.ffds[_class_sample](_ffd_coeff)
            faces = self.objs[_class_sample].faces.cuda()

            __vertices = vertices.unsqueeze(dim=0)
            __faces = faces.unsqueeze(dim=0)

            __scales = _scales[size].unsqueeze(dim=0)
            __rotations = _rotations[size].unsqueeze(dim=0)
            __translations = _translations[size].unsqueeze(dim=0)

            if self.training:
                __perspective_translations = _perspective_translations[size].unsqueeze(dim=0)
                __zooms = _zooms[size].unsqueeze(dim=0)

                __vertices = self.perspective_transform(
                    __vertices,
                    scales=__scales,
                    rotations=__rotations,
                    translations=__translations,
                    perspective_translations=__perspective_translations,
                    zooms=__zooms,
                )
            else:
                __zoom_tos = _zoom_tos[size].unsqueeze(dim=0)
                (__vertices, __zooms) = self.perspective_transform(
                    __vertices,
                    scales=__scales,
                    rotations=__rotations,
                    translations=__translations,
                    perspective_translations=__translations,
                    zoom_tos=__zoom_tos,
                )
                _zooms.append(__zooms)

            self.renderer.viewing_angle = np.arctan(self.render_size / (2.0 * _focals[size].item())) / np.pi * 180
            __masks = self.renderer(
                vertices=__vertices,
                faces=__faces,
                render_type=RenderType.Silhouette,
            )
            _masks.append(__masks)

            if self.mode & TargetType.normal:
                __normals = self.renderer(
                    vertices=__vertices,
                    faces=__faces,
                    render_type=RenderType.Normal,
                )
                _normals.append(__normals)

            if self.mode & TargetType.depth:
                __depth_maps = self.renderer(
                    vertices=__vertices,
                    faces=__faces,
                    render_type=RenderType.Depth,
                )
                _depth_maps.append(__depth_maps)

        if not self.training:
            _zooms = torch.cat(_zooms, dim=0)

        _masks = torch.cat(_masks, dim=0)

        if self.mode & TargetType.normal:
            _normals = torch.cat(_normals, dim=0)

        if self.mode & TargetType.depth:
            _depth_maps = torch.cat(_depth_maps, dim=0)

        return {
            '_thetas': _thetas,
            '_alphas': _alphas,
            '_rotations': _rotations,
            '_scales': _scales,
            '_depths': _depths,
            '_center2ds': _center2ds,
            '_translations': _translations,
            '_class_log_probs': _class_log_probs,
            '_zooms': _zooms,
            '_masks': _masks,
            '_normals': _normals,
            '_depth_maps': _depth_maps,
        }
