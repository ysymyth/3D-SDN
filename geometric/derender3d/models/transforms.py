import numpy as np
import torch
import scipy.ndimage
import scipy.special

from torch import Tensor
from torch.nn.modules import Module


class FFD(Module):
    class Constraint:
        class Type:
            symmetry = 0
            homogeneity = 1

        class Axis:
            x = 0
            y = 1
            z = 2

        @staticmethod
        def symmetry(axis):
            c = FFD.Constraint(FFD.Constraint.Type.symmetry)
            c.axis = axis
            return c

        @staticmethod
        def homogeneity(axis, index):
            c = FFD.Constraint(FFD.Constraint.Type.homogeneity)
            c.axis = axis
            c.index = index
            return c

        def __init__(self, type):
            self.type = type

    @staticmethod
    def flip(x, dim):
        shape = x.shape
        index = torch.arange(shape[dim] - 1, -1, -1, dtype=torch.long).cuda()
        return torch.index_select(x, dim, index)

    def __init__(self, vertices, num_grids=4, constraints=None):
        super(FFD, self). __init__()

        assert num_grids % 2 == 0

        self.num_grids = num_grids
        self.constraints = constraints

        grids = np.arange(num_grids)

        binoms = Tensor(scipy.special.binom(num_grids - 1, grids))
        grid_1ds = Tensor(grids)
        grid_3ds = Tensor(np.meshgrid(grids, grids, grids, indexing='ij'))

        coeff = (
            binoms *
            torch.pow(torch.unsqueeze(0.5 + vertices, dim=2), grid_1ds) *
            torch.pow(torch.unsqueeze(0.5 - vertices, dim=2), num_grids - 1 - grid_1ds)
        )

        self.B = torch.einsum('ni,nj,nk->nijk', torch.unbind(coeff, dim=1))
        self.B = torch.unsqueeze(self.B, dim=1)

        self.P0 = grid_3ds / (num_grids - 1) - 0.5

    def forward(self, ffd_coeff):
        dP = ffd_coeff.view(3, self.num_grids, self.num_grids, self.num_grids)

        for constraint in self.constraints:
            if constraint.type == FFD.Constraint.Type.symmetry:
                _dP = FFD.flip(dP, dim=constraint.axis + 1)
                (_dPx, _dPy, _dPz) = torch.unbind(_dP, dim=0)
                _dP = torch.stack([_dPx, _dPy, -_dPz], dim=0)

                dP = (dP + _dP) / 2

            elif constraint.type == FFD.Constraint.Type.homogeneity:
                dPs = torch.unbind(dP, dim=constraint.axis + 1)

                _dPs = [dPs[index] for index in constraint.index]
                _dP_mean = sum(_dPs) / len(_dPs)

                _dPs = []
                for index in range(self.num_grids):
                    if index in constraint.index:
                        _dP = _dP_mean.clone()
                        _dP[constraint.axis] = dPs[index][constraint.axis]
                    else:
                        _dP = dPs[index]

                    _dPs.append(_dP)

                dP = torch.stack(_dPs, dim=constraint.axis + 1)

        PB = (self.P0.cuda() + dP) * self.B.cuda()
        V = PB.view(-1, 3, self.num_grids * self.num_grids * self.num_grids).sum(dim=2)
        return V


class PerspectiveTransform(Module):
    def forward(self,
                vertices,
                scales=None,
                rotations=None,
                translations=None,
                perspective_translations=None,
                zooms=None,
                zoom_tos=None):

        if scales is not None:
            scales = scales.unsqueeze(dim=1)
            vertices = vertices * scales

        if rotations is not None:
            (a, b, c, d) = torch.unbind(rotations, dim=1)

            T = torch.stack([
                a * a + b * b - c * c - d * d,
                2 * b * c - 2 * a * d,
                2 * b * d + 2 * a * c,
                2 * b * c + 2 * a * d,
                a * a - b * b + c * c - d * d,
                2 * c * d - 2 * a * b,
                2 * b * d - 2 * a * c,
                2 * c * d + 2 * a * b,
                a * a - b * b - c * c + d * d,
            ], dim=1).view(-1, 3, 3)

            vertices = torch.matmul(vertices, torch.transpose(T, dim0=1, dim1=2))

        if translations is not None:
            translations = translations.unsqueeze(dim=1)
            vertices = vertices + translations

        if perspective_translations is not None:
            perspective_translations = perspective_translations.unsqueeze(dim=1)
        else:
            perspective_translations = translations

        (x, y, z) = torch.unbind(vertices, dim=2)
        (x0, y0, z0) = torch.unbind(perspective_translations, dim=2)

        x = x - x0 / z0 * z
        y = y - y0 / z0 * z

        if zoom_tos is not None:
            zooms = torch.min(torch.abs(z) / torch.max(torch.abs(x), torch.abs(y)), dim=1, keepdim=True)[0] * zoom_tos

        z = z / zooms

        vertices = torch.stack([x, y, z], dim=2)

        if zoom_tos is None:
            return vertices
        else:
            return (vertices, zooms)
