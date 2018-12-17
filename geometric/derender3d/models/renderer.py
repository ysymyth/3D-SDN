import chainer
import chainer.functions as cf
import torch

from torch import Tensor
from torch.autograd import Function
from torch.nn.modules import Module

import neural_renderer as nr


class RenderType:
    RGB = 0
    Silhouette = 1
    Depth = 2
    Normal = 3


class _Renderer(nr.Renderer):
    def render_silhouettes(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = cf.concat((faces, faces[:, :, ::-1]), axis=1).data

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction, self.up)

        # perspective transformation
        if self.perspective:
            vertices = nr.perspective(vertices, angle=self.viewing_angle)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def render_depth(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = cf.concat((faces, faces[:, :, ::-1]), axis=1).data

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction, self.up)

        # perspective transformation
        if self.perspective:
            vertices = nr.perspective(vertices, angle=self.viewing_angle)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render_normal(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = cf.concat((faces, faces[:, :, ::-1]), axis=1).data

        # normal
        faces_normal = nr.vertices_to_faces(vertices, faces)

        (bs, nf) = faces_normal.shape[:2]
        faces_normal = faces_normal.reshape((bs * nf, 3, 3))
        v10 = faces_normal[:, 0] - faces_normal[:, 1]
        v12 = faces_normal[:, 2] - faces_normal[:, 1]
        normals = cf.normalize(nr.cross(v10, v12))
        normals = normals.reshape((bs, nf, 3))

        textures = normals[:, :, None, None, None, :]
        textures = cf.tile(textures, (1, 1, 2, 2, 2, 1))

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction, self.up)

        # perspective transformation
        if self.perspective:
            vertices = nr.perspective(vertices, angle=self.viewing_angle)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return images

    def render(self, vertices, faces, textures):
        # fill back
        if self.fill_back:
            faces = cf.concat((faces, faces[:, :, ::-1]), axis=1).data
            textures = cf.concat((textures, textures.transpose((0, 1, 4, 3, 2, 5))), axis=1)

        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction, self.up)

        # perspective transformation
        if self.perspective:
            vertices = nr.perspective(vertices, angle=self.viewing_angle)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return images


class RenderFunction(Function):
    # Unfortunately direct on-GPU transfer from torch to chainer remains a mystery,
    # so let's settle for a GPU-CPU-GPU transfer scheme

    @staticmethod
    def torch2numpy(torch_tensor):
        return torch_tensor.detach().cpu().numpy()

    @staticmethod
    def chainer2numpy(chainer_tensor):
        return chainer.cuda.to_cpu(chainer_tensor)

    @staticmethod
    def torch2chainer(torch_tensor):
        device = torch_tensor.get_device()
        return chainer.cuda.to_gpu(RenderFunction.torch2numpy(torch_tensor), device=device)

    @staticmethod
    def chainer2torch(chainer_tensor):
        device = chainer.cuda.get_device_from_array(chainer_tensor).id
        return torch.Tensor(RenderFunction.chainer2numpy(chainer_tensor)).cuda(device=device)

    @staticmethod
    def forward(ctx,
                vertices,
                faces,
                textures,
                renderer,
                render_type,
                eye,
                camera_mode,
                camera_direction,
                camera_up):

        _vertices = chainer.Variable(RenderFunction.torch2chainer(vertices))
        _faces = chainer.Variable(RenderFunction.torch2chainer(faces))
        _textures = None
        _eye = chainer.Variable(RenderFunction.torch2chainer(eye))
        _camera_direction = chainer.Variable(RenderFunction.torch2chainer(camera_direction))
        _camera_up = chainer.Variable(RenderFunction.torch2chainer(camera_up))

        if render_type == RenderType.RGB:
            _textures = chainer.Variable(RenderFunction.torch2chainer(textures))

        renderer.eye = _eye
        renderer.camera_mode = camera_mode
        renderer.camera_direction = _camera_direction
        renderer.up = _camera_up

        if render_type == RenderType.RGB:
            _images = renderer.render(_vertices, _faces, _textures)
        elif render_type == RenderType.Silhouette:
            _images = renderer.render_silhouettes(_vertices, _faces)
            _images = chainer.functions.expand_dims(_images, axis=1)
        elif render_type == RenderType.Depth:
            _images = renderer.render_depth(_vertices, _faces)
            _images = chainer.functions.expand_dims(_images, axis=1)
        elif render_type == RenderType.Normal:
            _images = renderer.render_normal(_vertices, _faces)

        ctx._vertices = _vertices
        ctx._textures = _textures
        ctx._render_type = render_type
        ctx._images = _images

        images = RenderFunction.chainer2torch(_images.data)
        return images

    @staticmethod
    def backward(ctx, grad_images):
        _grad_images = chainer.Variable(RenderFunction.torch2chainer(grad_images.data))

        ctx._images.grad_var = _grad_images
        ctx._images.backward()

        grad_vertices = None
        if ctx.needs_input_grad[0]:
            grad_vertices = RenderFunction.chainer2torch(ctx._vertices.grad_var.data)

        grad_textures = None
        if ctx.needs_input_grad[2] and (ctx._render_type == RenderType.RGB):
            grad_textures = RenderFunction.chainer2torch(ctx._textures.grad_var.data)

        return (grad_vertices, None, grad_textures, None, None, None, None, None, None)


class Renderer(Module):
    def __init__(self,
                 image_size=256,
                 viewing_angle=30):

        super(Renderer, self).__init__()

        self.image_size = image_size
        self.viewing_angle = viewing_angle

        self.eye = Tensor([0, 0, 0])
        self.camera_mode = 'look'
        self.camera_direction = Tensor([0, 0, -1])
        self.camera_up = Tensor([0, 1, 0])

    def forward(self,
                vertices,
                faces,
                textures=None,
                render_type=RenderType.RGB):

        _renderer = _Renderer()
        _renderer.image_size = self.image_size
        _renderer.viewing_angle = self.viewing_angle

        # HUGE BUG FOR DIZ RENDERER: IT INVERTS X AXIS!!!
        # WE HENCE FIX IT HERE
        vertices = vertices * Tensor([-1, 1, 1]).cuda()

        eye = self.eye.cuda()
        camera_mode = self.camera_mode
        camera_direction = self.camera_direction.cuda()
        camera_up = self.camera_up.cuda()

        batch_size = len(vertices)

        eye = eye[None, :].expand(batch_size, -1)
        camera_direction = camera_direction[None, :].expand(batch_size, -1)
        camera_up = camera_up[None, :].expand(batch_size, -1)

        images = RenderFunction.apply(
            vertices,
            faces,
            textures,
            _renderer,
            render_type,
            eye,
            camera_mode,
            camera_direction,
            camera_up,
        )

        if render_type == RenderType.Normal:
            (x, y, z) = torch.unbind(images, dim=1)
            images = torch.stack([-x, y, z], dim=1)

        return images
