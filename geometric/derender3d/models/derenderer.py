import torch
import torchvision

from torch.nn.modules import Module


class Derenderer(Module):
    in_size = 4
    hidden_size = 256

    def __init__(self, num_classes=8, grid_size=4):
        super(Derenderer, self).__init__()

        self.num_classes = num_classes
        self.grid_size = grid_size
        self.out_sizes = {
            '_theta_deltas': 2,
            '_translation2ds': 2,
            '_log_scales': 3,
            '_log_depths': 1,
            '_class_probs': num_classes,
            '_ffd_coeffs': num_classes * (grid_size ** 3) * 3
        }

        self.net = torchvision.models.resnet18(pretrained=True)
        self.net.avgpool = torch.nn.AdaptiveAvgPool2d(1)  # PATCH
        self.net.fc = torch.nn.Linear(512, Derenderer.hidden_size)
        self.relu = torch.nn.ReLU(inplace=True)

        self.fc1 = torch.nn.Linear(self.hidden_size + self.in_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self._fc3 = torch.nn.Linear(self.hidden_size, sum(self.out_sizes.values()))

    def forward(self, images, mroi_norms, droi_norms):
        x = self.net(images)
        x = self.relu(x)

        x = torch.cat([x, mroi_norms, droi_norms], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self._fc3(x)

        (
            _theta_deltas,
            _translation2ds,
            _log_scales,
            _log_depths,
            _class_probs,
            _ffd_coeffs,
        ) = torch.split(x, list(self.out_sizes.values()), dim=1)

        _theta_deltas = _theta_deltas / torch.norm(_theta_deltas, p=2, dim=1, keepdim=True)
        _class_probs = torch.nn.functional.softmax(_class_probs, dim=1)
        _ffd_coeffs = _ffd_coeffs.view(-1, self.num_classes, (self.grid_size ** 3) * 3)

        return {
            '_theta_deltas': _theta_deltas,
            '_translation2ds': _translation2ds,
            '_log_scales': _log_scales,
            '_log_depths': _log_depths,
            '_class_probs': _class_probs,
            '_ffd_coeffs': _ffd_coeffs,
        }
