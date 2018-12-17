import collections
import numpy as np
import torch

from derender3d import TargetType
from derender3d.datasets import (
    VKitti,
    KittiObject,
    KittiSemantics,
    KittiSemanticsHybrid,
    CityscapesSemantics,
    CityscapesSemanticsHybrid,
)


# collate with missing keys
def collate_fn(batch):
    for d in batch:
        if d is not None:
            obj = d
            break

    if isinstance(obj, torch.Tensor):
        for (num, d) in enumerate(batch):
            if d is None:
                batch[num] = torch.zeros_like(obj)

    elif type(obj).__module__ == 'numpy':
        for (num, d) in enumerate(batch):
            if d is None:
                batch[num] = np.zeros_like(obj)

    elif isinstance(obj, collections.Mapping):
        keys = np.unique([key for d in batch for key in d.keys()])
        return {key: collate_fn([d.get(key, None) for d in batch]) for key in keys}

    return torch.utils.data.dataloader.default_collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, mode, batch_size, num_workers, is_train, debug=False):
        if dataset == 'vkitti':
            dataset = VKitti(is_train=is_train, debug=debug)
            shuffle = is_train
            sampler = None

        elif dataset == 'kitti':
            if (mode == TargetType.pretrain) or (mode == TargetType.extend):
                dataset = KittiObject(is_train=is_train, debug=debug)
                shuffle = is_train
                sampler = None

            elif mode == TargetType.finetune:
                dataset = KittiSemantics(is_train=is_train, debug=debug)
                shuffle = is_train
                sampler = None

            elif mode == TargetType.full:
                dataset = KittiSemanticsHybrid(is_train=is_train, debug=debug)
                shuffle = None
                sampler = torch.utils.data.sampler.WeightedRandomSampler(dataset.get_weights(), len(dataset))

        elif dataset == 'cityscapes':
            if mode == TargetType.full:
                dataset = CityscapesSemanticsHybrid(is_train=is_train, debug=debug)
                shuffle = is_train
                sampler = None

            elif mode == TargetType.extend:
                dataset = CityscapesSemantics(is_train=is_train, debug=debug)
                shuffle = is_train
                sampler = None

        super(DataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,  # No support for users without GPUs, sorry...
        )
