import glob
import json
import os
import torch


class Saver(object):
    @staticmethod
    def meta_path(dir_):
        return os.path.join(dir_, 'meta.json')

    @staticmethod
    def model_path(dir_, num_step=None, model_name=None):
        model_name = model_name or 'model-*.ckpt'

        if num_step is not None:
            model_name = model_name.replace('*', '{:d}'.format(num_step))

        return os.path.join(dir_, model_name)

    @staticmethod
    def load_model(ckpt_dir, num_step=None, model_name=None):
        if num_step is None:
            model_paths = glob.glob(Saver.model_path(ckpt_dir, num_step, model_name))
            model_path = max(model_paths, key=os.path.getmtime)
        else:
            model_path = Saver.model_path(ckpt_dir, num_step)

        return torch.load(model_path)

    def __init__(self, working_dir):
        self.working_dir = working_dir

    def save_meta(self, meta):
        with open(Saver.meta_path(self.working_dir), 'w') as f:
            json.dump(meta, f, indent=4)

    def save_model(self, obj, num_step):
        torch.save(obj, Saver.model_path(self.working_dir, num_step))
