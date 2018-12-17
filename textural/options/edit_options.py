# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions


class EditOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='edit', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='60', help='which epoch to load? set to latest to use latest cached model')
        self.isTrain = False

        self.parser.add_argument('--edit_source', type=str, default='', help='relative path of image on which we read features and edit')
        self.parser.add_argument('--edit_dir', type=str, default='', help='new instance map and json guiding how to manipulate')
        self.parser.add_argument('--edit_num', type=int, default=5, help='# of manipulation tasks')
        self.parser.add_argument('--edit_list', type=str, default='', help='edit list for benchmark edit')
        self.parser.add_argument('--experiment_name', type=str, default='edit', help='experiment name')
