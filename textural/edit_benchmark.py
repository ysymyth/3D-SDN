# Edit benchmark

# System libs
import os
import sys
from math import pi

# Third party libs
import torch
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
from PIL import Image
import json

# our libs
sys.path.insert(0, os.path.dirname(__file__))
from util import util, html
from util.visualizer import Visualizer
from models.models import create_model as create_pix2pix_model
from options.edit_options import EditOptions
from data.base_dataset import get_params, get_transform

# opt and setup
opt = EditOptions().parse(save=False)
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
if opt.feat_pose_num_bins:
    bins = np.array(list(range(-180, 181, 360 // opt.feat_pose_num_bins))) / 180

model = create_pix2pix_model(opt)
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, '%s_%s_%s_%s' % (opt.name, opt.experiment_name, opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Experiment name = %s, Phase = %s, Epoch = %s' % (opt.name, opt.experiment_name, opt.phase, opt.which_epoch))

loss = torch.nn.L1Loss()
losses = []
paths = []

# edit
edit_list = json.load(open(opt.edit_list))
edit_list = edit_list[:len(edit_list) // 2]  # last half of the edit list is reconstruction
for i, edit_item in enumerate(edit_list):
    # set up base images
    world, topic, source, target = edit_item['world'], edit_item['topic'], edit_item['source'], edit_item['target']
    edit_source = world + '/' + topic + '/' + source + '.png'
    edit_target = world + '/' + topic + '/' + target + '.png'

    base_img = Image.open(os.path.join(opt.dataroot, 'vkitti_1.3.1_rgb', edit_source)).convert('RGB')
    target_img = Image.open(os.path.join(opt.dataroot, 'vkitti_1.3.1_rgb', edit_target)).convert('RGB')
    base_segm = Image.open(os.path.join(opt.segm_precomputed_path, edit_source))
    base_inst_exist = 1
    try:
        base_inst = Image.open(os.path.join(opt.edit_dir, edit_source))
    except FileNotFoundError:
        print('no inst found at', edit_source)
        base_inst = base_segm.copy()
        base_inst_exist = 0
    print(edit_source, edit_target)

    # set up transforms
    params = get_params(opt, base_segm.size)
    transform_A = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    transform_B = get_transform(opt, params)
    base_segm, base_inst = transform_A(base_segm) * 255.0, transform_A(base_inst) * 255.0
    base_img, target_img = transform_B(base_img), transform_B(target_img)
    base_segm = base_segm + 1
    if not base_inst_exist:
        base_inst = base_segm.clone()
    else:
        base_inst *= 1000
        base_segm[(base_inst == 0) & (base_segm == 2)] = 5
        base_segm[(base_inst == 0) & (base_segm == 12)] = 5
        base_inst[base_inst == 0] = base_segm[base_inst == 0]

    # obtain instance feat
    feat_dict = model.netE.generate_feat_dict(Variable(base_img.unsqueeze(0)).cuda(), base_inst.unsqueeze(0).cuda())

    inst = Image.open(os.path.join(opt.edit_dir, edit_target))
    inst = (transform_A(inst) * 255.0).int()
    d = json.load(open(os.path.join(opt.edit_dir, edit_target.replace('.png', '.json'))))

    segm = base_segm.int()
    feat = torch.zeros(opt.feat_num, segm.size(1), segm.size(2))
    pose = torch.zeros(segm.size()) if opt.feat_pose_num_bins else torch.zeros(2, segm.size(1), segm.size(2))
    segm[segm == 2] = 5
    segm[segm == 12] = 5

    for k, v in d.items():
        k = int(k)
        alpha, class_id = v['alpha'], v['class_id']
        inst_id = k * 1000
        inst[inst == k] = inst_id
        # process segm: remove original car and add current car
        segm[inst == inst_id] = {1: 2, 2: 12}[v['class_id']]
        # process pose
        if opt.feat_pose_num_bins:
            pose[inst == inst_id] = int(np.digitize(alpha / pi, bins))

    # process inst: use segm to complete the background
    inst[inst == 0] = segm[inst == 0]

    # process normal
    normal = torch.zeros(base_img.size())
    if opt.feat_normal:
        try:
            normal_path = os.path.join(opt.edit_dir, edit_target.replace('.png', '-normal.png'))
            normal_map = Image.open(normal_path).convert('RGB')
            normal = transform_B(normal_map) + 1 / 255  # bias caused by 0..256 instead 0..255
        except FileNotFoundError:  # no cars
            normal = torch.zeros(base_img.size())

    # process feat
    inst_list = np.unique(inst.cpu().numpy().astype(int))
    for inst_id in inst_list:
        inst_id = int(inst_id)
        if inst_id not in feat_dict:
            print('error inst_id = %d !' % inst_id)
            continue
        indices = (inst == inst_id).nonzero()  # n x 3
        for j in range(opt.feat_num):
            feat[indices[:, 0] + j, indices[:, 1], indices[:, 2]] = feat_dict[inst_id][j]

    generated = model.fake_inference(
        base_img.unsqueeze(0), segm.unsqueeze_(0), inst.unsqueeze_(0),
        feat.unsqueeze_(0), pose.unsqueeze_(0), normal.unsqueeze_(0))

    visuals = [('input_label', util.tensor2label(segm[0], opt.label_nc)),
               ('input_inst', util.tensor2label(inst[0], opt.label_nc))]
    if opt.feat_pose:
        visuals += [('input_pose', util.tensor2label(pose[0], opt.feat_pose_num_bins))]
    if opt.feat_normal:
        visuals += [('input_normal', util.tensor2im(normal[0]))]
    visuals += [('synthesized_image', util.tensor2im(generated.data[0])),
                ('target_image', util.tensor2im(target_img)),
                ('source_image', util.tensor2im(base_img))]
    visuals += [('feat', util.tensor2im(feat[0][:3]))]
    visuals = OrderedDict(visuals)
    losses.append(loss(generated.cpu().data[0], target_img))
    img_path = ['%05d.png' % i]
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
