# System libs
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Third party libs
import torch
from torch.autograd import Variable
from collections import OrderedDict
import json

# our libs
from util import util, html
from util.visualizer import Visualizer
from models.models import create_model as create_pix2pix_model
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
data_loader = CreateDataLoader(opt)
print('#testing images = %d' % len(data_loader))

dataset = data_loader.load_data()
model = create_pix2pix_model(opt)
visualizer = Visualizer(opt)

# create website
name = '%s_%s_%s' % (opt.phase, opt.which_epoch, opt.experiment_name)
if opt.how_many > 0:
    name += '_%d_imgs' % opt.how_many
if opt.segm_precomputed_path:
    name += '_predicted'
web_dir = os.path.join(opt.checkpoints_dir, opt.name, name)
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch), verbose=True)

# attach results of other experiments, aligned by each line
exps = []
if exps:
    webpage.attach(exps)

# test
loss = torch.nn.L1Loss()
losses = []
paths = []
for i, data in enumerate((dataset)):
    if i >= opt.how_many:
        break

    generated = model.fake_inference(data['image'], data['label'], data['inst'], pose=data['pose'], normal=data['normal'], depth=data['depth'])

    visuals = [('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
               ('input_inst', util.tensor2label(data['inst'][0], opt.label_nc))]
    if opt.feat_pose:
        visuals += [('input_pose', util.tensor2label(data['pose'][0], opt.feat_pose_num_bins))]
    if opt.feat_normal:
        visuals += [('input_normal', util.tensor2im(data['normal'][0]))]
    if opt.feat_depth:
        visuals += [('input_depth', util.tensor2im(data['depth'][0]))]
    visuals += [('real_image', util.tensor2im(data['image'][0])),
                ('%s_%s_%s' % (opt.phase, opt.which_epoch, opt.experiment_name), util.tensor2im(generated.data[0])), ]
    visuals = OrderedDict(visuals)
    losses += [loss(Variable(data['image']), generated.cpu()).data[0]]
    path = data['path'][0]
    paths += [path]
    img_path = [data['path'][0].replace('.png', '-%04d.png' % i)]
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

json.dump(paths, open(os.path.join(opt.checkpoints_dir, opt.name, 'list.json'), 'w'))
avg_loss = sum(losses) / float(len(losses))
print('avg:', avg_loss)
webpage.add_header(str(avg_loss))
webpage.save()
print(sum(losses) / float(len(losses)))
