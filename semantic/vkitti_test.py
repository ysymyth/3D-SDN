# System libs
import os
import argparse

# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.io import loadmat
# Our libs
from vkitti_dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2


def precompute_result(info, preds, args):
    img_name = info.split('/')[-1]
    img_path = os.path.join(args.result, info.replace(img_name, ''))
    if not os.path.isdir(img_path):
        os.makedirs(img_path)
    cv2.imwrite(os.path.join(img_path, img_name), preds)


def visualize_result(data, preds, args):
    colors = loadmat(os.path.join(os.path.dirname(__file__), 'data/color150.mat'))['colors']
    (img, info) = data

    # prediction
    pred_color = colorEncode(preds, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=0).astype(np.uint8)
    img_name = info.split('/')[-1]
    img_path = os.path.join(args.result, info.replace(img_name, ''))
    if not os.path.isdir(img_path):
        os.makedirs(img_path)

    cv2.imwrite(os.path.join(img_path, img_name.replace('.png', '_visualize.png')), im_vis)


def test(segmentation_module, loader, args):

    segmentation_module.eval()

    for i, batch_data in enumerate(loader):
        # process data
        batch_data = batch_data[0]
        img_ori = as_numpy(batch_data['img_ori'])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            segSize = (img_ori.shape[0], img_ori.shape[1])
            pred = torch.zeros(1, args.num_class, segSize[0], segSize[1])
            pred = Variable(pred).cuda()

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, args.gpu_id)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                pred = pred + pred_tmp / len(args.imgSize)

            _, preds = torch.max(pred.data.cpu(), dim=1)
            preds = as_numpy(preds.squeeze(0))

        precompute_result(batch_data['info'], preds, args)
        if args.visualize:
            visualize_result(
                (batch_data['img_ori'], batch_data['info']),
                preds, args)


def main(args):
    torch.cuda.set_device(args.gpu_id)

    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(arch=args.arch_encoder,
                                        fc_dim=args.fc_dim,
                                        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(arch=args.arch_decoder,
                                        num_class=args.num_class,
                                        fc_dim=args.fc_dim,
                                        weights=args.weights_decoder,
                                        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_test = TestDataset(
        args, max_sample=args.num_val)
    loader_test = torchdata.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=4,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    test(segmentation_module, loader_test, args)

    print('Test Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', required=True,
                        help="a name for identifying the model to load")
    parser.add_argument('--suffix', default='_epoch_25.pth',
                        help="which snapshot to load")
    parser.add_argument('--arch_encoder', default='resnet50_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Path related arguments
    parser.add_argument('--root_dataset',
                        default='./data/')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=14, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[100, 150, 200, 300, 375], nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g.  300 400 500 600')
    parser.add_argument('--imgMaxSize', default=1242, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')

    # Misc arguments
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--result', default='./result',
                        help='folder to output visualization results')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='gpu_id for evaluation')

    # Added arguments
    parser.add_argument('--test_img', default='all',
                        help='if specified a single image, do single image testing; otherwise do all/train/test/benchmark set inference')
    parser.add_argument('--benchmark_json', default='', help='json path if test_img = benchmark')
    parser.add_argument('--visualize', action='store_true', help='generate visualizations of segmentations')

    args = parser.parse_args()
    print(args)

    # torch.cuda.set_device(args.gpu_id)

    # absolute paths of model weights
    args.weights_encoder = os.path.join(args.ckpt, args.id,
                                        'encoder' + args.suffix)
    args.weights_decoder = os.path.join(args.ckpt, args.id,
                                        'decoder' + args.suffix)
    assert os.path.exists(args.weights_encoder) and \
        os.path.exists(args.weights_encoder), 'checkpoint does not exist!'

    # args.result = os.path.join(args.result, args.id)
    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)
