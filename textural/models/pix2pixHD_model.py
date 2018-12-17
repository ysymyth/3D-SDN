# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none':  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        self.no_global_encoder = opt.no_global_encoder
        input_nc = opt.label_nc if opt.label_nc != 0 else 3

        # define networks
        self.model_names = []

        # Generator network
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        if opt.feat_pose:
            netG_input_nc += opt.feat_pose_num_bins + 1 if opt.feat_pose_num_bins else 2
        if opt.feat_normal:
            netG_input_nc += 3
        if opt.feat_depth:
            netG_input_nc += 1
        if not opt.no_global_encoder:
            netG_input_nc += opt.global_encoder_nz
        print('netG_input_nc:', netG_input_nc)

        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        self.model_names += ['G']

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            self.model_names += ['D']

        # Encoder network
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids, isTrain=opt.isTrain)
            self.model_names += ['E']

        self.print_networks(verbose=True)

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            # pretrained_path = opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)
            if not self.no_global_encoder:
                self.load_network(self.netGlobalE, 'GlobalE', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake', 'G_L1', 'E_VAE', 'E_regress']

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params': [value], 'lr':opt.lr}]
                    else:
                        params += [{'params': [value], 'lr':0.0}]
            else:
                params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            if not opt.no_global_encoder:
                params += list(self.netGlobalE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer global E
            # if not opt.no_global_encoder:
            #    params = list(self.netGlobalE.parameters())
            #    self.optimizer_E = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, pose_map=None, normal_map=None, depth_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        with torch.no_grad():
            input_label = Variable(input_label)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())
            # create one-hot vector for pose
            if self.opt.feat_pose:
                if self.opt.feat_pose_num_bins:  # quantize
                    size = pose_map.size()
                    oneHot_size = (size[0], self.opt.feat_pose_num_bins + 1, size[2], size[3])
                    _pose_map = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
                    pose_map = _pose_map.scatter_(1, pose_map.data.long().cuda(), 1.0)
                    pose_map = Variable(pose_map)
                else:                           # cos, sin
                    pose_map = Variable(pose_map.data.cuda())
            if self.opt.feat_normal:
                normal_map = Variable(normal_map.data.cuda())
            if self.opt.feat_depth:
                depth_map = Variable(depth_map.data.cuda())

        return input_label, inst_map, real_image, feat_map, pose_map, normal_map, depth_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, image, feat, pose=None, normal=None, depth=None, infer=False):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map, pose_map, normal_map, depth_map = self.encode_input(label, inst, image, feat, pose, normal, depth)

        # Fake Generation
        input_concat = input_label
        if self.use_features:
            if not self.opt.load_features:          # instance-wise code
                feat_map, loss_E_VAE = self.netE.forward(real_image, inst_map)
                input_concat = torch.cat((input_concat, feat_map), dim=1)
            if self.opt.feat_pose:                  # add pose code
                input_concat = torch.cat((input_concat, pose_map), dim=1)
            if self.opt.feat_normal:                # add normal code
                input_concat = torch.cat((input_concat, normal_map), dim=1)
            if self.opt.feat_depth:                 # add depth code
                input_concat = torch.cat((input_concat, depth_map), dim=1)
            if not self.opt.no_global_encoder:      # global code
                mu, logvar = self.netGlobalE.forward(real_image)
                std = logvar.mul(0.5).exp_()
                eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
                z = eps.mul(std).add_(mu)
                z = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), real_image.size(2), real_image.size(3))
                input_concat = torch.cat((input_concat, z), dim=1)
        fake_image = self.netG.forward(input_concat)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat

        # L1 reconstruction loss
        loss_G_L1 = 0
        if self.opt.lambda_L1 > 0:
            loss_G_L1 = self.criterionFeat(fake_image, real_image) * self.opt.lambda_L1

        # global and instance encoder VAE loss and regress loss
        loss_E_regress = 0
        if not self.opt.no_global_encoder:
            kl_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).mul_(-0.5)
            loss_E_VAE = loss_E_VAE + torch.sum(kl_element)
        loss_E_VAE = loss_E_VAE * self.opt.lambda_KL
        if isinstance(loss_E_VAE, float):
            loss_E_VAE = 0

        # loss_E_regress: not implemented

        # Only return the fake_B image if necessary to save BW
        # Added losses with global encoder: loss_E_VAE, loss_E_regress
        return [[loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_G_L1, loss_E_VAE, loss_E_regress], None if not infer else fake_image]

    def fake_inference(self, image, label, inst, feat=None, pose=None, normal=None, depth=None):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map, pose_map, normal_map, depth_map = self.encode_input(Variable(label), inst_map=Variable(inst), pose_map=Variable(pose), normal_map=Variable(normal), depth_map=Variable(depth), real_image=Variable(image), infer=True)

        # Fake Generation
        if self.use_features:
            if feat is None:                        # instance-wise code
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                feat_map = Variable(feat.cuda())
            input_concat = torch.cat((input_label.cuda(), feat_map), dim=1)
            if self.opt.feat_pose:                  # add pose code
                input_concat = torch.cat((input_concat, pose_map), dim=1)
            if self.opt.feat_normal:                  # add normal code
                input_concat = torch.cat((input_concat, normal_map), dim=1)
            if self.opt.feat_depth:
                input_concat = torch.cat((input_concat, depth_map), dim=1)
            if not self.opt.no_global_encoder:      # global code
                mu, logvar = self.netGlobalE.forward(Variable(image.cuda()))
                std = logvar.mul(0.5).exp_()
                eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
                z = eps.mul(std).add_(mu)
                z = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), image.size(2), image.size(3))
                z = mu.view(mu.size(0), mu.size(1), 1, 1).expand(mu.size(0), mu.size(1), image.size(2), image.size(3))
                input_concat = torch.cat((input_concat, z), dim=1)
        else:
            input_concat = input_label
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def inference(self, label, inst):
        # Encode Inputs
        input_label, inst_map, _, _ = self.encode_input(Variable(label), Variable(inst), infer=True)

        # Fake Generation
        if self.use_features:
            # sample clusters from precomputed features
            feat_map = self.sample_features(inst_map)
            feat_map = Variable(feat_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst):
        # read precomputed feature clusters
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = torch.cuda.FloatTensor(1, self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 5000 else i // 5000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])

                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2], idx[:, 3]] = feat[cluster_idx, k]
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 5000 else i // 5000
            i = int(i)
            idx = (inst == i).nonzero()
            num = idx.size()[0]
            idx = idx[num // 2, :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)
        if not self.no_global_encoder:
            self.save_network(self.netGlobalE, 'GlobalE', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
