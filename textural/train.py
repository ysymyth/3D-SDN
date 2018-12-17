# System libs
import time
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Third party libs
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict

# our libs
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from util import util
from util.visualizer import Visualizer
from models.models import create_model as create_pix2pix_model


def main():
    # Dealing with options
    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10

    # Data loader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    # pix2pix model
    pix2pix_model = create_pix2pix_model(opt)
    visualizer = Visualizer(opt)

    # Training
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        losses_G = []
        losses_D = []
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == 0

        # Forward Pass
            losses, generated = pix2pix_model(
                Variable(data['label']), Variable(data['inst']), Variable(data['image']), Variable(data['feat']),
                Variable(data['pose']), Variable(data['normal']), Variable(data['depth']), infer=save_fake)

            # sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(pix2pix_model.module.loss_names, losses))
            # print(loss_dict)

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG'] + loss_dict['G_L1'] + loss_dict['E_VAE']
            losses_D.append(loss_D.data[0])
            losses_G.append(loss_G.data[0])
            loss_dict['d_total'] = loss_D
            loss_dict['g_total'] = loss_G

            # Backward Pass
            # update generator weights
            pix2pix_model.module.optimizer_G.zero_grad()
            loss_G.backward()
            pix2pix_model.module.optimizer_G.step()

            # update discriminator weights
            pix2pix_model.module.optimizer_D.zero_grad()
            loss_D.backward()
            pix2pix_model.module.optimizer_D.step()

            # Display results and errors
            # print out errors
            if total_steps % opt.print_freq == 0:
                errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            # display output images
            if save_fake:
                visuals = [('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('input_inst', util.tensor2label(data['inst'][0], opt.label_nc))]
                if opt.feat_pose:
                    visuals += [('input_pose', util.tensor2label(data['pose'][0], opt.feat_pose_num_bins))]
                if opt.feat_normal:
                    visuals += [('input_normal', util.tensor2im(data['normal'][0]))]
                if opt.feat_depth:
                    visuals += [('input_depth', util.tensor2im(data['depth'][0]))]
                visuals += [('synthesized_image', util.tensor2im(generated.data[0])),
                            ('real_image', util.tensor2im(data['image'][0]))]
                visuals = OrderedDict(visuals)
                visualizer.display_current_results(visuals, epoch, total_steps)

            # save latest model
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                pix2pix_model.module.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        # end of epoch
        print('End of epoch %d / %d \t Time Taken: %d sec \t loss_G: %lf loss_D: %lf' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time, np.mean(losses_G), np.mean(losses_D)))
        visualizer.print_current_error(epoch, np.mean(losses_G), np.mean(losses_D))

        # save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            pix2pix_model.module.save('latest')
            pix2pix_model.module.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        # instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            pix2pix_model.module.update_fixed_params()

        # linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            pix2pix_model.module.update_learning_rate()


if __name__ == '__main__':
    main()
