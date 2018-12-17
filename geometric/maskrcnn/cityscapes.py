import torch
import os
import time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import zipfile
import urllib.request
import shutil
from config import Config
import utils
import model as modellib
from scipy.misc import imread, imsave
from torchvision import transforms
from PIL import Image
import json

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "pretrained/mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class CityscapesConfig(Config):
    # Give the configuration a recognizable name
    NAME = "cityscapes"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 8

    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 2


############################################################
#  Dataset
############################################################

class CityscapesDataset(utils.Dataset):
    def load_cityscapes(self, dataset_dir, subset):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, test)
        """

        self.inst_dir = os.path.join(dataset_dir, 'gtFine', subset)

        # preprocess split list: filter out images without objects
        self.list = json.load(open("{}/annotations/instancesonly_gtFine_{}.json".format(dataset_dir, subset)))['images']
        havecar_name = "{}/annotations/instanceonly_gtFine_{}_have_car.json".format(dataset_dir, subset)
        if not os.path.exists(havecar_name):
            havecar = []
            for image_id in range(len(self.list)):
                if self.load_mask(image_id):
                    havecar.append(image_id)
            json.dump(havecar, open(havecar_name, "w"))
        havecar = json.load(open(havecar_name))
        self.list = [self.list[i] for i in havecar]

        # image jittering
        self.img_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.2)

        # Add classes
        self.add_class("cityscapes", 1, "car")

        # Add images
        for item in self.list:
            self.add_image(
                "cityscapes", image_id=item['id'],
                path=os.path.join(dataset_dir, 'images', item['file_name']),
                width=item['width'], height=item['height'])

        print("#{}_dataset = {}".format(subset, len(self.list)))
        self.subset = subset

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        inst_file = self.list[image_id]['seg_file_name']
        inst_map = imread(os.path.join(self.inst_dir, inst_file.split('_')[0], inst_file))
        ids, counts = np.unique(inst_map, return_counts=True)
        ids = ids[counts > 50]             # filter out objects with area < 50 pixels
        ids = ids[ids // 1000 == 26]           # filter out non-cars
        if len(ids) == 0:
            return False
        masks = np.stack([(inst_map == x) for x in ids], axis=-1)
        class_ids = np.ones(len(ids))
        return masks, class_ids

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image and jitter
        image = Image.open(self.image_info[image_id]['path'])
        if self.subset == 'train':
            image = self.img_jitter(image)
        image = np.array(image)

        # per inst augment
        if self.subset == 'train':
            inst_file = self.list[image_id]['seg_file_name']
            inst_map = imread(os.path.join(self.inst_dir, inst_file.split('_')[0], inst_file))
            noise = np.zeros(image.shape)
            for inst in np.unique(inst_map):
                if np.random.rand() < 0.3:  # w.p. 0.3
                    noise[inst_map == inst, :] = np.random.randint(-20, 20, (3,))
            image = image + noise
            # if np.random.rand() < 0.1:
            #    imsave(os.path.join('./tmp', inst_file), image)
        return image

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Cityscapes.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test' on cityscapes")
    parser.add_argument('--dataset', default='./dataset/cityscapes',
                        metavar="/path/to/cityscapes/",
                        help='Directory of the cityscapes dataset')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        default="",
                        help="Path to weights .pth file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Cityscapes: now only support training
    assert args.command == 'train'

    # Configurations
    if args.command == "train":
        config = CityscapesConfig()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    if config.GPU_COUNT:
        model = model.cuda()

    # Select weights file to load
    args.transfer = 0
    if args.model:
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
            args.transfer = 1
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
        else:
            model_path = args.model
    else:
        model_path = ""

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, transfer=args.transfer)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CityscapesDataset()
        dataset_train.load_cityscapes(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CityscapesDataset()
        dataset_val.load_cityscapes(args.dataset, "val")
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        if args.transfer:
            # Training - Stage 0
            print("Training for transferring num_classes")
            model.train_model(dataset_train, dataset_val,
                              learning_rate=1e-5,
                              epochs=10,
                              layers="transfer")
            # layers=r"(mask.conv5.*)|(classifier.linear_class.*)|(classifier.linear_bbox.*)")

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE,
                          epochs=40,
                          layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE / 2,
                          epochs=70,
                          layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE / 5,
                          epochs=100,
                          layers='all')
