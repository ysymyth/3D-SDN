import os
import sys
import numpy as np
import json
from config import Config
import utils
import model as modellib
from scipy.misc import imread
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from datasets.vkitti_utils import get_tables, get_lists

# Root directory of the project
ROOT_DIR = os.path.dirname(__file__)

# Path to trained weights file
VKITTI_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_vkitti.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class VKittiConfig(Config):
    # Give the configuration a recognizable name
    NAME = "vkitti"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 8

    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 3


############################################################
#  Dataset
############################################################

class VKittiDataset(utils.Dataset):
    def load_vkitti(self, dataset_dir, subset):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, test)
        """

        if subset == 'val':
            subset = 'test'

        self.image_dir = os.path.join(dataset_dir, "vkitti_1.3.1_rgb")
        self.label_dir = os.path.join(dataset_dir, "vkitti_1.3.1_scenegt")
        self.table_inst = get_tables('inst', dataset_dir)

        # preprocess split list: filter out images without objects
        self.list_images = json.load(open(os.path.join(
            os.path.dirname(__file__), "assets/vkitti_maskrcnn_{}.json".format(subset))))

        self.img_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.1)

        # Add classes
        for i, vechicle_type in enumerate(["car", "van"]):
            self.add_class("vkitti", i + 1, vechicle_type)

        # Add images
        for i in range(len(self.list_images)):
            self.add_image(
                "vkitti", image_id=i,
                path=os.path.join(self.image_dir, self.list_images[i]),
                width=1242, height=375)

        print("#{}_dataset = {}".format(subset, len(self.list_images)))

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        inst_map = imread(os.path.join(self.label_dir, self.list_images[image_id]))
        worldId, sceneId = self.list_images[image_id].split('/')[:2]
        inst_map = np.apply_along_axis(
            lambda a: self.table_inst[(worldId, sceneId, a[0], a[1], a[2])], 2, inst_map)
        ids, counts = np.unique(inst_map, return_counts=True)
        ids = ids[counts > 50]          # filter out objects with area < 50 pixels
        ids = ids[ids > 5000]           # filter out non-vechicles
        ids = ids[ids // 5000 != 11]      # filter out trucks
        assert len(ids) > 0, self.list_images[image_id]
        masks = np.stack([(inst_map == x) for x in ids], axis=-1)
        class_ids = np.fromiter(map(lambda x: {2: 1, 12: 2}[x // 5000], ids), dtype=np.int)  # dont care about trucks no more
        return masks, class_ids

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = Image.open(self.image_info[image_id]['path'])
        image = self.img_jitter(image)
        image = np.array(image)
        return image

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Virtual KITTI.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test' on vkitti")
    parser.add_argument('--dataset', default='./dataset',
                        metavar="/path/to/vkitti/",
                        help='Directory of the vkitti dataset')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        default="",
                        help="Path to weights .pth file or 'vkitti'")
    parser.add_argument('--pretrain_dir', required=False,
                        metavar="/path/to/weights/dir",
                        default="./pretrained",
                        help="Path to weights dir")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    # parser.add_argument('--transfer', action='store_true',
    #                     help='transfer coco model to new model')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Vkitti: we only need training
    assert args.command == 'train'

    # Configurations
    if args.command == "train":
        config = VKittiConfig()
        # Path to pretrained imagenet model
        config.IMAGENET_MODEL_PATH = os.path.join(args.pretrain_dir, "resnet50_imagenet.pth")
        # Path to pretrained coco model
        config.COCO_MODEL_PATH = os.path.join(args.pretrain_dir, "mask_rcnn_coco.pth")

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    if config.GPU_COUNT:
        model = model.cuda()

    # Select weights file to load
    args.transfer = 0
    if args.model:
        if args.model.lower() == "vkitti":
            model_path = VKITTI_MODEL_PATH
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = config.IMAGENET_MODEL_PATH
            args.transfer = 1
        elif args.model.lower() == "coco":
            # Start from COCO trained weights
            model_path = config.COCO_MODEL_PATH
            args.transfer = 1
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
        dataset_train = VKittiDataset()
        dataset_train.load_vkitti(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = VKittiDataset()
        dataset_val.load_vkitti(args.dataset, "val")
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
