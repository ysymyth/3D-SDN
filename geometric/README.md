# The geometric branch of 3D-SDN

This branch consists of two components: MaskRCNN and Derender3D. During testing time, the code will load two components to GPU, which requires more than 8GB memory.

## MaskRCNN

The instance segmentation model is based on [pytorch-mask-rcnn](https://github.com/multimodallearning/pytorch-mask-rcnn).

### Training
We leverage the pre-trained COCO model by first re-training the last few layers.

For Virtual KITTI, run
```bash
python geometric/maskrcnn/vkitti.py train \
    --dataset=./datasets/vkitti \
    --pretrain_dir=./models/vkitti-geometric-maskrcnn/pretrained \
    --model=coco
```

The trained model checkpoint file, by default in `geometric/maskrcnn/logs`, is then used by the 3d derenderer of the geometric branch.

## Derender3D

### Training

The training for 3D derenderer consists of two training phases: pre-training and fine-tuning. Models will be stored in `./models`.

- Pre-training
    ```bash
    python geometric/scripts/main.py \
        --do train \
        --dataset vkitti \
        --mode pretrain \
        --num_epochs 256 \
        --batch_size 64 \
        --lr 1e-3 \
        --lr_decay_epochs 16 \
        --lr_decay_rate 0.5 \
        --working_dir_root ./models \
        --name vkitti-geometric-derender3d-pretrain
    ```

- Fine-tuning
    ```bash
    python geometric/scripts/main.py \
        --do train \
        --dataset vkitti \
        --mode full \
        --num_epochs 256 \
        --batch_size 64 \
        --mask_weight 0.1 \
        --ffd_coeff_reg 10.0 \
        --lr 1e-4 \
        --lr_decay_epochs 16 \
        --lr_decay_rate 0.5 \
        --ckpt_dir ./models/*-vkitti-geometric-derender3d-pretrain \
        --working_dir_root ./models \
        --name vkitti-geometric-derender3d-full
    ```

## Testing
This command loads both modules and feed inputs to both of them.

```bash
python geometric/scripts/main.py \
    --do test \
    --dataset vkitti \
    --mode extend \
    --source maskrcnn \
    --ckpt_dir ./models/vkitti-geometric-derender3d-full \
    --maskrcnn_path ./models/vkitti-geometric-maskrcnn/mask_rcnn_vkitti_0100.pth \
    --output_dir ./imgs/vkitti-test
```
