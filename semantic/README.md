# The semantic branch of 3D-SDN

The semantic segmentation model is based on [Semantic Segmentation on MIT ADE20K dataset in PyTorch](https://github.com/CSAILVision/semantic-segmentation-pytorch). We use the architecture `ResNet50_dilated8 + ppm_bilinear_deepsup`.

## Prerequisites

- Note that the training only runs with **more than 1 GPU**.
```bash
conda activate 3dsdn
source scripts/env.sh # or  make sure $VKITTI_ROOT_DIR is specified
cd semantic
```

## Usage
For Virtual KITTI. You can adapt to other datasets easily.
### Train
```bash
CUDA_VISIBLE_DEVICES=${GPU_IDS} python vkitti_train.py \
    --id vkitti \
    --num_gpus ${NUM_GPUS} \
    --root_dataset ${VKITTI_ROOT_DIR} \
```

### Evaluation
```bash
python vkitti_eval.py \
    --id ${MODEL_ID} \
    --gpu_id ${GPU_ID} \
    --root_dataset ${VKITTI_ROOT_DIR} \
```
``
### Test
```bash
python vkitti_test.py \
    --id ${MODEL_ID} \
    --gpu_id ${GPU_ID} \
    --root_dataset ${VKITTI_ROOT_DIR} \
    --result ${VKITTI_SEMANTIC_PRECOMPUTED_DIR} \
    --test_img ${IMG_ID} \
```
Notes:
1. By default, `VKITTI_SEMANTIC_PRECOMPUTED_DIR` is `semantic/results`.
Please update `VKITTI_SEMANTIC_PRECOMPUTED_DIR` in `env.sh` after testing if needed, to provide the textural branch with precomputed semantic maps.

2. If ``--test_img`` is not specified, by default the model runs testing on the whole dataset.
