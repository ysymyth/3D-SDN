# The textural branch of 3D-SDN

## Prerequisites
To run the textural branch, we need to precompute results from the semantic and geometric branch. 
- The semantic segmentation results from **the semantic branch**, stored at `$VKITTI_SEMANTIC_PRECOMPUTED_DIR`.
- The instance map, pose and normal map results from **the geometric branch**, stored at `$VKITTI_GEOMETRIC_PRECOMPUTED_DIR`. 

## Train
For Virtual KITTI, run
```
python train.py \
    --name vkitti \
    --gpu_ids ${GPU_IDS} \
    --dataroot ${VKITTI_ROOT_DIR} \
    --segm_precomputed_path ${VKITTI_SEMANTIC_PRECOMPUTED_DIR} \
    --inst_precomputed_path ${VKITTI_GEOMETRIC_PRECOMPUTED_DIR} \
    --feat_normal ${VKITTI_GEOMETRIC_PRECOMPUTED_DIR} \
    --feat_pose ${VKITTI_GEOMETRIC_PRECOMPUTED_DIR} \
```
Note that as the Virtual KITTI dataset is small, by default we add **random cropping** on top of **image augmentation** to address overfitting. 
Please refer to options `--resize_or_crop` and `--use_augmentation`. 

## Test
For testing, we reconstruct images in the test set.

For Virtual KITTI, run
```
python test.py \
    --name vkitti \
    --gpu_ids ${GPU_IDS} \
    --dataroot ${VKITTI_ROOT_DIR} \
    --segm_precomputed_path ${VKITTI_SEMANTIC_PRECOMPUTED_DIR} \
    --inst_precomputed_path ${VKITTI_GEOMETRIC_PRECOMPUTED_DIR} \
    --feat_normal ${VKITTI_GEOMETRIC_PRECOMPUTED_DIR} \
    --feat_pose ${VKITTI_GEOMETRIC_PRECOMPUTED_DIR} \
    --which_epoch 60 \
    --how_many ${HOW_MANY} \
```
