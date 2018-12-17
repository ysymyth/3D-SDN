### PATH SETTINGS
export PYTHONPATH=$PWD/geometric:$PWD/geometric/bulb:$PYTHONPATH
export WORKING_DIR_ROOT=$PWD/models

### CUPY
export CUDA_PATH=/usr/local/cuda-9.0
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export CUPY_CACHE_DIR=/tmp/.cupy/$(hostname) 

### NEURAL_RENDERER
export NEURAL_RENDERER_UNSAFE=1

### DATASET PATHS
export SHAPENET_ROOT_DIR=$PWD/geometric/assets
export VKITTI_ROOT_DIR=$PWD/datasets/vkitti
