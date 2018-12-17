#!/bin/bash

NVCC_CUDA9="-arch=sm_30 \
-gencode=arch=compute_30,code=sm_30 \
-gencode=arch=compute_35,code=sm_35 \
-gencode=arch=compute_37,code=sm_37 \
-gencode=arch=compute_50,code=sm_50 \
-gencode=arch=compute_52,code=sm_52 \
-gencode=arch=compute_60,code=sm_60 \
-gencode=arch=compute_61,code=sm_61 \
-gencode=arch=compute_70,code=sm_70 \
-gencode=arch=compute_70,code=compute_70"

cd geometric/maskrcnn 

cd nms/src/cuda/ 
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC $NVCC_CUDA9
cd ../../ 
python build.py 
cd ../ 

cd roialign/roi_align/src/cuda/ 
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC $NVCC_CUDA9
cd ../../ 
python build.py 
cd ../../

cd ../../
