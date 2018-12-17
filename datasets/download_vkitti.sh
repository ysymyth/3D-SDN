#!/bin/bash

echo "Downloading Virtual KITTI dataset..."

mkdir ./datasets/vkitti
cd ./datasets/vkitti

URL=http://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_motgt.tar.gz
TAR_FILE=vkitti_1.3.1_motgt.tar.gz
TARGET_DIR=./vkitti_1.3.1_motgt/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./
rm $TAR_FILE

URL=http://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_extrinsicsgt.tar.gz
TAR_FILE=vkitti_1.3.1_extrinsicsgt.tar.gz
TARGET_DIR=./vkitti_1.3.1_extrinsicsgt/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./
rm $TAR_FILE

URL=http://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_scenegt.tar
TAR_FILE=vkitti_1.3.1_scenegt.tar
TARGET_DIR=./vkitti_1.3.1_scenegt/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -xvf $TAR_FILE -C ./
rm $TAR_FILE

URL=http://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti_1.3.1_rgb.tar
TAR_FILE=vkitti_1.3.1_rgb.tar
TARGET_DIR=./vkitti_1.3.1_rgb/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -xvf $TAR_FILE -C ./
rm $TAR_FILE

echo "Complete downloading Virtual KITTI!"
cd ..
