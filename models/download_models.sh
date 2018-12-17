#!/bin/bash

echo "Downloading pretrained weights..."

for TARGET in geometric-derender3d geometric-maskrcnn semantic textural
do
    URL=http://3dsdn.csail.mit.edu/assets/vkitti-$TARGET.tar.gz

    echo $URL
    curl $URL | tar -zx -C ./models
done
