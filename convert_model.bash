#! /bin/bash

# set this path to your matconvnet root directory
matconvnet_root=/Users/moshan/Documents/Research/deep_learning/matconvnet-1.0-beta23	

# use the import-caffe.py script as converter
converter="python $matconvnet_root/utils/import-caffe.py"

# set paths to caffe model prototxt and weights
model_prototxt=/Users/moshan/Documents/Research/deep_learning/featurevis-master/deploy.prototxt
model_weights=/Users/moshan/Documents/Research/deep_learning/featurevis-master/render4cnn_3dview.caffemodel

# set destination for matconvnet model 
import_dir=/Users/moshan/Documents/Research/deep_learning/featurevis-master

# create destination if it doesn't exist
mkdir -pv "$import_dir"

# set the name of the converted matconvnet network file
out="$import_dir/render4cnn-matconvnet.mat"

# run the converter
$converter \
    --output-format=simplenn \
    --caffe-variant=caffe \
    --caffe-data=$model_weights \
    $model_prototxt \
    $out
