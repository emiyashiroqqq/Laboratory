import os
from tensorflow.python import pywrap_tensorflow

import tensorflow as tf


checkpoint_path = 'G:/CKPTFile/vgg_16_2016_08_28/vgg_16.ckpt'
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    # print(reader.get_tensor(key))