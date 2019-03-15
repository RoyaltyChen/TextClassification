import tensorflow as tf
import os

logdir='checkpoints/textcnn'

from tensorflow.python import pywrap_tensorflow
# checkpoint_path = os.path.join(model_dir, "model.ckpt-9999")
ckpt = tf.train.get_checkpoint_state(logdir)
reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)

var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))

import tensorflow.contrib.slim as slim

model_variables = slim.get_variables()
restore_variables = [var for var in model_variables]
for var in restore_variables:
    print(var.name)