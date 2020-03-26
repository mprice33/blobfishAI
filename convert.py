"""
This is a file used for the conversion of Keras models to onnx model format
The input_keras_model and output_onnx_model files will need to be altered for the given conversion
Currently does not work with Lambda layers/Custom Layers in Keras, for that use the combination of the other convert files created
"""
import os
import onnxmltools
import keras2onnx
from keras.models import load_model


# Update the input name and path for your Keras model
input_keras_model = 'simple_zoo_imageAug_with_Transfer_learning.h5'

# Change this path to the output name and path for the ONNX model
output_onnx_model = 'simple_zoo_imageAug_with_Transfer_learning.onnx'

# Load your Keras model
keras_model = load_model(input_keras_model)

# Convert the Keras model into ONNX
onnx_model = onnxmltools.convert_keras(keras_model)

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, output_onnx_model)