#Revert TF to 1.4.0 to run this conversion to make a TF frozen graph. (.pb)
#Return to TF 2.0.0 once you've generated the model. Output name can be viewed within netron.
#Adjust the model names for conversion, input/output to run script


import tensorflow as tf
import tf2onnx
import winmltools
from tensorflow.core.framework import graph_pb2


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and return it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

graph = load_graph("simple_zoo_imageAug_with_Transfer_learningTest.pb")
for op in graph.get_operations():
    print(op.name, op.outputs)


output_names = ['output:0']

graph_def = graph_pb2.GraphDef()
with open("simple_zoo_imageAug_with_Transfer_learning.pb", 'rb') as file:
  graph_def.ParseFromString(file.read())
g = tf.import_graph_def(graph_def, name='')

with tf.Session(graph=g) as sess:
  converted_model = winmltools.convert_tensorflow(sess.graph, 7, output_names=['dense_3/Softmax:0'])
  winmltools.save_model(converted_model, 'simple_zoo_imageAug_with_Transfer_learning.onnx')
