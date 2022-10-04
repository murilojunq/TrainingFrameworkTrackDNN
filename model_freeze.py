'''
Produce a protobuf file that contains the model architecture and the model in one and can be read into CMSSWs
TensorFlow API. This is a fragile part of the code mostly due to my limited understanding combined with the
poor documentation of C++ API.

It seems that the input node(s) get the name 'x' and 'y' while the output node is 'Identity'. These need to be
set in the CMSSW end when evaluating the network. To figure out what names TF decides to give the nodes, one
can use Tensorflows summarize_graph tool in graph_transforms, following instructions at
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms

:param model: trained model to store

Frozen graph is written to the directory storing the results of the training run

NOTES: As tensorflow can't handle control flow operations in a graph when converting variables to constants for creating the
protobuf file, one has launch a new clean process that executes this script to avoid problems in trying to save the model
when it was trained with multiple GPUs (physical or logical).

I dont really understand the details, but this seems to be the only way I can get it to work within the same process.
ModelManager.save_model() takes care of launching the independent process to run this script with the correct arguments.

This implementation may be very fragile. It tries to detect the correct python executable you are running to use this program
in order to use it to launch this process as well, but I really can't say how robust this is.
'''


from os import environ
environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def main(result_folder_path):
    path = "{}/model".format(result_folder_path)
    model = tf.keras.models.load_model(path)
    freeze_model(model, result_folder_path)

def freeze_model(model, result_folder_path):
    # Written for model with two input nodes. If using only one node, adjust by modifying the following five lines.
    model_function = tf.function(lambda x, y: model([x, y]))
    #model_function = tf.function(lambda x: model(x))
    model_function = model_function.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype),
        tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype)
    )

    frozen_function = convert_variables_to_constants_v2(model_function)
    graph = frozen_function.graph.as_graph_def()

    tf.io.write_graph(graph_or_graph_def=graph,
                      logdir=result_folder_path,
                      name="frozen_graph.pb",
                      as_text=False)

model_path = sys.argv[1]
main(model_path)
