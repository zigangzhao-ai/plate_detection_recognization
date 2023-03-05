from onnx_tf.backend import prepare
import onnx

TF_PATH = "weights/LPRNet_Simplified.pb" # where the representation of tensorflow model will be stored
ONNX_PATH = "weights/LPRNet_Simplified.onnx" # path to my existing ONNX model
onnx_model = onnx.load(ONNX_PATH)  # load onnx model
tf_rep = prepare(onnx_model)  # creating TensorflowRep object
tf_rep.export_graph(TF_PATH)

import tensorflow as tf

TF_PATH = "weights/LPRNet_Simplified.pb" 
TFLITE_PATH = "weights/LPRNet_Simplified.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tf_lite_model = converter.convert()
with open(TFLITE_PATH, 'wb') as f:
    f.write(tf_lite_model)

