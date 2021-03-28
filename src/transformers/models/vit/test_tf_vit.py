from transformers import ViTConfig, TFViTModel
import tensorflow as tf

config = ViTConfig()
model = TFViTModel(config)

# TF Conv2D op currently only supports the NHWC tensor format on CPU.
# See https://github.com/onnx/onnx-tensorflow/issues/535
# So we provide channels as last dimension, as we are testing this on CPU.  
pixel_values = tf.random.normal((1,224,224,3))

# test with dictionary 
inputs = {"input_ids": None, "pixel_values": pixel_values}
outputs = model(inputs)

print("Shape of last hidden states:")
print(outputs.last_hidden_state.shape)

# test with dummy inputs
# not working right now since dummy inputs are set to 30x30
#outputs = model(model.dummy_inputs)

# test with symbolic inputs
input_ids = tf.keras.Input(batch_shape=(2, 512), name="input_ids", dtype="int32")
pixel_values = tf.keras.Input(batch_shape=(2, 224, 224, 3), name="pixel_values", dtype="float32")

outputs_symbolic = model([input_ids, pixel_values])

print("Shape of last hidden states:")
print(outputs_symbolic.last_hidden_state.shape)

# print out all parameters
# for param in model.trainable_weights:
#     print(param.name, param.shape)