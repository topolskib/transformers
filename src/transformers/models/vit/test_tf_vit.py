from transformers import ViTConfig, TFViTModel
import tensorflow as tf

config = ViTConfig()
model = TFViTModel(config)

# TF Conv2D op currently only supports the NHWC tensor format on CPU.
# See https://github.com/onnx/onnx-tensorflow/issues/535
# So we provide channels as last dimension, as we are testing this on CPU.  
pixel_values = tf.random.normal((1,224,224,3))

outputs = model(input_ids=None, pixel_values=pixel_values)

print("Shape of last hidden states:")
print(outputs.last_hidden_state.shape)

for param in model.trainable_weights:
    print(param.name, param.shape)