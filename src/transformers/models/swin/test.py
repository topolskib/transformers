from transformers import SwinConfig, SwinModel
import torch

config = SwinConfig()
model = SwinModel(config)

pixel_values = torch.randn(1, 3, 224, 224)

outputs = model(pixel_values, output_hidden_states=True)

for i in outputs.reshaped_hidden_states:
    print(i.shape, i[0,0,:3,:3])