import torch

from transformers import SegFormerConfig, SegFormerForImageSegmentation

config = SegFormerConfig()

model = SegFormerForImageSegmentation(config)

pixel_values = torch.randn((2, 3, 512, 512))

outputs = model(pixel_values)

print(outputs.logits.shape)

# print(outputs.last_hidden_state.shape)
