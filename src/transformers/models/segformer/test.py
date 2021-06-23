from transformers import SegFormerConfig, SegFormerModel
import torch

config = SegFormerConfig()

model = SegFormerModel(config)

pixel_values = torch.randn((2,3,224,224))

outputs = model(pixel_values)

for i in outputs.encoder_hidden_states:
    print(i.shape)

# for name, param in model.named_parameters():
#     print(name, param.shape)