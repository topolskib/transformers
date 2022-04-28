import torch

from transformers import ResNetConfig, ResNetForImageClassification


pixel_values = torch.randn(1, 3, 224, 224)

config = ResNetConfig(replace_stride_with_dilation=[False, False, True])
# config = ResNetConfig()
model = ResNetForImageClassification(config)

for name, param in model.named_parameters():
    print(name, param.shape)

outputs = model(pixel_values)
