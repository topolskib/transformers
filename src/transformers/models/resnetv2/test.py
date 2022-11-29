import torch

from transformers import ResNetv2Config, ResNetv2ForImageClassification


model = ResNetv2ForImageClassification(ResNetv2Config())

for name, param in model.named_parameters():
    print(name, param.shape)

outputs = model(torch.randn(1, 3, 224, 224))
