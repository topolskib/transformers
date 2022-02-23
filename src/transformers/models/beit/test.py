from transformers import BeitConfig, BeitForSemanticSegmentation
import torch

config = BeitConfig(image_size=224)

model = BeitForSemanticSegmentation(config)

pixel_values = torch.randn(1,3,224,224)

outputs = model(pixel_values)