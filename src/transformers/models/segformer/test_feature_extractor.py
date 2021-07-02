import torch
from PIL import Image

import requests
from transformers import SegFormerFeatureExtractor


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = SegFormerFeatureExtractor(do_random_crop=False)

image = torch.randn((3, 256, 304))

image = torch.randn((3, 256, 2048))

inputs = feature_extractor(images=image, return_tensors="pt")

print(inputs["pixel_values"].shape)

feature_extractor = SegFormerFeatureExtractor(image_scale=(1024, 2048), do_random_crop=False)

image = torch.randn((3, 1024, 2048))

inputs = feature_extractor(images=image, return_tensors="pt")

print(inputs["pixel_values"].shape)
