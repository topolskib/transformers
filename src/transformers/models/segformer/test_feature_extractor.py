from transformers import SegFormerFeatureExtractor
from PIL import Image
import requests
import torch

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = SegFormerFeatureExtractor()

image = torch.randn((3, 256, 304))

inputs = feature_extractor(images=image, return_tensors="pt")

print(inputs["pixel_values"].shape)