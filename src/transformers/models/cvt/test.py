import torch
from PIL import Image

import requests
from transformers import CvtConfig, CvtModel, ViTFeatureExtractor


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

encoding = feature_extractor(image, return_tensors="pt")

config = CvtConfig()
model = CvtModel(config)

outputs = model(encoding.pixel_values)

print("Outputs shape:", outputs.last_hidden_state.shape)
