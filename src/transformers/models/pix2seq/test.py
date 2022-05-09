import torch

from transformers import Pix2SeqConfig, Pix2SeqForConditionalGeneration


model = Pix2SeqForConditionalGeneration(Pix2SeqConfig())

pixel_values = torch.randn(1,3,640,640)

outputs = model.generate(pixel_values)

print("Outputs:", outputs)