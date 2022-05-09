from transformers import Pix2SeqConfig, Pix2SeqForConditionalGeneration
import torch

model = Pix2SeqForConditionalGeneration(Pix2SeqConfig())

prompt = torch.tensor([[10]])

outputs = model.generate(prompt)

print("Outputs:", outputs)