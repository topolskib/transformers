from transformers import OPTConfig, InstructBlipConfig, InstructBlipForConditionalGeneration
import torch

text_config = OPTConfig().to_dict()
config = InstructBlipConfig(text_config=text_config)

model = InstructBlipForConditionalGeneration(config=config)

pixel_values = torch.randn(1, 3, 224, 224)
input_ids = torch.tensor([[1,2]])

outputs = model(pixel_values, input_ids)

print("Output keys:", outputs.keys())