import torch

from transformers import SegFormerConfig, SegFormerForImageSegmentation


config = SegFormerConfig(
    image_size=64,
    num_channels=3,
    num_encoder_blocks=4,
    depths=[2, 2, 2, 2],
    sr_ratios=[8, 4, 2, 1],
    hidden_sizes=[16, 32, 64, 128],
    downsampling_rates=[1, 4, 8, 16],
    num_attention_heads=[1, 2, 4, 8],
    is_training=True,
)

model = SegFormerForImageSegmentation(config)

pixel_values = torch.randn((13, 3, 64, 64))

outputs = model(pixel_values, output_hidden_states=True)

print("Shape of logits:", outputs.logits.shape)

# print(outputs.last_hidden_state.shape)
