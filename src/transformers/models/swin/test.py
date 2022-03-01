from transformers import SwinConfig, SwinForSemanticSegmentation
import torch

config = SwinConfig(embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path_rate=0.3)

model = SwinForSemanticSegmentation(config)

pixel_values = torch.randn(1, 3, 512, 512)

outputs = model(pixel_values)