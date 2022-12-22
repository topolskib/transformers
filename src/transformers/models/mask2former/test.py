from transformers import SwinConfig, Mask2FormerDecoderConfig, Mask2FormerConfig, Mask2FormerForUniversalSegmentation
import torch

backbone_config = SwinConfig(embed_dim = 128, window_size=12, depths = (2, 2, 18, 2), num_heads = (4, 8, 16, 32), out_features=["stage1", "stage2", "stage3", "stage4"])
decoder_config = Mask2FormerDecoderConfig(decoder_layers=9)

config = Mask2FormerConfig(backbone_config=backbone_config, decoder_config=decoder_config)

model = Mask2FormerForUniversalSegmentation(config)

for name, param in model.named_parameters():
    print(name, param.shape)

outputs = model(torch.randn(1, 3, 384, 384))