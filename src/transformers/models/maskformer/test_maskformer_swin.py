from transformers.models.maskformer import MaskFormerSwinConfig, MaskFormerSwinBackbone
import torch

config = MaskFormerSwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
backbone = MaskFormerSwinBackbone(config)

pixel_values = torch.randn(1, 3, 224, 224)

outputs = backbone(pixel_values)