from transformers import ResNetConfig, DeformableDetrConfig, DeformableDetrForObjectDetection

backbone_config = ResNetConfig()

config = DeformableDetrConfig(use_timm_backbone=False, backbone_config=backbone_config,
                              two_stage=True, with_box_refine=True,
                              backbone=None, dilation=None,)
model = DeformableDetrForObjectDetection(config)

for name, param in model.named_parameters():
    print(name, param.shape)