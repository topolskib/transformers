from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection, ResNetConfig


backbone_config = ResNetConfig.from_pretrained("microsoft/resnet-50", out_features=["stage2", "stage3", "stage4"])

config = DeformableDetrConfig(
    use_timm_backbone=False,
    backbone_config=backbone_config,
    two_stage=True,
    with_box_refine=True,
)
model = DeformableDetrForObjectDetection(config)

for name, param in model.state_dict().items():
    print(name, param.shape)
