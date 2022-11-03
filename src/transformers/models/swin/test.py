from transformers import SwinConfig, SwinForSemanticSegmentation

model = SwinForSemanticSegmentation(SwinConfig())

for name, param in model.named_parameters():
    print(name, param.shape)