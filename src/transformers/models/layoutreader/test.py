from transformers import LayoutReaderConfig, LayoutReaderForPreTraining


config = LayoutReaderConfig()
model = LayoutReaderForPreTraining(config)

for name, param in model.named_parameters():
    print(name, param.shape)
