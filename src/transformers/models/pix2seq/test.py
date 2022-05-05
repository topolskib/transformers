from transformers import Pix2SeqConfig, Pix2SeqModel

model = Pix2SeqModel(Pix2SeqConfig())

for name, param in model.named_parameters():
    print(name, param.shape)