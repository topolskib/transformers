import torch

from transformers import Pix2SeqFeatureExtractor


PADDING_TOKEN = 0
eos_token_weight = 0.1

feature_extractor = Pix2SeqFeatureExtractor()

# suppose you have an image, with bounding boxes of shape (1, 100, 4) and labels of shape (1, 100):
bbox = torch.stack(
    [
        torch.randn(
            4,
        )
    ]
    * 3
).unsqueeze(0)
# fake labels are set to vocab.FAKE_CLASS_TOKEN - vocab.BASE_VOCAB_SHIFT
# because later on, when encoding the target sequence, we shift all labels by vocab.BASE_VOCAB_SHIFT
label = torch.tensor([[0, 5, -70]])

print("Shape of bbox:", bbox.shape)
print("Shape of label:", label.shape)

response_seq, response_seq_class_m, token_weights = feature_extractor.build_response_seq_from_bbox(bbox, label)

print("Response seq:", response_seq)
print("Token weights:", token_weights)

# Assign lower weights for ending/padding tokens.
token_weights = torch.where(
    response_seq == PADDING_TOKEN, torch.zeros_like(token_weights) + eos_token_weight, token_weights
)

print("Token weights after assigning lower weight to padding:", token_weights)

logits = torch.randn(1, 50, 3000)

pred_class, pred_bbox, pred_score = feature_extractor.decode_object_seq_to_bbox(
    pred_seq=response_seq,
    logits=logits,
)
