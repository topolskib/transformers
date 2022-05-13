import torch

from transformers import Pix2SeqFeatureExtractor


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
label = torch.tensor([[0, 5, 30]])

print("Shape of bbox:", bbox.shape)
print("Shape of label:", label.shape)

response_seq, response_seq_class_m, token_weights = feature_extractor.build_response_seq_from_bbox(bbox, label)

print("Response seq:", response_seq)

logits = torch.randn(1, 50, 3000)

pred_class, pred_bbox, pred_score = feature_extractor.decode_object_seq_to_bbox(
    pred_seq=response_seq,
    logits=logits,
)

print("Pred class:", pred_class)
