import torch

from transformers import Pix2SeqFeatureExtractor


feature_extractor = Pix2SeqFeatureExtractor()

pred_seq = torch.tensor(
    [
        [
            1113,
            1063,
            1183,
            1274,
            175,
            1121,
            1521,
            1292,
            1578,
            175,
            1040,
            1540,
            1581,
            1999,
            117,
            1000,
            1000,
            1749,
            1999,
            163,
            1085,
            1013,
            1739,
            1494,
            117,
            1000,
            1000,
            1749,
            1999,
            30,
            1083,
            1000,
            1749,
            1999,
            30,
            1114,
            1062,
            1183,
            1274,
            30,
            1041,
            1540,
            1580,
            1999,
            30,
            1121,
            1521,
            1292,
            1578,
            30,
        ]
    ]
)
logits = torch.randn(1, 50, 3000)

pred_class, pred_bbox, pred_score = feature_extractor.decode_object_seq_to_bbox(
    pred_seq=pred_seq,
    logits=logits,
)
