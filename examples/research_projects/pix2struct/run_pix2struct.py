import json
import logging
import random
from typing import Any, List
from nltk import edit_distance
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import wandb
from pytorch_lightning.loggers import WandbLogger

import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner

from transformers import Pix2StructForConditionalGeneration, AutoProcessor
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)

# step 1. load model and processor
repo_id = "google/pix2struct-base"
processor = AutoProcessor.from_pretrained(repo_id)
model = Pix2StructForConditionalGeneration.from_pretrained(repo_id)

# step 2. define PyTorch dataset
added_tokens = []

class ImageCaptioningDataset(Dataset):
    def __init__(
        self,
        dataset_name_or_path: str,
        max_patches: int = 1024,
        max_length: int = 512,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.split = split
        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.max_patches = max_patches
        self.max_length = max_length
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key

        self.gt_token_sequences = []
        for ground_truth in self.dataset["ground_truth"]:
            ground_truth = json.loads(ground_truth)
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    )
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )

        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in added_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj
    
    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            model.decoder.resize_token_embeddings(len(processor.tokenizer))
            added_tokens.extend(list_of_tokens)
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # prepare inputs
        encoding = processor(images=item["image"], max_patches=self.max_patches, return_tensors="pt")
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        
        # prepare targets
        target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        input_ids = processor.tokenizer(
            target_sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        labels = input_ids.squeeze().clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        encoding["labels"] = labels
        # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
        return encoding, target_sequence
    

# step 3: create PyTorch datasets
logging.info(f"Creating PyTorch datasets...")
train_dataset = ImageCaptioningDataset("naver-clova-ix/cord-v2",
                                       split="train", sort_json_key=False) # cord dataset is preprocessed, so no need for this
val_dataset = ImageCaptioningDataset("naver-clova-ix/cord-v2",
                                       split="validation", sort_json_key=False) # cord dataset is preprocessed, so no need for this

logging.info(f"Number of tokens added: {len(added_tokens)}")


# step 4: define PyTorch Lightning Module
class Pix2Struct(pl.LightningModule):
    def __init__(self, config, processor, model, batch_size=32):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = batch_size

    def training_step(self, batch, batch_idx):
        encoding, _ = batch
        
        outputs = self.model(**encoding)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        encoding, answers = batch
        flattened_patches, attention_mask = encoding["flattened_patches"], encoding["attention_mask"]
        
        outputs = self.model.generate(flattened_patches=flattened_patches,
                                      attention_mask=attention_mask)
    
        predictions = []
        for seq in self.processor.batch_decode(outputs):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            # seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            predictions.append(seq)

        scores = []
        for pred, answer in zip(predictions, answers):
            # pred = re.sub(r"(?:(?<=>) | (?=", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))
            
            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores)) 
        
        return scores

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, lr=self.config.get("lr"), weight_decay=1e-05)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.config.get("num_warmup_steps"),
                                                    num_training_steps=self.config.get("max_steps"))
        
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=self.batch_size)
    

# step 5: instantiate module
config = {
          "num_warmup_steps": 1000,
          "max_steps": 30000,
          "lr": 0.01,
          "check_val_every_n_epoch": 5,
          "gradient_clip_val": 1.0,
          "verbose": True,
          }

pl_module = Pix2Struct(config, processor, model)

# step 6: define Trainer
logging.info(f"Defining Trainer...")
wandb.finish()
wandb_logger = WandbLogger(project="Pix2Struct", name="demo-run-pix2struct-adafactor-dgx-auto-batch-size")

trainer = pl.Trainer(
        accelerator="gpu",
        devices=[3],
        precision="bf16",
        max_steps=config.get("max_steps"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"), # use gradient clipping
        logger=wandb_logger,
)
tuner = Tuner(trainer)

# Auto-scale batch size by growing it exponentially (default)
tuner.scale_batch_size(pl_module, mode="power")

# step 7: train
logging.info(f"Starting training...")
trainer.fit(pl_module)