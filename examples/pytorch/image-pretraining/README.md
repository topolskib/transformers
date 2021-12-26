<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Image pretraining examples

This directory contains a script, `run_mim.py`, that can be used to pre-train any Vision Transformer in the library (concretly, any model supported by the `AutoModelForMaskedImageModeling` API) for masked image modeling (as proposed in [SimMIM: A Simple Framework for Masked Image Modeling](https://arxiv.org/abs/2111.09886)) using PyTorch.

The goal for the model is to predict raw pixel values for the masked patches. The model is trained using a simple L1 loss.

## Using datasets from 🤗 `datasets`

Here we show how to pre-train a `ViT` from scratch for masked image modeling on the [cifar10](https://huggingface.co/datasets/cifar10) dataset. To pre-train a `BEiT` model (which is a Vision Transformer with relative position embeddings instead of absolute ones) from scratch, just change the `model_type` argument to "beit". 

Alternatively, one can decide to further pre-train an already pre-trained (or fine-tuned) checkpoint from the [hub](https://huggingface.co/). This can be done by setting the `model_name_or_path` argument to "google/vit-base-patch16-224-in21k" for example (and not specifying the `model_type` argument).

```bash
python run_mim.py \
    --model_type vit \
    --output_dir ./outputs/ \
    --remove_unused_columns False \
    --label_names bool_masked_pos \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337
```

## Using your own data

To use your own dataset, the training script expects the following directory structure:

```bash
root/dog/xxx.png
root/dog/xxy.png
root/dog/[...]/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/[...]/asd932_.png
```

Once you've prepared your dataset, you can can run the script like this:

```bash
python run_mim.py \
    --model_type vit \
    --dataset_name nateraw/image-folder \
    --train_dir <path-to-train-root> \
    --output_dir ./outputs/ \
    --remove_unused_columns False \
    --label_names bool_masked_pos \
    --do_train \
    --do_eval
```

### 💡 The above will split the train dir into training and evaluation sets
  - To control the split amount, use the `--train_val_split` flag.
  - To provide your own validation split in its own directory, you can pass the `--validation_dir <path-to-val-root>` flag.


## Sharing your model on 🤗 Hub

0. If you haven't already, [sign up](https://huggingface.co/join) for a 🤗 account

1. Make sure you have `git-lfs` installed and git set up.

```bash
$ apt install git-lfs
$ git config --global user.email "you@example.com"
$ git config --global user.name "Your Name"
```

2. Log in with your HuggingFace account credentials using `huggingface-cli`

```bash
$ huggingface-cli login
# ...follow the prompts
```

3. When running the script, pass the following arguments:

```bash
python run_mim.py \
    --push_to_hub \
    --push_to_hub_model_id <name-of-your-model> \
    ...
```