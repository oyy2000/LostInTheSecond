---
license: other
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
library_name: peft
model-index:
- name: _test_run
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# _test_run

This model is a fine-tuned version of [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) on the gsm8k_3b_sft_normal dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1331

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 4
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1.0

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.0938        | 1.0   | 45   | 0.1331          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.42.3
- Pytorch 2.3.0+cu121
- Datasets 3.2.0
- Tokenizers 0.19.1