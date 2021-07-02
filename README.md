# SMP

Source code for "[Know what you don't need: Single-Shot Meta-Pruning for attention heads](https://www.sciencedirect.com/science/article/pii/S2666651021000140)".

## Requirements

```
    pip install -r requirements.txt
```

## Pruner Training

To train the SMP, we select seven datasets from GLUE (MNLI, QQP, SST-2, CoLA, STS-B, MRPC, RTE) as the training data. `GLUE_DATA` is the GLUE directory. `OUTPUT_DIR` is the directory of SMP checkpoints. The packaged data is available [here](https://cloud.tsinghua.edu.cn/f/08d653c7134b4c94b5e7/).

```
python3 train_all.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_lower_case \
  --data_dir GLUE_DATA \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 60 \
  --learning_rate 2e-2 \
  --output_dir OUTPUT_DIR \
  --max_steps 6000 \
  --gradient_accumulation_steps 8
```

## Pruning and Fine-tuning

Before fine-tuning, we use SMP to identify the redundant attention heads. `TASK_NAME` and `TASK_DATA` are the name and the data path of downstream tasks. `PRUNER_CKPT` is the pruner model checkpoint path. `MASK_DIR` is the attention head mask path.

```
python3 cal.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name TASK_NAME \
        --do_cal \
        --do_lower_case \
        --data_dir TASK_DATA \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 20 \
        --pruner-ckpt PRUNER_CKPT \ 
        --output_dir MASK_DIR \
        --prune-ratio 0.5
```

Using the generated attention masks, we can efficiently conduct the fine-tuning and inference of PLMs. `OUTPUT_DIR` is the output model directory. `MASK_PATH` is the attention mask path.

```
python3 run_glue_prune.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --task_name TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir TASK_DATA \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --output_dir  OUTPUT_DIR \
  --eval_all_checkpoints \
  --head_mask  MASK_PATH
```

## Pruner Checkpoints

The pruner checkpoint used in the experiments is available at `checkpoints`. `base.bin` is trained on BERT-Base and `large.bin` is trained on BERT-Large.

## Cite

```
@article{ZHANG202136,
    title = {Know what you don't need: Single-Shot Meta-Pruning for attention heads},
    journal = {AI Open},
    volume = {2},
    pages = {36-42},
    year = {2021},
    issn = {2666-6510},
    doi = {https://doi.org/10.1016/j.aiopen.2021.05.003},
    url = {https://www.sciencedirect.com/science/article/pii/S2666651021000140},
    author = {Zhengyan Zhang and Fanchao Qi and Zhiyuan Liu and Qun Liu and Maosong Sun}
    }
```
