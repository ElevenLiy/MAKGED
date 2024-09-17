#!/bin/bash

lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

# 设置日志文件路径
export LOG_FILE="/kg-error/log/fb15k-Head_entity_as_tail.log"

pretrained_model=/kg-error/chinese-alpaca-2-7b-hf
chinese_tokenizer_path=/kg-error/alpaca_tokenizer
dataset_dir=/kg-error/dataset/fb15k-train
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=8
max_seq_length=512
output_dir=/kg-error/output_lora_fb15k_Head_entity_as_tail_0813
validation_file=/dataset/fb15k-dev/fb15k_dev.json

# 新增：子图嵌入文件路径和子图类型
train_subgraph_embedding_file=/kg-error/embedding/subgraph_embeddings_fb15k_train.txt
val_subgraph_embedding_file=/kg-error/embedding/subgraph_embeddings_fb15k_dev.txt
# subgraph_type="Head entity as head"
subgraph_type="Head entity as tail"
# subgraph_type="Tail entity as head"
# subgraph_type="Tail entity as tail"


deepspeed_config_file=ds_zero2_no_offload.json

torchrun --nnodes 1 --nproc_per_node 8 run_sft_kg.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --load_in_kbits 16 \
    --save_safetensors False \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --subgraph_embedding_file ${train_subgraph_embedding_file} \
    --val_subgraph_embedding_file ${val_subgraph_embedding_file} \
    --subgraph_type "${subgraph_type}"
