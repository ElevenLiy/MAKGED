#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# 基础配置
lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=/seu_share/home/qiguilin/220224345/kg-error/chinese-alpaca-2-7b-hf
chinese_tokenizer_path=/seu_share/home/qiguilin/220224345/kg-error/alpaca_tokenizer
dataset_dir=/seu_share/home/qiguilin/220224345/kg-error/dataset/wn18rr-train
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=8
max_seq_length=512
validation_file=/seu_share/home/qiguilin/220224345/kg-error/dataset/wn18rr-dev/wn18rr_dev.json

train_subgraph_embedding_file=/seu_share/home/qiguilin/220224345/kg-error/embedding/subgraph_embeddings_wn18rr_train.txt
val_subgraph_embedding_file=/seu_share/home/qiguilin/220224345/kg-error/embedding/subgraph_embeddings_wn18rr_dev.txt

deepspeed_config_file=ds_zero2_no_offload.json


# 定义子图类型数组
    # "Head entity as head"
    # "Head entity as tail"
    # "Tail entity as head"
    # "Tail entity as tail"
subgraph_types=(
    "Tail entity as head"
    "Tail entity as tail"
)

# 循环训练每种子图类型的模型
for subgraph_type in "${subgraph_types[@]}"; do
    # 设置输出目录（基于子图类型）
    output_dir="/seu_share/home/qiguilin/220224345/kg-error/output_lora_wn18rr_${subgraph_type// /_}_$(date +%Y%m%d)_instruction"
    
    # 设置日志文件路径（基于子图类型）
    LOG_FILE="/seu_share/home/qiguilin/220224345/kg-error/log/wn18rr-${subgraph_type// /_}_$(date +%Y%m%d)_instruction.log"
    
    echo "Starting training for subgraph type: $subgraph_type"
    echo "Output directory: $output_dir"
    echo "Log file: $LOG_FILE"
    
    # 运行训练命令
    torchrun --nnodes 1 --nproc_per_node 8 run_sft_kg_instruction.py \
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
        --num_train_epochs 5 \
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
        --subgraph_type "${subgraph_type}" \
        2>&1 | tee -a "$LOG_FILE"
    
    echo "Finished training for subgraph type: $subgraph_type"
    echo "--------------------------------------------"
done

echo "All training completed."