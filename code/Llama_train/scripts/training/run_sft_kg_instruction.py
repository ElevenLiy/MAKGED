import logging
from logging.handlers import RotatingFileHandler
import math
import os
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Sequence, Dict, Union, List, Optional
from typing import Optional, List, Union
from pathlib import Path
import datasets
import torch
from datasets import load_dataset
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
from peft.tuners.lora import LoraLayer

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import numpy as np
import json

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

PROMPT_TEMPLATE = (
    "[INST] <<SYS>>\n"
    "You are a helpful assistant. Based on your knowledge, is the following triplet correct? \n"
    "<</SYS>>\n\n{instruction} [/INST]"
)

from transformers import TrainerCallback

class LoggingCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 这里将日志信息记录到文件中
        if logs is not None:
            self.logger.info(logs)

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "sft_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)





def setup_logging():
    log_file = os.environ.get('LOG_FILE', 'output.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除已存在的所有 handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger




@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    # Add this new field
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

@dataclass
class DataTrainingArguments:
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_seq_length: Optional[int] = field(default=512)
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    subgraph_embedding_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the training subgraph embedding file"}
    )
    val_subgraph_embedding_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the validation subgraph embedding file"}
    )
    subgraph_type: Optional[str] = field(
        default="Head entity as head",
        metadata={"help": "Type of subgraph to use for embedding"}
    )

class SubgraphEmbeddingProcessor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

class CustomModel(nn.Module):
    def __init__(self, base_model, subgraph_embedding_dim, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.subgraph_embedding_dim = subgraph_embedding_dim
        self.hidden_size = hidden_size
        self._config = base_model.config

        # 调整嵌入层
        old_embed = self.base_model.get_input_embeddings()
        new_embed = nn.Embedding(old_embed.num_embeddings, old_embed.embedding_dim + subgraph_embedding_dim)
        new_embed.weight.data[:, subgraph_embedding_dim:] = old_embed.weight.data
        self.base_model.set_input_embeddings(new_embed)

        # 调整模型的隐藏层大小
        self._config.hidden_size += subgraph_embedding_dim

        # 调整所有需要匹配hidden_size的层
        for layer in self.base_model.model.layers:
            layer.input_layernorm = nn.LayerNorm(self._config.hidden_size)
            layer.post_attention_layernorm = nn.LayerNorm(self._config.hidden_size)
            layer.self_attn.q_proj = nn.Linear(self._config.hidden_size, layer.self_attn.head_dim * layer.self_attn.num_heads, bias=False)
            layer.self_attn.k_proj = nn.Linear(self._config.hidden_size, layer.self_attn.head_dim * layer.self_attn.num_heads, bias=False)
            layer.self_attn.v_proj = nn.Linear(self._config.hidden_size, layer.self_attn.head_dim * layer.self_attn.num_heads, bias=False)
            layer.self_attn.o_proj = nn.Linear(layer.self_attn.head_dim * layer.self_attn.num_heads, self._config.hidden_size, bias=False)
            layer.mlp.gate_proj = nn.Linear(self._config.hidden_size, layer.mlp.gate_proj.out_features, bias=False)
            layer.mlp.up_proj = nn.Linear(self._config.hidden_size, layer.mlp.up_proj.out_features, bias=False)
            layer.mlp.down_proj = nn.Linear(layer.mlp.down_proj.in_features, self._config.hidden_size, bias=False)

        # 更新最后的归一化层
        self.base_model.model.norm = nn.LayerNorm(self._config.hidden_size)

        # 调整输出层
        self.base_model.lm_head = nn.Linear(self._config.hidden_size, self.base_model.lm_head.out_features, bias=False)

    
    @property
    def config(self):
        return self._config


    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, embeddings):
        return self.base_model.set_input_embeddings(embeddings)
    def resize_token_embeddings(self, new_num_tokens):
        self.base_model.resize_token_embeddings(new_num_tokens)
        # 更新配置
        self._config.vocab_size = new_num_tokens

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def gradient_checkpointing_enable(self):
        return self.base_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        return self.base_model.gradient_checkpointing_disable()


    def forward(self, input_ids=None, attention_mask=None, subgraph_embedding=None, labels=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            
        if subgraph_embedding is not None:
            # 将子图嵌入拼接到整个prompt的前面
            batch_size, seq_len, _ = inputs_embeds.shape
            subgraph_expanded = subgraph_embedding.unsqueeze(1).expand(batch_size, 1, -1)
            inputs_embeds = torch.cat([subgraph_expanded, inputs_embeds], dim=1)

        # 使用拼接后的嵌入进行前向传播
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs

@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable: Optional[str] = field(default="q_proj,v_proj")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_alpha: Optional[float] = field(default=32.)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    use_flash_attention_2: Optional[bool] = field(default=False)
    double_quant: Optional[bool] = field(default=True)
    quant_type: Optional[str] = field(default="nf4")
    load_in_kbits: Optional[int] = field(default=16)
    full_finetuning: Optional[bool] = field(default=False)

def read_embeddings(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip().split('\n\n')
    
    embeddings = []
    for group in content:
        group_embeddings = [list(map(float, line.split())) for line in group.split('\n')]
        embeddings.append(group_embeddings)
    
    return embeddings

def build_instruction_dataset(
    data_path: Union[List[str], str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int,
    data_cache_dir = None,
    preprocessing_num_workers = None,
    subgraph_embeddings = None,
    subgraph_type_index = 0
):
    def tokenization(examples):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE
        for instruction, input, output in zip(examples['instruction'], examples['input'], examples['output']):
            if input is not None and input != "":
                instruction = instruction + '\n' + input
            source = prompt.format_map({'instruction': instruction})
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        tokenized_sources = tokenizer(sources, return_attention_mask=False)
        tokenized_targets = tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        all_embeddings = []

        for i, (s, t) in enumerate(zip(tokenized_sources['input_ids'], tokenized_targets['input_ids'])):
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            
            if subgraph_embeddings is not None:
                all_embeddings.append(torch.tensor(subgraph_embeddings[i][subgraph_type_index]))

        results = {'input_ids': all_input_ids, 'labels': all_labels}
        if subgraph_embeddings is not None:
            results['subgraph_embedding'] = [torch.tensor(emb, dtype=torch.float32) for emb in all_embeddings]
        return results

    logging.warning("building dataset...")
    all_datasets = []

    if not isinstance(data_path, (list, tuple)):
        data_path = [data_path]
    for file in data_path:
        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
        cache_path = os.path.join(data_cache_dir, os.path.basename(file).split('.')[0] + f"_{max_seq_length}")
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'training datasets-{file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=["instruction", "input", "output"],
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    all_datasets = datasets.concatenate_datasets(all_datasets)
    return all_datasets

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        
        if all("subgraph_embedding" in instance for instance in instances):
            subgraph_embeddings = [instance["subgraph_embedding"] for instance in instances]
            result["subgraph_embedding"] = torch.stack(subgraph_embeddings)
        
        return result

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        subgraph_embedding = inputs.pop("subgraph_embedding", None)
        
        if isinstance(model, CustomModel):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                subgraph_embedding=subgraph_embedding,
                labels=inputs["labels"]
            )
        else:
            outputs = model(**inputs)

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def main():
    logger = setup_logging()

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )

    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)

    logger.info(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
                f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}")
    logger.info(f"Training/evaluation parameters {training_args}")

    log_level = training_args.get_process_log_level()

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path)

    subgraph_embeddings = None
    if data_args.subgraph_embedding_file:
        subgraph_embeddings = read_embeddings(data_args.subgraph_embedding_file)
        logger.info(f"Loaded subgraph embeddings from {data_args.subgraph_embedding_file}")

    subgraph_types = ['Head entity as head', 'Head entity as tail', 'Tail entity as head', 'Tail entity as tail']
    subgraph_type_index = subgraph_types.index(data_args.subgraph_type)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset = None
    train_dataset = None

    if training_args.do_train:
        with training_args.main_process_first(desc="loading and tokenization"):
            path = Path(data_args.dataset_dir)
            files = [os.path.join(path, file.name) for file in path.glob("*.json")]
            train_dataset = build_instruction_dataset(
                data_path=files,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                data_cache_dir=None,
                preprocessing_num_workers=data_args.preprocessing_num_workers,
                subgraph_embeddings=read_embeddings(data_args.subgraph_embedding_file),
                subgraph_type_index=subgraph_type_index
        )
        logger.info(f"Num train_samples {len(train_dataset)}")
        logger.info("Training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))

    if training_args.do_eval:
        with training_args.main_process_first(desc="loading and tokenization"):
            files = [data_args.validation_file]
            logger.info(f"Evaluation files: {' '.join(files)}")
            eval_dataset = build_instruction_dataset(
                data_path=files,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                data_cache_dir=None,
                preprocessing_num_workers=data_args.preprocessing_num_workers,
                subgraph_embeddings=read_embeddings(data_args.val_subgraph_embedding_file),
                subgraph_type_index=subgraph_type_index
        )
        logger.info(f"Num eval_samples {len(eval_dataset)}")
        logger.info("Evaluation example:")
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # if training_args.load_in_kbits in [4, 8]:
    #     load_in_4bit = training_args.load_in_kbits == 4
    #     load_in_8bit = training_args.load_in_kbits == 8
    #     if training_args.modules_to_save is not None:
    #         load_in_8bit_skip_modules = training_args.modules_to_save.split(',')
    #     else:
    #         load_in_8bit_skip_modules = None
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=load_in_4bit,
    #         load_in_8bit=load_in_8bit,
    #         llm_int8_threshold=6.0,
    #         load_in_8bit_skip_modules=load_in_8bit_skip_modules,
    #         bnb_4bit_compute_dtype=compute_dtype,
    #         bnb_4bit_use_double_quant=training_args.double_quant,
    #         bnb_4bit_quant_type=training_args.quant_type
    #     )
    # else:
    #     load_in_4bit = False
    #     load_in_8bit = False
    #     quantization_config = None
    if training_args.load_in_kbits in [4, 8]:
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=training_args.load_in_kbits == 4,
        load_in_8bit=training_args.load_in_kbits == 8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=training_args.double_quant,
        bnb_4bit_quant_type=training_args.quant_type
    )
    else:
        quantization_config = None

    if quantization_config is not None:
        logger.info(f"quantization_config:{quantization_config.to_dict()}")
    
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    base_model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=AutoConfig.from_pretrained(model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        quantization_config=quantization_config,
        # use_flash_attention_2=training_args.use_flash_attention_2
    )
    # 获取子图嵌入的维度
    subgraph_embeddings = read_embeddings(data_args.subgraph_embedding_file)
    subgraph_embedding_dim = len(subgraph_embeddings[0][0])

    model = CustomModel(base_model, subgraph_embedding_dim, base_model.config.hidden_size)
    model.train()  # 将模型设置为评估模式


    # if training_args.load_in_kbits in [4, 8]:
    #     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    model.config.use_cache = False
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"Model vocab size: {model_vocab_size}")
    logger.info(f"len(tokenizer):{len(tokenizer)}")
    if model_vocab_size != len(tokenizer):
        logger.info(f"Resize model vocab size to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))




    if not training_args.full_finetuning:
        if training_args.peft_path is not None:
            logger.info("Peft from pre-trained model")
            model = PeftModel.from_pretrained(model, training_args.peft_path, device_map=device_map)
        else:
            logger.info("Init new peft model")
            target_modules = training_args.trainable.split(',')
            modules_to_save = training_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            lora_rank = training_args.lora_rank
            lora_dropout = training_args.lora_dropout
            lora_alpha = training_args.lora_alpha
            logger.info(f"target_modules: {target_modules}")
            logger.info(f"lora_rank: {lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=lora_rank, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                modules_to_save=modules_to_save
            )
            model = get_peft_model(model, peft_config)
        
        model.print_trainable_parameters()
        
        model.print_trainable_parameters()
        logger.info(f"model.modules_to_save: {model.modules_to_save}")
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    if not training_args.full_finetuning and training_args.gradient_checkpointing and \
        (not model.modules_to_save or 'embed_tokens' not in model.modules_to_save):
        if hasattr(model.base_model, "enable_input_require_grads"):
            model.base_model.enable_input_require_grads()
        elif hasattr(model.base_model, "get_input_embeddings"):
            def make_inputs_require_grad(_module, _input, _output):
                _output.requires_grad_(True)
            model.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LoggingCallback(logger)]
    )
    trainer.add_callback(SavePeftModelCallback)

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        trainer.save_model()
        logger.info(f"Training metrics: {metrics}")  # 添加日志输出

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info(f"Evaluation metrics: {metrics}")  # 添加日志输出

if __name__ == "__main__":
    main()