# -*- coding: utf-8 -*-
# Copyright 2023 XuMing(xuming624@qq.com) and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

part of this code is adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

PT:自监督任务，输入中南大学 输出：中 ->中南 中南->大 中南大->学

"""
import math
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional, List, Dict, Any, Mapping

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.utils.versions import require_version
from transformers.integrations import is_deepspeed_zero3_enabled


#***********************************************************************************#
# 模型参数
#***********************************************************************************#
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    关于我们将采用哪种模型/配置/分词器进行微调，还是从头开始训练
    """
    """
    model_name_or_path: Optional[str] = field(...)
    model_name_or_path: 变量名
    : Optional[str]: 类型提示（Type Hint）。表示这个变量的值可以是 str（字符串）类型，也可以是 None
    = field(...): 调用 dataclasses.field() 函数。这是 @dataclass 装饰器用来配置字段元数据（如默认值、帮助信息、是否参与比较等）的标准方式，而不是直接赋值（如 = None）。
    """

    # model_name_or_path: 预训练模型的路径或名称
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    # tokenizer_name_or_path: tokenizer的路径
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    # load_in_8bit/4bit: 是否使用8位/4位量化加载模型
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})

    # cache_dir: 预训练模型缓存目录
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    # model_revision:模型版本
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )

    # hf_hub_token: HF 资源库中登录的授权令牌
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Auth token to log in with Hugging Face Hub."})

    # use_fast_tokenizer 分词器版本 基本上仅在fast（Rust版本）报错的情况下使用纯python实现的慢速版本
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    # torch_dtype: 模型数据类型（auto/bfloat16/float16/float32）
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
    # device_map: 模型权重在计算设备（GPU/CPU/磁盘）之间如何分布
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )

    # trust_remote_code: 安全开关，用于控制是否允许加载和执行模型仓库中自定义的 Python 代码
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")

#***********************************************************************************#
# 数据参数
#***********************************************************************************#
@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    关于我们将哪些数据输入模型以进行训练和评估
    """
    # dataset_name: Hugging Face datasets 库中数据集的名称 通常和train_file_dir二选一
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    # dataset_config_name: 数据集的子配置或版本名称
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    # train_file_dir: 本地训练数据文件夹路径
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The train text data file folder."})

    # validation_file_dir: 本地验证数据文件夹路径
    validation_file_dir: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on text file folder."},
    )

    # max_train_samples: 最大训练样本数
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )

    # max_eval_samples : 最大验证/评估样本数
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    # streaming ： 流式模式
    # flase - 一次性将整个数据集下载到本地磁盘并加载到内存。速度快，但需要大硬盘和大内存
    # true -  边读取边处理，不下载完整数据集
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})

    # block_size: 输入序列的最大长度（Token 数量）
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

    # overwrite_cache: 是否覆盖已缓存的处理后数据
    # True: 强制删除旧缓存，重新进行预处理
    # False: 如果之前已经处理过该数据并保存了缓存文件，下次运行直接加载缓存，跳过耗时的预处理
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    # validation_split_percentage: 果没有提供独立的验证集（validation_file_dir 为 None），则从训练集中切分 百分之几 作为验证集
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )

    # preprocessing_num_workers: 数据预处理时使用的多进程 worker 数量
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # keep_linebreaks:  在处理纯文本（.txt）文件时，是否保留换行符
    # True: 保留 \n。这对代码生成、诗歌或需要段落结构的任务很重要
    # False: 移除所有换行符，将文本视为连续流。
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    # packing:  是否启用样本打包
    # True(推荐): 多条短文本拼接在一起，填满一个 block_size 的长度，中间用 EOS (End of Sequence) token 分隔
    # False：     每条样本独立处理，不足block_size的部分用Padding填充
    packing: bool = field(
        default=True,
        metadata={"help": "Whether to pack multiple texts into one sequence for efficient training. "
                          "Texts are concatenated with EOS separator and split into block_size chunks."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

#***********************************************************************************#
# LORA参数
#***********************************************************************************#
@dataclass
class ScriptArguments:
    """
    配置 PEFT (Parameter-Efficient Fine-Tuning，参数高效微调) 和 LoRA/QLoRA 具体参数的
    """
    # 是否启用参数高效微调LORA： 模型权重以 float16 或 bfloat16 加载。精度稍高，显存占用中等
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    # 是否启用QLORA：模型权重先被量化为 4-bit (NF4)，然后再挂载 LoRA 适配器
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
    # 指定模型中哪些层需要应用 LoRA
    target_modules: Optional[str] = field(default="all")
    # LoRA 的秩
    lora_rank: Optional[int] = field(default=8)
    # LoRA 层内部的 Dropout 概率
    lora_dropout: Optional[float] = field(default=0.05)
    # LORA的缩放系数：LoRA 的更新量会乘以lora_rank/lora_alpha
    lora_alpha: Optional[float] = field(default=32.0)
    # LoRA 微调后需要保存的模块
    modules_to_save: Optional[str] = field(default=None)
    # 现有 PEFT 适配器模型的路径，NONE：从头开始训练一个新的 LoRA 适配器；路径：加载一个已经训练好的 LoRA 权重，在此基础上继续训练或直接用于推理。
    peft_path: Optional[str] = field(default=None)


#***********************************************************************************#
# 评价函数
#***********************************************************************************#
# 准确率计算，工具函数，后续具体任务中的指标会调用此函数
def accuracy(predictions, references, normalize=True, sample_weight=None):
    """
    predictions: 模型预测出的标签列表
    references: 真实的标签列表
    sample_weight:  用于给每个样本赋予不同的权重。如果为 None，则所有样本权重相等
    """
    return {
        "accuracy": float(accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight))
    }

# 大语言模型（LLM）生成任务评估
# 计算模型在“下一个词预测”任务中的 Token 级准确率
def compute_metrics(eval_preds):
    '''
    输入序列:    [我]  [爱]  [学]  [习]  [</s>]                │
                  │     │     │     │      │                    │
    模型预测:      ↓     ↓     ↓     ↓      ↓                    │
                [爱]  [学]  [习]  [</s>]  [???] 
    '''
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics, we need to shift the labels
    # 将 labels 移除第一位 展平
    labels = labels[:, 1:].reshape(-1)
    # 将 preds 移除最后一位 展平
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)

# 预处理模型输出 logits
# 将模型输出的原始概率分布（Logits）转换为具体的预测 Token ID（即取最大值对应的索引），分词表中的哪个索引
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


#***********************************************************************************#
# 数据处理函数
#***********************************************************************************#
def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    """
    # 数据批次整理函数（Data Collator），用于在训练过程中将多个独立样本合并成一个批次（batch）
    # 输入：4 个样本
    features = [
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], "labels": [1, 2, 3, 4]},
        {"input_ids": [5, 6, 7, 8], "attention_mask": [1, 1, 1, 1], "labels": [5, 6, 7, 8]},
        {"input_ids": [9, 10, 11, 12], "attention_mask": [1, 1, 1, 1], "labels": [9, 10, 11, 12]},
        {"input_ids": [13, 14, 15, 16], "attention_mask": [1, 1, 1, 1], "labels": [13, 14, 15, 16]},
    ]

    # 调用函数
    batch = fault_tolerance_data_collator(features)

    # 输出：
    batch = {
        "input_ids": tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]),
        "attention_mask": tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
        "labels": tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    }

    """

    #  查第一个样本是否是字典类型，如果不是字典，说明是对象（如 dataclass），用 vars() 转换为字典
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError:  # quick fix by simply take the first example
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch



"""
序列打包
# 将多个文本样本重新分组为固定长度的块，同时保留每个样本的第一个和最后一个token
first +多个中间内容的组合+ last 长度 == max_seq_length
为什么保留每个样本的第一个和最后一个token？ 因为是首位标记

LM 预训练/微调中的数据优化策略，核心目的是提高训练效率和数据利用率
假设 max_seq_length = 512，
样本	 实际长度 传统方式	  GroupTexts 方式
文本1	50	填充到 512	拼接后切分
文本2	80	填充到 512	拼接后切分
文本3	30	填充到 512	拼接后切分
文本4	120	填充到 512	拼接后切分
文本5	60	填充到 512	拼接后切分

传统方式
样本1: [内容50][padding 462]  ← 90% 是 padding          
样本2: [内容80][padding 432]  ← 85% 是 padding          
样本3: [内容30][padding 482]  ← 94% 是 padding         
样本4: [内容120][padding 392] ← 77% 是 padding          
样本5: [内容60][padding 452]  ← 88% 是 padding

GroupTexts 方式
新样本1: [首][内容340中的510个][尾]  ← 几乎无 padding


max_seq_length = 8
builder = GroupTextsBuilder(max_seq_length)

# 输入：2个样本，每个样本5个token
examples = {
    "input_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
}

result = {
    "input_ids": [[1,2,3,4,7,8,9,5]]
}

"""
class GroupTextsBuilder:
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
    # __call__ 方法在将类实例当作函数调用时触发
    def __call__(self, examples):
        # Concatenate all texts.
        firsts = {k: examples[k][0][0] for k in examples.keys()}
        lasts = {k: examples[k][0][-1] for k in examples.keys()}
        contents = {k: sum([vi[1:-1] for vi in v], []) for k, v in examples.items()}
        total_length = len(contents[list(examples.keys())[0]])

        content_length = self.max_seq_length - 2
        if total_length >= content_length:
            total_length = (total_length // content_length) * content_length
        # Split by chunks of max_len.
        result = {
            k: [[firsts[k]] + t[i: i + content_length] + [lasts[k]] for i in range(0, total_length, content_length)] for
            k, t in contents.items()}
        return result


#***********************************************************************************#
# 保存函数
#***********************************************************************************#
class SavePeftModelTrainer(Trainer):
    """
    继承HuggingFace Trainer ，具有 基础训练器，包含训练循环、评估、保存等完整功能
    Trainer for lora models
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        # 第1行：创建保存目录
        os.makedirs(output_dir, exist_ok=True)
        # 第2行：保存训练参数
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        # 第3行：保存模型：只保存 LoRA 适配器权重，不保存完整模型
        self.model.save_pretrained(output_dir)


def save_model(model, tokenizer, args):
    """Save the model and the tokenizer.
        adapter_config.json           LoRA 配置
        pytorch_model.safetensors     LoRA 权重
        
        tokenizer_config.json         tokenizer 配置
        tokenizer.json                保存词表和特殊 token
     
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    # 检查 model 是否有 module 属性
    model_to_save = model.module if hasattr(model, "module") else model
    # 保存模型：只保存 LoRA 适配器权重，不保存完整模型 （传入的是PeftModel对象：model = get_peft_model(model, peft_config)）
    model_to_save.save_pretrained(output_dir)
    # 保存分词器配置和词表
    tokenizer.save_pretrained(output_dir)


def save_model_zero3(model, tokenizer, args, trainer):
    """Save the model for deepspeed zero3.
    #  DeepSpeed ZeRO-3 训练场景下的保存模型
    refer https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L209
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)
    tokenizer.save_pretrained(output_dir)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    输出trainable parameters
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper.
    自动查找模型中所有线性层名称的工具函数，主要用于 LoRA/QLoRA 微调时自动确定 target_modules 参数
    # 目标：获取可复用的模块名，用于 target_modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    输入：
    | 参数 | 类型 | 默认值 | 说明 |
    |------|------|--------|------|
    | `peft_model` | `PeftModel` | 必需 | 已应用 PEFT 的模型 |
    | `int4` | `bool` | `False` | 是否使用 4bit 量化（QLoRA） |
    | `int8` | `bool` | `False` | 是否使用 8bit 量化 |
    输出：
    | 类型 | 说明 |
    |------|------|
    | `List[str]` | 排序后的线性层名称列表，用于 `target_modules` |


    for name, module in model.named_modules():
        print(f"{name}: {type(module)}")
    # 输出示例:
    # "": LlamaForCausalLM
    # "model": LlamaModel
    # "model.layers.0.self_attn": LlamaAttention
    # "model.layers.0.self_attn.q_proj": Linear  ← 目标层
    # "model.layers.0.self_attn.k_proj": Linear  ← 目标层
    # "model.layers.0.self_attn.v_proj": Linear  ← 目标层
    # "model.layers.0.self_attn.o_proj": Linear  ← 目标层
    # "lm_head": Linear  ← 需要排除

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    """
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)

#***********************************************************************************#
# 主函数
#***********************************************************************************#
def main():
    #***********************************************************************************#
    # 参数解析
    #***********************************************************************************#
    # HuggingFace transformers 库中的命令行参数解析工具，用于将多个配置数据类转换为命令行参数
    # 自定模型参数 自定数据参数 HF库为Seq2Seq任务设计的训练参数 自定LORA参数
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))
    # 解析命令行参数并提取前4个数据类实例
    model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )[:4]

    logger.info(f"deepspeed_zero3:{is_deepspeed_zero3_enabled()}")
    
    # Remove the explicit distributed initialization and simplify the process check
    # The Trainer will handle distributed training setup
    # 去除明确的分布式初始化步骤，并简化流程检查
    # 训练器将负责分布式训练的设置工作
    # 如果 local_rank是 -1或 0，返回 True 否则返回 False
    # 返回为1说明是单卡训练local_rank=-1 或分布式主进程 local_rank=1
    is_main_process = training_args.local_rank in [-1, 0]

    # Only log on main process
    # 只有是主进程才log
    if is_main_process:
        logger.info(f"Model args: {model_args}")
        logger.info(f"Data args: {data_args}")
        logger.info(f"Training args: {training_args}")
        logger.info(f"Script args: {script_args}")
        logger.info(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )

    # Set seed before initializing model.
    # 设置随机数种子
    set_seed(training_args.seed)


    #***********************************************************************************#
    # 加载分词器
    #***********************************************************************************#
    # Load tokenizer
    # Hugging Face Transformers 库 加载分词器（Tokenizer）
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,                     # 预训练模型缓存目录
        "use_fast": model_args.use_fast_tokenizer,             # 快速分词器开关
        "trust_remote_code": model_args.trust_remote_code,     # 安全开关：安全开关，用于控制是否允许加载和执行模型仓库中自定义的 Python 代码
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    # AutoTokenizer 类从指定路径加载分词器：将文本拆分成 token（词元/子词） 映射 token 到数字 ID（input_ids）
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    # 训练数据处理时设置序列长度(block_size)
    # model_max_length：模型理论最多支持多长
    # block_size：你训练时实际切多长
    
    # 限制模型一次能处理的最大文本长度
    # tokenizer.model_max_length 是分词器本身支持的最大文本长度
    if data_args.block_size is None:
        #  如果用户未指定，则使用 tokenizer 支持的最大长度 作为默认值
        block_size = tokenizer.model_max_length
        # 如果 tokenizer 的最大长度 超过 2048，则发出警告
        if block_size > 2048:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 2048. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
    else:
        # 超过 tokenizer 支持的最大长度，警告
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        # 取 用户指定值 和 tokenizer 最大长度 中的较小值，确保不会超出限制
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    
    
    #***********************************************************************************#
    # 利用分词器定义数据的预处理函数，通过dataset的map方法使用，主要涉及将文本 -> input_ids 和 lable
    #***********************************************************************************#
    # Preprocessing the datasets.
    # 语言模型训练的数据预处理函数，用于将原始文本转换为模型可训练的格式
    # 在语言建模任务中，模型的目标是根据输入的文本预测每个位置的下一个 token。
    # 在这个过程中，我们不需要额外的标签数据，而是使用输入本身作为标签
    def tokenize_function(examples):
        # examples - 来自数据集的批量样本（字典格式）
        # 分词后的字典，包含 input_ids、attention_mask、labels 等
        # 输入：
        # examples：{"text": ["今天天气真好", "你好世界"]}
        # 输出：
        '''
            {
                'input_ids': [
                    [101, 2769, 2850, 4308, 3221, 2864, 643, 102, 0, 0],  # 对应“今天天气真好”
                    [101, 872, 2776, 4135, 102, 0, 0, 0, 0, 0]           # 对应“你好世界”
                ],
                'attention_mask': [
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # 对应“今天天气真好”，0是填充位，1是实际token
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]   # 对应“你好世界”
                ],
                'labels': [
                    [101, 2769, 2850, 4308, 3221, 2864, 643, 102, 0, 0],  # 用于预测的标签（这里等于input_ids）
                    [101, 872, 2776, 4135, 102, 0, 0, 0, 0, 0]            # 用于预测的标签
                ]
            }
        '''
        
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding='max_length',   # padding方式
            max_length=block_size   # 最大长度约束
        )
        # Copy the input_ids to the labels for language modeling. This is suitable for both
        # masked language modeling (like BERT) or causal language modeling (like GPT).
        # labels = input_ids 用于自监督学习
        # # 将输入的“ids”值复制到“标签”中，用于语言模型训练。这适用于以下两种情况：
        # 1. 嵌入式语言模型（如 BERT）
        # 2. 递归式语言模型（如 GPT）
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()

        return tokenized_inputs

    # 没有截断的数据预处理函数，也没有标签
    # 先 tokenize_wo_pad_function 后 group_text_function
    def tokenize_wo_pad_function(examples):
        return tokenizer(examples["text"])

    # Packing function: concatenate texts with EOS separator, then split into block_size chunks
    # # 包装函数：将文本用“EOS”分隔符连接起来，然后分成block_size大小的块。
    def group_text_function(examples):
        """
        Packing implementation: concatenate texts with EOS separator, then split into block_size chunks
        [doc1_ids][EOS][doc2_ids][EOS][doc3_ids][EOS]... → chunks of block_size
        """
        """
        它的输入是 tokenize_function 的输出
       {
                'input_ids': [
                    [101, 2769, 2850, 4308, 3221, 2864, 643, 102, 0, 0],  # 对应“今天天气真好”
                    [101, 872, 2776, 4135, 102, 0, 0, 0, 0, 0]           # 对应“你好世界”
                ],
                'attention_mask': [
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # 对应“今天天气真好”，0是填充位，1是实际token
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]   # 对应“你好世界”
                ],
                'labels': [
                    [101, 2769, 2850, 4308, 3221, 2864, 643, 102, 0, 0],  # 用于预测的标签（这里等于input_ids）
                    [101, 872, 2776, 4135, 102, 0, 0, 0, 0, 0]            # 用于预测的标签
                ]
            }
        它的作用是 把短序列拼接成大序列再切成固定长度 block
        1. 把多个短序列（比如句子）拼接成一个大序列，并在每个序列后面加上 EOS（句子结束标记）；
        2. 然后按固定长度 block_size 切分成小块，用于模型训练（如语言模型的 next-token 预测）
        输出
        result = {
            "input_ids": [
                [101, 2769, 2850, 4308, 3221, 2864, 643, 102],  # 第一块
                 # 第二块不满 8，被丢弃
            ],
            "attention_mask": [
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            "labels": [
                [101, 2769, 2850, 4308, 3221, 2864, 643, 102],
            ]
        }

        """
        # 获取 EOS token id，如果 tokenizer 没有定义 EOS，就用 pad token 代替
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = tokenizer.pad_token_id  # fallback
        
        # Concatenate all texts with EOS separator
        # # 将所有文本用“EOS”分隔符连接起来
        all_input_ids = []
        all_attention_mask = []
        for i, ids in enumerate(examples["input_ids"]):
            all_input_ids.extend(ids)
            # Ensure each document ends with EOS
            if len(ids) == 0 or ids[-1] != eos_token_id:
                all_input_ids.append(eos_token_id)
            if "attention_mask" in examples:
                all_attention_mask.extend(examples["attention_mask"][i])
                if len(ids) == 0 or ids[-1] != eos_token_id:
                    all_attention_mask.append(1)
        
        total_length = len(all_input_ids)
        # Drop the small remainder
        # 如果总长度不是 block_size 的整数倍，丢弃最后不足一个 block 的部分。
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        
        # Split by chunks of block_size
        # 将 all_input_ids 按 block_size 切分成小块，并存到 result["input_ids"]。
        result = {
            "input_ids": [all_input_ids[i: i + block_size] for i in range(0, total_length, block_size)],
        }
        # 同样地，把 attention_mask 也按 block_size 切分。
        if all_attention_mask:
            result["attention_mask"] = [all_attention_mask[i: i + block_size] for i in range(0, total_length, block_size)]
        
        result["labels"] = result["input_ids"].copy()
        return result

    
    #***********************************************************************************#
    # 加载数据
    #***********************************************************************************#
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # 获取数据集：你可以提供你自己的 CSV/JSON/TXT 格式的训练和评估文件（见下文）
    # 或者直接提供 Hugging Face 数据集中心（https://huggingface.co/datasets/）上公开数据集的名字
    # （数据集会被自动下载）。
    #
    # 对于 CSV/JSON 文件，本脚本会使用名为 'text' 的列，如果没有名为 'text' 的列，则使用第一列。
    # 你可以很容易地修改这个行为（见下文）。
    #
    # 在分布式训练中，load_dataset 函数保证只有一个本地进程会同时下载数据集。
    # 数据处理：
    '''
        返回的结果 raw_datasets 是一个 字典，通常包含不同的数据集划分：
        "train" → 训练集
        "validation" → 验证集（如果有的话）
        "test" → 测试集（如果有的话）
        典型类型是 DatasetDict。
    '''
    if data_args.dataset_name is not None: # 从HF平台下载并加载数据集
        # Downloading and loading a dataset from the hub.
        # 调用 Hugging Face datasets 库的 load_dataset 函数，从 Hugging Face Hub 或本地加载数据集。
        # raw_datasets
        '''
            返回的结果 raw_datasets 是一个 字典，通常包含不同的数据集划分：
            "train" → 训练集
            "validation" → 验证集（如果有的话）
            "test" → 测试集（如果有的话）
            典型类型是 DatasetDict。
        '''
        raw_datasets = load_dataset(
            data_args.dataset_name,                   # 想要加载的数据集名称
            data_args.dataset_config_name,            # 有些数据集提供多个配置（configuration）, 如果没有配置，可以传 None
            cache_dir=model_args.cache_dir,           # 指定下载数据集的本地缓存目录
            streaming=data_args.streaming,            # 流式开关 开则数据集不会一次性加载到内存，而是按需从磁盘或网络流式读取
        )
        # 没有验证集的情况下，从训练集中切分出验证集
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",  # train[:10%]取前10%
                cache_dir=model_args.cache_dir,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",   # train[10%:]取后90%
                cache_dir=model_args.cache_dir,
                streaming=data_args.streaming,
            )
    else:  # 自定义数据集
        data_files = {}      # 用来存储训练集和验证集的文件路径列表
        dataset_args = {}    # 存放传递给 load_dataset 的额外参数（如处理 TXT 文件的换行符选项）
        # 扫描用户提供的训练数据目录，收集所有 TXT/JSON/JSONL 文件，检查文件类型是否一致，并将训练文件路径保存到 data_files["train"]
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(f'{data_args.train_file_dir}/**/*.txt', recursive=True) + glob(
                f'{data_args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {train_data_files}")
            # Train data files must be same type, e.g. all txt or all jsonl
            types = [f.split('.')[-1] for f in train_data_files]
            if len(set(types)) > 1:
                raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")
            data_files["train"] = train_data_files
        # 将验证文件保存到 data_files["validation"]
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
            eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.txt', recursive=True) + glob(
                f'{data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {eval_data_files}")
            data_files["validation"] = eval_data_files
            # Train data files must be same type, e.g. all txt or all jsonl
            types = [f.split('.')[-1] for f in eval_data_files]
            if len(set(types)) > 1:
                raise ValueError(f"train files must be same type, e.g. all txt or all jsonl, but got {types}")
        
        # 根据训练集文件类型确定数据集的格式，用于后续调用 load_dataset。extension 的值只可能是两个之一：text" → 用于 TXT 文件"json" → 用于 JSON / JSONL 文件
        extension = "text" if data_files["train"][0].endswith('txt') else 'json'
        if extension == "text": # 只有训练集是 TXT 文件时才会执行下面操作
            # 否保留每行末尾的换行符 ，默认保留 True
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        # 数据加载：把本地训练/验证文件加载为 Hugging Face DatasetDict
        raw_datasets = load_dataset(
            extension,                      # 解析方式
            data_files=data_files,          # 加载训练/验证文件
            cache_dir=model_args.cache_dir, # 指定本地缓存目录，数据集会被下载/缓存到这里 将数据解析为Arrow文件，不用重复解析 None则使用 默认缓存目录linxu: ~/.cache/huggingface/datasets/
            **dataset_args,                 # 可选的额外参数字典  **dataset_args 表示把字典里的键值对解包成关键字参数传给函数
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        # # 如果没有验证数据，将使用“验证分割百分比”来对数据集进行划分。
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                **dataset_args, 
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )
    logger.info(f"Raw datasets: {raw_datasets}")
    '''
        txt 文件 每行一条
        Raw datasets: DatasetDict({
            train: Dataset({
                features: ['text'],
                num_rows: 3876
            })
            validation: Dataset({
                features: ['text'],
                num_rows: 3876
            })
        })
    '''
    
    #***********************************************************************************#
    # 预处理数据
    #***********************************************************************************#
    # True → 执行训练 | False → 不训练，只评估或推理
    # column_names = text
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    
    # 这里面进行的数据处理只在主进程先执行
    # 主要涉及 数据集的预处理逻辑”，主进程先执行，其他进程等候使用缓存
    with training_args.main_process_first(desc="Dataset tokenization and grouping"):
        if not data_args.streaming: # 不是流式 数据一次性加载到内存，可以生成缓存文件 默认 streaming = False
            if data_args.packing: # 对文本做 拼接 + block_size 切块 默认True
                '''
                在 Hugging Face datasets 库中，Dataset 或 DatasetDict 对象都有一个方法叫 .map()
                作用：对数据集的每条样本或每批样本应用一个函数，返回一个新的 Dataset
                '''
                tokenized_datasets = raw_datasets.map(
                    tokenize_wo_pad_function,                                         # tokenize_wo_pad_function 对文本分词
                    batched=True,                                                     # batched=True → 批量分词，提高速度
                    num_proc=data_args.preprocessing_num_workers,                     # 多进程处理
                    remove_columns=column_names,                                      # 删除原始字段（如 "text"）
                    load_from_cache_file=not data_args.overwrite_cache,               # 使用缓存，加快重复运行
                    desc="Running tokenizer on dataset" if is_main_process else None, # 进度条显示，仅主进程显示
                )
                # lm_datasets:经过预处理（分词 + 可选切块）后的数据集，准备用于语言模型LM的训练
                lm_datasets = tokenized_datasets.map(
                    group_text_function,                                              # 调用group_text_function，先拼接后分块
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Packing texts in chunks of {block_size}",
                )
            else:                # 不做拼接/切块
                lm_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset" if is_main_process else None,
                )
        else: # streaming=True，流式加载 流式模式下，不使用缓存，也不支持多进程
            if data_args.packing:
                tokenized_datasets = raw_datasets.map(
                    tokenize_wo_pad_function,
                    batched=True,
                    remove_columns=column_names,
                )
                lm_datasets = tokenized_datasets.map(
                    group_text_function,
                    batched=True,
                )
            else:
                lm_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

    # train_dataset → 用来存放最终训练集
    # max_train_samples → 用来记录训练样本数
    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets['train']
        max_train_samples = len(train_dataset)
        # 使用用户指定的data_args.max_train_samples，但若大于实际训练数量则进行修正
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            # 从训练集中选择前 max_train_samples 条样本
            train_dataset = train_dataset.select(range(max_train_samples))
        # 输出训练样本数量
        logger.debug(f"Num train_samples: {len(train_dataset)}")
        # 打印第一条训练样本的 tokenized 内容（解码成文本）
        logger.debug("Tokenized training example:")
        logger.debug(tokenizer.decode(train_dataset[0]['input_ids']))

    # eval_dataset → 用来存放最终验证集
    # max_eval_samples → 用来记录验证集数
    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.debug(f"Num eval_samples: {len(eval_dataset)}")
        logger.debug("Tokenized eval example:")
        logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))
    
    
    #***********************************************************************************#
    # 加载模型
    #***********************************************************************************#
    # Load model
    if model_args.model_name_or_path:
        # 设置Torch 模型的计算数据类型
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        # 判断是否分布式训练
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        ddp = world_size != 1
        if ddp: # 分布式
            model_args.device_map = None # 不使用自动分配显存
        if model_args.device_map in ["None", "none", ""]:
            model_args.device_map = None
        # QLoRA（量化 LoRA）不支持 FSDP（Fully Sharded Data Parallel）或 DeepSpeed ZeRO-3 如果同时启用 → 输出警告
        if script_args.qlora and (len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
            logger.warning("FSDP and DeepSpeed ZeRO-3 are both currently incompatible with QLoRA.")

        # 从HF中加载模型的配置参数
        config_kwargs = {
            "trust_remote_code": model_args.trust_remote_code, # 是否信任远程模型代码（允许加载自定义模型架构）
            "cache_dir": model_args.cache_dir,                 # 缓存 Hugging Face 模型的位置
            "revision": model_args.model_revision,             # 模型分支或版本 默认main
            "token": model_args.hf_hub_token,                  # Hugging Face Hub 授权 token
        }
        # 加载模型配置（模型结构、层数、hidden size 等）
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        
        # 是否启用 4bit 或 8bit 量化 并加载参数到config_kwargs
        # 量化不允许启用DeepSpeed ZeRO-3
        load_in_4bit = model_args.load_in_4bit
        load_in_8bit = model_args.load_in_8bit
        if load_in_4bit and load_in_8bit:
            raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")
        elif load_in_8bit or load_in_4bit:
            logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
            if is_deepspeed_zero3_enabled():
                raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
            if load_in_8bit:
                # bitsandbytes 库配置 8bit 量化
                config_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
            elif load_in_4bit:
                # QLoRA → 使用 双量化(double quant) + NF4 类型(NormalFloat4正态分布调整的小数格式)
                if script_args.qlora:
                    config_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch_dtype,
                    )
                else:
                # 只指定 4bit 和计算 dtype ( 训练或推理时 4bit 权重反量化后会转换成指定 dtype)
                    config_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                    )
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            device_map=model_args.device_map,
            **config_kwargs,
        )
    else:
        raise ValueError(f"Error, model_name_or_path is None, Continue PT must be loaded from a pre-trained model")

    # 微调训练方式设置 LORA / 全参数训练
    if script_args.use_peft:    # LORA微调配置
        logger.info("Fine-tuning method: LoRA(PEFT)")
        if script_args.peft_path is not None:   # 加载已有的LORA权重
            logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
        else:                                   # 初始化新的 LoRA 模型
            logger.info("Init new peft model")
            # 准备 8bit / 4bit 量化模型
            if load_in_8bit or load_in_4bit:
                model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)
            
            # 确定 LoRA 目标模块
            target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
            # 如果target_modules=='all' 则find所有线性层：q k v o 
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)
            
            # 指定LoRA 微调后需要保存的模块
            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
                # Resize the embedding layer to match the new tokenizer
                # 如果 tokenizer 扩充了词表，需要调整模型的 embedding 层
                embedding_size = model.get_input_embeddings().weight.shape[0]
                if len(tokenizer) > embedding_size:
                    model.resize_token_embeddings(len(tokenizer))
            
            # 打印LORA配置，并生成配置文件     
            logger.info(f"Peft target_modules: {target_modules}")
            logger.info(f"Peft lora_rank: {script_args.lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,         # 自回归语言模型任务
                target_modules=target_modules,        # LoRA 应用到的层
                inference_mode=False,                 # LoRA 应用到的层
                r=script_args.lora_rank,              # 低秩矩阵维度
                lora_alpha=script_args.lora_alpha,    # LoRA 缩放系数
                lora_dropout=script_args.lora_dropout,# LoRA dropout
                modules_to_save=modules_to_save)      # 保存的模块
            # 加载LORA模型
            model = get_peft_model(model, peft_config)
        
        # 强制 LoRA 层为 float32 即使模型量化为 4bit/8bit，LoRA 权重仍用 float32 存储 避免训练中梯度精度损失
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
            
        # 输出可训练参数    
        model.print_trainable_parameters()
    else:  # 全参数训练
        logger.info("Fine-tuning method: Full parameters training")
        model = model.float()
        print_trainable_parameters(model)

    
    #***********************************************************************************#
    # 训练模型
    #***********************************************************************************#
    # Initialize our Trainer
    # gradient_checkpointing=True 训练默认开启 推理关闭
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    
    # 使模型可训练
    model.enable_input_require_grads()
    
    
    if not ddp and torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # trainer 初始化
    trainer = SavePeftModelTrainer(
        model=model,                                                                         # 当前训练的模型（可能是 4bit/8bit + LoRA/全参数）
        args=training_args,                                                                  # 训练参数
        train_dataset=train_dataset if training_args.do_train else None,                     # 训练集
        eval_dataset=eval_dataset if training_args.do_eval else None,                        # 验证集
        processing_class=tokenizer,                                                          # tokenizer
        data_collator=fault_tolerance_data_collator,                                         # 自定数据批次整理函数
        compute_metrics=compute_metrics if training_args.do_eval else None,                  # 评价指标
        preprocess_logits_for_metrics=preprocess_logits_for_metrics                          # training_args.do_eval = true 预处理模型输出 logits
        if training_args.do_eval                                                                
        else None,
    )

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        logger.debug(f"Train dataloader example: {next(iter(trainer.get_train_dataloader()))}")
        checkpoint = None
        
        # resume_from_checkpoint → 指定训练断点
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        # 调用 Trainer 的 train() 方法开始训练 checkpoint 不为空 → 从断点继续训练
        # loss = CrossEntropyLoss(logits, labels)
        # 优化器：AdamW
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # train_result 包含 metrics（训练指标）、global_step、训练状态等
        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # 训练结束后恢复缓存，提高推理速度
        # 恢复左填充，保证生成/推理与训练兼容
        model.config.use_cache = True  # enable cache after training
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"

        # 分布式训练时只有主进程执行保存
        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {training_args.output_dir}")
            if is_deepspeed_zero3_enabled():  # DeepSpeed ZeRO-3分布式训练的保存
                save_model_zero3(model, tokenizer, training_args, trainer)
            else:                             # 直接保存
                save_model(model, tokenizer, training_args)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        
        # 计算模型输出与标签的 loss
        # 如果指定了 compute_metrics → 计算其他指标（如准确率）
        metrics = trainer.evaluate()

        # 验证样本数
        metrics["eval_samples"] = max_eval_samples
        
        # 评价指标：计算困惑度 Perplexity
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        # metrics 准确率指标
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
