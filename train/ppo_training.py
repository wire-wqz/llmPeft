# -*- coding: utf-8 -*-
"""
PPO 无监督，只需要SFT中的人类提示词，生成输出，由奖励函数打分，分数越高越好
"""

import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional
from datasets import load_dataset
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForCausalLM,
)
from trl import (
    RLOOConfig,
    RLOOTrainer,
    ModelConfig,
    get_peft_config,
)
from template import get_conv_template

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



@dataclass
class RLOOArguments:
    """
    The name of the Casual LM model we wish to fine with RLOO
    """
    sft_model_path: Optional[str] = field(default=None, metadata={"help": "Path to the SFT model."})
    reward_model_path: Optional[str] = field(default=None, metadata={"help": "Path to the reward model."})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name."})
    dataset_config: Optional[str] = field(default=None, metadata={"help": "Dataset configuration name."})
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."}, )
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "The template name."})
    max_source_length: Optional[int] = field(default=1024, metadata={"help": "Max length of prompt input text"})
    dataset_num_proc: Optional[int] = field(default=1)


def main():
    
    #***********************************************************************************#
    # 参数解析
    #***********************************************************************************#
    parser = HfArgumentParser((RLOOArguments, RLOOConfig, ModelConfig))
    args, training_args, model_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )[:3]

    # Add distributed training initialization
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = local_rank == 0

    # Only log on main process
    if is_main_process:
        logger.info(f"Parse args: {args}")
        logger.info(f"Training args: {training_args}")
        logger.info(f"Model args: {model_args}")

    # Load tokenizer
    #***********************************************************************************#
    # 加载分词器
    #***********************************************************************************#
    sft_model_path = args.sft_model_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        sft_model_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.sep_token
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    logger.debug(f"Tokenizer: {tokenizer}")

    # Load reward model as reward function
    #***********************************************************************************#
    # 加载奖励模型
    #***********************************************************************************#
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )

    
    
    # Load policy model
    #***********************************************************************************#
    # 加载policy model 这个模型就是后面要被 RL 继续优化的 policy（策略模型）
    #***********************************************************************************#
    policy = AutoModelForCausalLM.from_pretrained(
        sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    peft_config = get_peft_config(model_args)

    # Get datasets
    #***********************************************************************************#
    # 加载数据集
    #***********************************************************************************#
    prompt_template = get_conv_template(args.template_name)
    if args.dataset_name is not None:  # 从 HF中下载
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_train_split
        )
        eval_samples = 100
        train_dataset = dataset.select(range(len(dataset) - eval_samples))
        eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    else:
        data_files = {}
        if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
            train_data_files = glob(f'{args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {', '.join(train_data_files)}")
            data_files["train"] = train_data_files
        if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
            eval_data_files = glob(f'{args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {', '.join(eval_data_files)}")
            data_files["validation"] = eval_data_files
        dataset = load_dataset(
            'json',
            data_files=data_files,
        )
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        # 取val_dataset中的100个作为eval_dataset
        eval_dataset = val_dataset.select(range(min(100, len(val_dataset))))
    logger.info(f"Get datasets: {train_dataset}, {eval_dataset}")

    # Preprocessing the datasets
    max_source_length = args.max_source_length


    #***********************************************************************************#
    # 数据集预处理
    #***********************************************************************************#
    def preprocess_function(examples):
        """
        提取所有的人类指令Prompts
        
        输入：
        examples = {
            "system_prompt": ["你是一个有帮助的助手。"],
            "conversations": [
                [
                    {"from": "human", "value": "你好"},
                    {"from": "gpt", "value": "你好，请问有什么可以帮你？"},
                    {"from": "human", "value": "什么是强化学习？"},
                    {"from": "gpt", "value": "强化学习是一种通过奖励信号学习策略的方法。"}
                ]
            ]
        }
        输出：
        new_examples = 
        {
            "prompt": [
                "你好", 
                "什么是强化学习？"
            ]
        }
    
        
        """
        
        new_examples = {"prompt": []}
        roles = ["human", "gpt"]

        def get_dialog(examples):
            system_prompts = examples.get("system_prompt", "")
            for i, source in enumerate(examples['conversations']):
                if len(source) < 2:
                    continue
                data_role = source[0].get("from", "")
                if data_role not in roles or data_role != roles[0]:
                    # Skip the first one if it is not from human
                    source = source[1:]
                if len(source) < 2:
                    continue
                messages = []
                for j, sentence in enumerate(source):
                    data_role = sentence.get("from", "")
                    if data_role not in roles:
                        logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                        break
                    if data_role == roles[j % 2]:
                        messages.append(sentence["value"])
                if len(messages) < 2 or len(messages) % 2 != 0:
                    continue
                # Convert the list to pairs of elements
                history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
                system_prompt = system_prompts[i] if system_prompts else None
                yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)

        for dialog in get_dialog(examples):
            for i in range(len(dialog) // 2):
                source_txt = dialog[2 * i]
                new_examples["prompt"].append(source_txt)

        return new_examples

    # Preprocess the dataset
    if is_main_process:
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        
        # tokenized_train_dataset 只保留人类Prompt
        tokenized_train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.dataset_num_proc,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset" if is_main_process else None,
        )
        # 清除为空的Prompt
        train_dataset = tokenized_train_dataset.filter(
            lambda x: len(x['prompt']) > 0
        )
        logger.debug(f"Train samples top3: {train_dataset[:3]}")

        # Preprocess the dataset for evaluation
        logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
        tokenized_eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.dataset_num_proc,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset" if is_main_process else None,
        )
        eval_dataset = tokenized_eval_dataset.filter(
            lambda x: len(x['prompt']) > 0
        )
        logger.debug(f"Eval samples top3: {eval_dataset[:3]}")

    # We then build the RLOOTrainer, passing the model, the reward function, the tokenizer
    # RLOO does not need a separate value model or ref model (unlike PPO)
    # 接着，我们构建了 RLOOTrainer，传入了模型、奖励函数以及分词器
    # RLOO 不需要单独的值模型或参考模型（与 PPO 不同）
    trainer = RLOOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        reward_funcs=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Training
    if training_args.do_train:
        if is_main_process:
            logger.info("*** Train ***")
        trainer.train()

        # Only log on main process
        if is_main_process:
            trainer.save_model(training_args.output_dir)

    # trainer.generate_completions()


if __name__ == "__main__":
    main()
