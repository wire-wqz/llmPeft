# -*- coding: utf-8 -*-
import argparse

import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)


def main():
    # lora参数融合 config
    parser = argparse.ArgumentParser()
    # 基础模型的名字或路径
    parser.add_argument('--base_model', default=None, required=True, type=str,help="Base model name or path")
    # tokenizer 的路径 / 有些时候 tokenizer 和 base model 在同一个目录 无需指定
    parser.add_argument('--tokenizer_path', default=None, type=str, help="Please specify tokenization path.")
    # 要合并进基础模型的 LoRA 权重路径
    parser.add_argument('--lora_model', default=None, required=True, type=str, help="Please specify LoRA model to be merged.")
    # 是否需要调整模型维度：用于tokenizer 词表变大了的情况
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    # 合并后的模型保存到哪里
    parser.add_argument('--output_dir', default='./merged', type=str)
    # 合并后的模型上传到 Hugging Face Hub，这个参数就是目标仓库名
    parser.add_argument('--hf_hub_model_id', default='', type=str)
    # 传Hugging Face Hub 时用的访问 token 令牌
    parser.add_argument('--hf_hub_token', default=None, type=str)
    
    args = parser.parse_args()
    print(args)

    base_model_path = args.base_model
    lora_model_path = args.lora_model
    output_dir = args.output_dir
    print(f"Base model: {base_model_path}")
    print(f"LoRA model: {lora_model_path}")
    
    # 加载LORA模型
    peft_config = PeftConfig.from_pretrained(lora_model_path)

    # 加载BASE模型
    if peft_config.task_type == "SEQ_CLS": # 序列分类模型加载方式
        print("Loading LoRA for sequence classification model")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=1,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="auto",
        )
    else:                                # 加载 LoRA 以用于因果语言模型
        print("Loading LoRA for causal language model")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype='auto',
            trust_remote_code=True,
            device_map="auto",
        )
        
    # 有tokenizer_path则从中加载，否则从base_model_path中加载
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 如果用户指定了 --resize_emb，就检查“模型词表大小”和“tokenizer 词表大小”是否一致；
    # 如果不一致，就把模型的 embedding 层扩展或缩小到和 tokenizer 一样大。
    if args.resize_emb:
        base_model_token_size = base_model.get_input_embeddings().weight.size(0)
        if base_model_token_size != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))
            print(f"Resize vocabulary size {base_model_token_size} to {len(tokenizer)}")

    # 融合模型（这里只是一个base+lora分支的模型）
    new_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map="auto",
        torch_dtype='auto',
    )
    new_model.eval()
    
    # 把 LoRA 的低秩增量权重真正加回到基础模型权重里，然后卸载 LoRA 结构，返回一个普通模型
    print(f"Merging with merge_and_unload...")
    base_model = new_model.merge_and_unload()

    # 保存为HF格式
    print("Saving to Hugging Face format...")
    tokenizer.save_pretrained(output_dir)
    base_model.save_pretrained(output_dir, max_shard_size='10GB')
    print(f"Done! model saved to {output_dir}")
    if args.hf_hub_model_id:
        print(f"Pushing to Hugging Face Hub...")
        base_model.push_to_hub(
            args.hf_hub_model_id,
            token=args.hf_hub_token,
            max_shard_size="10GB",
        )
        tokenizer.push_to_hub(
            args.hf_hub_model_id,
            token=args.hf_hub_token,
        )
        print(f"Done! model pushed to Hugging Face Hub: {args.hf_hub_model_id}")


if __name__ == '__main__':
    main()
