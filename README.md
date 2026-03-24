# MedicalGPT / llmPert

一个面向中文医疗场景的大语言模型微调与对齐训练项目。

项目基于 **PyTorch + Transformers + PEFT + TRL** 搭建，围绕 **Qwen / DeepSeek** 等开源大模型，支持从领域知识注入到偏好对齐的完整训练流程，包括：

- 预训练 / 继续预训练（PT）
- 监督微调（SFT）
- 奖励建模（Reward Modeling）
- RLHF / PPO
- DPO
- GRPO
- LoRA / QLoRA 参数高效微调
- 模型合并与推理部署

项目目标是将通用基座模型适配到 **中文医疗问答与垂域生成场景**，并提供从训练到推理的一体化工程范式。

基于本项目训练的Qwen2.5-7B权重参数如下： [MedQwen2.5-7B](https://github.com/shibing624/MedicalGPT)

---

## 1. 项目是什么

MedicalGPT / llmPert 是一个面向医疗垂域的大模型训练项目，围绕 **Qwen、DeepSeek** 等模型，完成从领域知识注入到偏好对齐的完整训练链路，支持：

- 基于中文医疗语料的 PT / SFT
- 基于偏好数据的奖励建模
- 基于 RLHF、DPO、GRPO 的模型对齐
- 基于 LoRA / QLoRA 的参数高效微调
- 训练后模型合并与推理验证

适用于：

- 中文医疗问答
- 医疗知识注入
- LLM 垂域适配
- 偏好对齐实验
- Qwen / DeepSeek 微调工程实践

---

## 2. 项目依赖

项目核心依赖包括：

- accelerate
- datasets
- loguru
- peft >= 0.14.0
- sentencepiece
- scikit-learn
- tensorboard
- tqdm >= 4.47.0
- transformers >= 5.1.0
- trl >= 0.27.0
- latex2sympy2_extended
- math-verify == 0.5.2

安装方式：

```bash
pip install -r requirements.txt
```

> 如需使用 QLoRA / 4bit 训练，请根据运行环境额外安装 `bitsandbytes`。

---

## 3. 项目路径

当前项目目录结构如下：

```text
.
├── data/
├── inference/
├── merge/
├── train/
├── demo_run_Qwen2.5_0.5B.ipynb
├── readme.md
└── requirements.txt
```

---

## 4. 项目文件夹作用

### `data/`
用于存放训练与评测数据的 demo 数据（主要用于展示数据格式），包括：

- 预训练语料
- SFT 指令数据
- 奖励建模偏好数据
- DPO / GRPO / RLHF 数据
- 数据转换后的中间结果

### `train/`
训练目录，是项目核心模块。主要用于放置各类训练脚本与训练流程实现，例如：

- PT / Continued Pretraining 脚本
- SFT 脚本
- Reward Modeling 脚本
- PPO / RLHF 训练脚本
- DPO / GRPO 训练脚本
- 训练参数配置与通用工具函数

### `merge/`
模型合并目录。主要用于：

- 合并 LoRA / QLoRA Adapter 与基座模型
- 导出最终可部署模型权重
- 生成推理所需的完整模型目录

### `inference/`
推理与测试目录。主要用于：

- 加载训练后的模型进行推理
- 构建医疗问答 Demo
- 进行效果验证与样例测试
- 部署前的接口联调与结果检查

### `demo_run_Qwen2.5_0.5B.ipynb`
项目演示 Notebook，用于快速体验项目流程与验证环境。

### `requirements.txt`
项目依赖文件，用于快速安装运行环境。

---

## 5. 项目 Demo

项目提供一个 Notebook Demo：

### `demo_run_Qwen2.5_0.5B.ipynb`

推荐用途：

1. 验证环境是否安装成功
2. 快速了解项目基本流程
3. 测试 Qwen2.5-0.5B 的加载、训练或推理
4. 作为项目上手入口

推荐使用步骤：

```bash
pip install -r requirements.txt
```

然后打开：

```text
demo_run_Qwen2.5_0.5B.ipynb
```

按顺序运行 Notebook 单元即可。

---

## Acknowledgements / 致谢

This project builds upon the impressive work of [MedicalGPT](https://github.com/shibing624/MedicalGPT).  
本项目基于 [MedicalGPT](https://github.com/shibing624/MedicalGPT) 的优秀工作进行构建。

**Citation / 引用：**

```bibtex
@misc{MedicalGPT,
  title={MedicalGPT: Training Medical GPT Model},
  author={Ming Xu},
  year={2023},
  howpublished={\url{https://github.com/shibing624/MedicalGPT}},
}
```
