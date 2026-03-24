#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把奖励模型数据从三列格式：
    question, response_chosen, response_rejected
转换为五列格式：
    system, history, question, response_chosen, response_rejected

兼容两种输入：
1) JSON 数组: [ {...}, {...} ]
2) JSONL: 每行一个 JSON 对象

示例：
python convert_reward_dataset.py \
    --input ./minedata/rl/train/train.json \
    --output ./minedata/rl/train/train_converted.json

批量转换整个目录：
python convert_reward_dataset.py \
    --input_dir ./minedata/rl \
    --output_dir ./minedata/rl_converted
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


REQUIRED_OLD_FIELDS = ["question", "response_chosen", "response_rejected"]
NEW_FIELDS = ["system", "history", "question", "response_chosen", "response_rejected"]


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # 优先尝试普通 JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            if not all(isinstance(x, dict) for x in data):
                raise ValueError(f"文件 {path} 是 JSON 数组，但其中元素不是对象。")
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"文件 {path} 的 JSON 顶层必须是对象或数组。")
    except json.JSONDecodeError:
        pass

    # 再尝试 JSONL
    records: List[Dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"文件 {path} 第 {lineno} 行不是合法 JSON: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"文件 {path} 第 {lineno} 行的 JSON 不是对象。")
        records.append(obj)
    return records


def save_as_json(data: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_history(history: Any) -> List[Any]:
    if history in (None, ""):
        return []
    if isinstance(history, list):
        return history
    # 有些脏数据可能把 history 存成字符串，保守处理为 []
    return []


def convert_record(record: Dict[str, Any], default_system: str = "") -> Dict[str, Any]:
    missing = [k for k in REQUIRED_OLD_FIELDS if k not in record]
    if missing:
        raise ValueError(f"样本缺少必要字段: {missing}. 当前样本 keys={list(record.keys())}")

    # 如果原本就有 system/history，就尽量保留；没有则补默认值
    system = record.get("system", default_system)
    if system is None:
        system = default_system

    history = normalize_history(record.get("history", []))

    new_record = {
        "system": system,
        "history": history,
        "question": record["question"],
        "response_chosen": record["response_chosen"],
        "response_rejected": record["response_rejected"],
    }
    return new_record


def convert_records(records: Iterable[Dict[str, Any]], default_system: str = "") -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        try:
            converted.append(convert_record(record, default_system=default_system))
        except Exception as e:
            raise ValueError(f"第 {idx} 条样本转换失败: {e}") from e
    return converted


def print_preview(records: List[Dict[str, Any]], limit: int = 2) -> None:
    print("\n转换后的字段顺序:", NEW_FIELDS)
    print(f"总样本数: {len(records)}")
    if not records:
        return
    print("\n预览前几条样本:")
    for i, rec in enumerate(records[:limit]):
        print(f"--- sample {i} ---")
        print(json.dumps(rec, ensure_ascii=False, indent=2))


def convert_one_file(input_path: Path, output_path: Path, default_system: str = "") -> None:
    records = load_json_or_jsonl(input_path)
    converted = convert_records(records, default_system=default_system)
    save_as_json(converted, output_path)
    print(f"已转换: {input_path} -> {output_path}")
    print_preview(converted, limit=1)


def iter_json_files(input_dir: Path) -> List[Path]:
    files = sorted([p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}])
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="奖励模型数据格式转换脚本")
    parser.add_argument("--input", type=str, help="单个输入文件路径（json 或 jsonl）")
    parser.add_argument("--output", type=str, help="单个输出文件路径")
    parser.add_argument("--input_dir", type=str, help="批量输入目录")
    parser.add_argument("--output_dir", type=str, help="批量输出目录")
    parser.add_argument("--default_system", type=str, default="", help="当原数据没有 system 字段时，补入的默认 system 提示词")
    args = parser.parse_args()

    if args.input:
        if not args.output:
            raise SystemExit("使用 --input 时，必须同时提供 --output")
        convert_one_file(Path(args.input), Path(args.output), default_system=args.default_system)
        return

    if args.input_dir:
        if not args.output_dir:
            raise SystemExit("使用 --input_dir 时，必须同时提供 --output_dir")

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        files = iter_json_files(input_dir)
        if not files:
            raise SystemExit(f"目录中未找到 json/jsonl 文件: {input_dir}")

        for in_file in files:
            rel = in_file.relative_to(input_dir)
            out_file = output_dir / rel.with_suffix(".json")
            convert_one_file(in_file, out_file, default_system=args.default_system)
        return

    raise SystemExit("请提供 --input/--output 或 --input_dir/--output_dir")


if __name__ == "__main__":
    main()
