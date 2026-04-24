"""
因果语言建模训练脚本
支持 LLaMA-3-1B / Falcon-1B / MPT-1B / OPT-1.3B
数据集: WikiText-2 / OpenWebText
注意力机制差异: MHA(OPT/MPT) / GQA(LLaMA-3) / MQA(Falcon)
"""

import os
import math
import time
import csv
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from datasets import load_dataset, concatenate_datasets
import pynvml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 模型配置：名称 → HuggingFace 模型 ID + 注意力类型标注
MODEL_CONFIGS = {
    "llama3-1b": {
        "model_id": "meta-llama/Llama-3.2-1B",
        "attn_type": "GQA",
    },
    "falcon-1b": {
        "model_id": "tiiuae/falcon-rw-1b",
        "attn_type": "MQA",
    },
    "mpt-1b": {
        "model_id": "mosaicml/mpt-1b-redpajama-200b",
        "attn_type": "MHA",
    },
    "opt-1.3b": {
        "model_id": "facebook/opt-1.3b",
        "attn_type": "MHA",
    },
}


@dataclass
class LMExperimentArgs:
    model_name: str = field(
        default="opt-1.3b",
        metadata={"help": f"模型选择: {list(MODEL_CONFIGS.keys())}"},
    )
    dataset: str = field(
        default="wikitext",
        metadata={"help": "数据集: wikitext / openwebtext / both"},
    )
    output_dir: str = field(default="./results")
    max_seq_len: int = field(default=256)
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=5e-5)
    warmup_ratio: float = field(default=0.05)
    weight_decay: float = field(default=0.01)
    fp16: bool = field(default=True)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=5000)
    results_csv: str = field(default="./results/results_lm.csv")
    seed: int = field(default=42)
    # openwebtext 子集大小（条数）
    owt_subset_size: int = field(default=50000)


def get_gpu_memory_mb() -> float:
    """返回当前进程占用的 GPU 显存 (MB)，无 GPU 时返回 0"""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 / 1024
    except Exception:
        return 0.0


class MetricsCallback(TrainerCallback):
    """训练进度日志 + 每个 epoch 结束后记录指标并追加到 CSV"""

    def __init__(self, args: LMExperimentArgs, model_name: str, attn_type: str, dataset_name: str):
        self.exp_args = args
        self.model_name = model_name
        self.attn_type = attn_type
        self.dataset_name = dataset_name
        self.epoch_start_time: float = 0.0
        self.step_start_time: float = time.time()
        self.epoch_train_tokens: int = 0
        self.rows: list[dict] = []

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.epoch_start_time = time.time()
        logger.info(f"▶ Epoch {int((state.epoch or 0) + 1)} 开始")

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.step_start_time = time.time()

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """打印每步训练日志，包含步耗时"""
        if logs is None:
            return
        if "loss" in logs:
            step_sec = time.time() - self.step_start_time
            tokens_per_step = (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * args.world_size
                * self.exp_args.max_seq_len
            )
            self.epoch_train_tokens += tokens_per_step
            tok_s = tokens_per_step / (step_sec + 1e-6)
            logger.info(
                f"  step {state.global_step}/{state.max_steps}"
                f"  loss={logs['loss']:.4f}"
                f"  lr={logs.get('learning_rate', 0):.2e}"
                f"  {step_sec:.1f}s/step  {tok_s:.0f} tok/s"
            )

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None:
            return

        epoch = round(state.epoch or 0, 2)
        eval_loss = metrics.get("eval_loss", float("nan"))
        ppl = math.exp(eval_loss) if not math.isnan(eval_loss) else float("nan")

        # 评估阶段吞吐（tokens/sec）
        elapsed = time.time() - self.epoch_start_time + 1e-6
        tokens_sec = self.epoch_train_tokens / elapsed if self.epoch_train_tokens > 0 else 0.0
        gpu_mem = get_gpu_memory_mb()

        row = {
            "model": self.model_name,
            "attn_type": self.attn_type,
            "dataset": self.dataset_name,
            "epoch": epoch,
            "eval_loss": round(eval_loss, 4),
            "perplexity": round(ppl, 2),
            "tokens_per_sec": round(tokens_sec, 1),
            "gpu_mem_mb": round(gpu_mem, 1),
        }
        self.rows.append(row)

        logger.info(
            f"[{self.model_name}|{self.attn_type}] epoch={epoch}  "
            f"loss={eval_loss:.4f}  ppl={ppl:.2f}  "
            f"tok/s={tokens_sec:.0f}  gpu={gpu_mem:.0f}MB"
        )

        # 追加写入 CSV
        csv_path = self.exp_args.results_csv
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # 重置计数
        self.epoch_start_time = time.time()
        self.epoch_train_tokens = 0


def load_and_tokenize(tokenizer, args: LMExperimentArgs):
    """加载 wikitext-2 / openwebtext 并做 block 式 tokenization"""
    import datasets as _ds
    _ds.disable_caching()  # 禁用缓存，防止旧缓存导致列长度不匹配

    def _tokenize(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    def _group_texts(examples):
        """将所有 token 拼接后按 max_seq_len 切块，只处理 input_ids"""
        all_ids = sum(examples["input_ids"], [])
        total = (len(all_ids) // args.max_seq_len) * args.max_seq_len
        chunks = [all_ids[i : i + args.max_seq_len] for i in range(0, total, args.max_seq_len)]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * args.max_seq_len for _ in chunks],
            "labels": [c[:] for c in chunks],
        }

    datasets_list = []

    if args.dataset in ("wikitext", "both"):
        logger.info("加载 WikiText-2 ...")
        wt = load_dataset("wikitext", "wikitext-2-raw-v1")
        # 过滤空行
        wt = wt.filter(lambda x: len(x["text"].strip()) > 0)
        datasets_list.append(("wikitext", wt))

    if args.dataset in ("openwebtext", "both"):
        logger.info(f"加载 OpenWebText (前 {args.owt_subset_size} 条) ...")
        owt = load_dataset("openwebtext", split="train", streaming=False)
        owt = owt.select(range(min(args.owt_subset_size, len(owt))))
        # 构造与 wikitext 相同格式的 DatasetDict
        split = owt.train_test_split(test_size=0.01, seed=args.seed)
        owt_dict = {"train": split["train"], "validation": split["test"], "test": split["test"]}
        from datasets import DatasetDict
        owt_dict = DatasetDict(owt_dict)
        datasets_list.append(("openwebtext", owt_dict))

    # 合并多数据集（如 both）
    if len(datasets_list) == 2:
        (_, ds1), (_, ds2) = datasets_list
        combined = {
            "train": concatenate_datasets([ds1["train"], ds2["train"]]),
            "validation": concatenate_datasets([ds1["validation"], ds2["validation"]]),
        }
        from datasets import DatasetDict
        raw = DatasetDict(combined)
        dataset_label = "wikitext+openwebtext"
    else:
        raw = datasets_list[0][1]
        dataset_label = datasets_list[0][0]

    logger.info("Tokenizing ...")
    # tokenize：移除 text，保留 input_ids / attention_mask
    tokenized = raw.map(
        _tokenize,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="tokenizing",
    )
    # grouping：输出行数与输入不同，必须显式 remove_columns 避免 PyArrow 长度校验报错
    lm_dataset = tokenized.map(
        _group_texts,
        batched=True,
        remove_columns=tokenized["train"].column_names,
        desc="grouping",
    )

    train_ds = lm_dataset["train"]
    eval_ds = lm_dataset["validation"]

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    logger.info(f"训练集: {len(train_ds)} 块  评估集: {len(eval_ds)} 块")
    return train_ds, eval_ds, dataset_label


def compute_metrics_fn(eval_pred):
    """Trainer compute_metrics 钩子（仅输出 loss，PPL 在 callback 里算）"""
    return {}


def run_experiment(args: LMExperimentArgs):
    assert args.model_name in MODEL_CONFIGS, (
        f"未知模型 {args.model_name}，可选: {list(MODEL_CONFIGS.keys())}"
    )

    cfg = MODEL_CONFIGS[args.model_name]
    model_id = cfg["model_id"]
    attn_type = cfg["attn_type"]

    logger.info(f"=== 实验: {args.model_name} | 注意力: {attn_type} ===")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, eval_ds, dataset_label = load_and_tokenize(tokenizer, args)

    # 根据可用设备选择加载策略
    # - CUDA: fp16 + device_map="auto"（多卡自动分配）
    # 设备与精度策略：
    # - CUDA : fp16 + device_map="auto"
    # - MPS  : bfloat16（Apple Silicon 原生支持，比 fp32 快 2-4x，比 fp16 稳定）
    # - CPU  : fp32
    if torch.cuda.is_available():
        # 混合精度训练(fp16=True)要求权重保持fp32，Trainer内部会自动cast到fp16做前向，
        # 并用GradScaler在反向时unscale梯度。若此处把权重直接加载为fp16，
        # 会触发"Attempting to unscale FP16 gradients"报错。
        load_dtype = torch.float32
        device_map = "auto"
        use_fp16 = args.fp16
        use_bf16 = False
        target_device = "cuda"
        logger.info("设备: CUDA (权重fp32 + 混合精度fp16)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        load_dtype = torch.bfloat16
        device_map = None
        use_fp16 = False
        use_bf16 = True
        target_device = "mps"
        logger.info("设备: MPS (Apple Silicon, bfloat16)")
    else:
        load_dtype = torch.float32
        device_map = None
        use_fp16 = False
        use_bf16 = False
        target_device = "cpu"
        logger.info("设备: CPU")

    logger.info(f"加载模型 {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=load_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if device_map is None:
        model = model.to(target_device)
        logger.info(f"模型已加载到 {target_device}")

    model_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"模型参数量: {model_params:.1f}M")

    # 记录注意力配置信息
    attn_info_path = os.path.join(args.output_dir, f"{args.model_name}_attn_info.json")
    os.makedirs(args.output_dir, exist_ok=True)
    attn_info = {
        "model": args.model_name,
        "model_id": model_id,
        "attn_type": attn_type,
        "params_M": round(model_params, 1),
    }
    # 尝试获取 num_key_value_heads
    mc = model.config
    attn_info["num_attention_heads"] = getattr(mc, "num_attention_heads", None)
    attn_info["num_key_value_heads"] = getattr(mc, "num_key_value_heads", None)
    with open(attn_info_path, "w") as f:
        json.dump(attn_info, f, indent=2)

    run_output = os.path.join(args.output_dir, f"run_{args.model_name}_{args.dataset}")
    training_args = TrainingArguments(
        output_dir=run_output,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=use_fp16,
        bf16=use_bf16,   # MPS bfloat16 训练
        # Adafactor：CPU/MPS 上替换 Adam，优化器状态从 O(2n) 降至 O(√n)
        optim="adafactor" if not use_fp16 else "adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        prediction_loss_only=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    metrics_cb = MetricsCallback(
        args=args,
        model_name=args.model_name,
        attn_type=attn_type,
        dataset_name=dataset_label,
    )

    # 非 CUDA 设备用 Adafactor（固定 lr 模式），显著降低优化器内存
    optimizer = None
    if not use_fp16:
        from transformers.optimization import Adafactor
        optimizer = Adafactor(
            model.parameters(),
            lr=args.learning_rate,
            scale_parameter=False,   # 固定学习率，不自适应缩放
            relative_step=False,     # 关闭相对步长（配合固定 lr 使用）
            warmup_init=False,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[metrics_cb],
        optimizers=(optimizer, None) if optimizer is not None else (None, None),
    )

    logger.info("开始训练 ...")
    train_result = trainer.train()

    # 保存训练摘要
    summary = {
        "model": args.model_name,
        "attn_type": attn_type,
        "dataset": dataset_label,
        "train_runtime_s": round(train_result.metrics.get("train_runtime", 0), 1),
        "train_samples_per_second": round(
            train_result.metrics.get("train_samples_per_second", 0), 2
        ),
        "epoch_records": metrics_cb.rows,
    }
    summary_path = os.path.join(args.output_dir, f"summary_{args.model_name}_{args.dataset}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"训练完成，摘要保存至 {summary_path}")
    return metrics_cb.rows


def parse_args():
    parser = argparse.ArgumentParser(description="因果语言建模实验")
    parser.add_argument("--model_name", default="opt-1.3b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--dataset", default="wikitext", choices=["wikitext", "openwebtext", "both"])
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=5000)
    parser.add_argument("--results_csv", default="./results/results_lm.csv")
    parser.add_argument("--owt_subset_size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    exp_args = LMExperimentArgs(**vars(cli))
    run_experiment(exp_args)
