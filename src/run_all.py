"""
批量实验入口：读取 configs/lm_config.yaml，依次跑每个 model × dataset 组合
用法：python src/run_all.py [--config configs/lm_config.yaml] [--skip_done]
"""

import os
import sys
import argparse
import logging
import yaml

# 将 src 目录加入 path，确保 train_lm 可被导入
sys.path.insert(0, os.path.dirname(__file__))
from train_lm import LMExperimentArgs, run_experiment, MODEL_CONFIGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="批量 LM 实验")
    parser.add_argument("--config", default="configs/lm_config.yaml")
    parser.add_argument(
        "--skip_done",
        action="store_true",
        help="若 results_csv 中已存在该 model+dataset 组合则跳过",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    experiments = cfg.pop("experiments", [])
    base_kwargs = {k: v for k, v in cfg.items() if v is not None}

    # 检查已完成的实验（用于 --skip_done）
    done_set: set[tuple] = set()
    if args.skip_done:
        csv_path = base_kwargs.get("results_csv", "./results/results_lm.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                done_set.add((row["model"], row["dataset"].split("+")[0]))

    total = len(experiments)
    for idx, exp in enumerate(experiments, start=1):
        model_name = exp["model_name"]
        dataset = exp.get("dataset", "wikitext")

        if args.skip_done and (model_name, dataset) in done_set:
            logger.info(f"[{idx}/{total}] 跳过已完成: {model_name} × {dataset}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx}/{total}] 开始实验: model={model_name}  dataset={dataset}")
        logger.info(f"{'='*60}")

        kwargs = {**base_kwargs, **exp}
        # yaml 中 null 转 None
        kwargs = {k: (None if v == "null" else v) for k, v in kwargs.items()}

        try:
            exp_args = LMExperimentArgs(**kwargs)
            run_experiment(exp_args)
        except Exception as e:
            logger.error(f"实验失败 {model_name} × {dataset}: {e}", exc_info=True)
            # 继续跑下一个，不中断整体流程
            continue

    logger.info("\n所有实验完成！")
    logger.info(f"结果保存于: {base_kwargs.get('results_csv', './results/results_lm.csv')}")
    logger.info("运行可视化: python src/visualize_lm.py")


if __name__ == "__main__":
    main()
