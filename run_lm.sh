#!/usr/bin/env bash
# ============================================================
# 语言建模实验一键运行脚本
# 用法：
#   bash run_lm.sh                         # 完整实验（读 configs/lm_config.yaml）
#   bash run_lm.sh --skip_done             # 跳过已完成的组合
#   bash run_lm.sh single opt-1.3b wikitext  # 只跑单个实验
# ============================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 禁用 HuggingFace XET 协议，使用普通 HTTP 下载（XET 在某些网络环境会卡住）
export HF_HUB_DISABLE_XET=1

# 安装依赖（首次运行）
pip install -q -r requirements.txt

# ---- 单个实验模式 ----
if [[ "${1:-}" == "single" ]]; then
    MODEL=${2:-opt-1.3b}
    DATASET=${3:-wikitext}
    echo ">>> 单次实验: model=$MODEL  dataset=$DATASET"
    EXTRA=("${@:4}")
    python src/train_lm.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --output_dir ./results \
        --results_csv ./results/results_lm.csv \
        ${EXTRA[@]+"${EXTRA[@]}"}
    echo ">>> 生成可视化..."
    python src/visualize_lm.py --csv ./results/results_lm.csv --save_dir ./results/figures
    exit 0
fi

# ---- 批量实验模式 ----
echo ">>> 批量实验（读取 configs/lm_config.yaml）"
python src/run_all.py --config configs/lm_config.yaml "${@}"

echo ">>> 生成可视化图表..."
python src/visualize_lm.py \
    --csv ./results/results_lm.csv \
    --save_dir ./results/figures

echo ""
echo "实验完成！结果文件："
echo "  表格: ./results/results_lm.csv"
echo "  图表: ./results/figures/"
