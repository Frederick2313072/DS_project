"""
语言建模实验结果可视化脚本
读取 results_lm.csv，生成：
  1. 折线图：Loss vs Epoch（各模型/数据集）
  2. 折线图：Perplexity vs Epoch
  3. 箱线图：PPL 分布（各注意力类型）
  4. 散点图：tokens/sec vs PPL（速度-质量权衡）
  5. 散点图：GPU 显存 vs PPL
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# 注意力类型颜色映射
ATTN_COLORS = {
    "MHA": "#2196F3",
    "GQA": "#4CAF50",
    "MQA": "#FF9800",
}

MODEL_MARKERS = {
    "llama3-1b": "o",
    "falcon-1b": "s",
    "mpt-1b": "^",
    "opt-1.3b": "D",
}


def load_results(csv_path: str) -> pd.DataFrame:
    assert os.path.exists(csv_path), f"结果文件不存在: {csv_path}"
    df = pd.read_csv(csv_path)
    # 去除 PPL 异常大的行（训练初期 loss 极高时）
    df = df[df["perplexity"] < 1e6].copy()
    return df


def plot_loss_curves(df: pd.DataFrame, save_dir: str):
    """折线图：各模型 Validation Loss vs Epoch"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    datasets = df["dataset"].unique()
    for ax, ds in zip(axes, datasets[:2]):
        sub = df[df["dataset"] == ds]
        for model in sub["model"].unique():
            mdf = sub[sub["model"] == model].sort_values("epoch")
            attn = mdf["attn_type"].iloc[0]
            color = ATTN_COLORS.get(attn, "#9E9E9E")
            marker = MODEL_MARKERS.get(model, "o")
            ax.plot(
                mdf["epoch"],
                mdf["eval_loss"],
                marker=marker,
                color=color,
                label=f"{model} ({attn})",
                linewidth=2,
                markersize=6,
            )
        ax.set_title(f"Validation Loss — {ds}", fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Eval Loss")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, "loss_vs_epoch.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"保存: {path}")


def plot_ppl_curves(df: pd.DataFrame, save_dir: str):
    """折线图：各模型 Perplexity vs Epoch"""
    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        for model in sub["model"].unique():
            mdf = sub[sub["model"] == model].sort_values("epoch")
            attn = mdf["attn_type"].iloc[0]
            color = ATTN_COLORS.get(attn, "#9E9E9E")
            marker = MODEL_MARKERS.get(model, "o")
            ax.plot(
                mdf["epoch"],
                mdf["perplexity"],
                marker=marker,
                color=color,
                label=f"{model} ({attn})",
                linewidth=2,
                markersize=6,
            )
        ax.set_title(f"Perplexity — {ds}", fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Perplexity")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, "ppl_vs_epoch.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"保存: {path}")


def plot_ppl_boxplot(df: pd.DataFrame, save_dir: str):
    """箱线图：不同注意力类型的 PPL 分布"""
    attn_groups = {at: df[df["attn_type"] == at]["perplexity"].values for at in df["attn_type"].unique()}
    labels = list(attn_groups.keys())
    data = [attn_groups[k] for k in labels]
    colors = [ATTN_COLORS.get(k, "#9E9E9E") for k in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("PPL Distribution by Attention Type", fontsize=13)
    ax.set_xlabel("Attention Type")
    ax.set_ylabel("Perplexity")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # 添加各组均值标注
    for i, (k, vals) in enumerate(attn_groups.items(), start=1):
        if len(vals) > 0:
            ax.text(
                i, np.mean(vals), f"μ={np.mean(vals):.1f}",
                ha="center", va="bottom", fontsize=9, color="black"
            )

    plt.tight_layout()
    path = os.path.join(save_dir, "ppl_boxplot_attn.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"保存: {path}")


def plot_speed_scatter(df: pd.DataFrame, save_dir: str):
    """散点图：tokens/sec vs PPL（速度-质量权衡）"""
    # 取各模型最后一个 epoch 的数值
    last_epoch = df.groupby(["model", "dataset"]).apply(
        lambda g: g.sort_values("epoch").iloc[-1]
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in last_epoch.iterrows():
        attn = row["attn_type"]
        model = row["model"]
        color = ATTN_COLORS.get(attn, "#9E9E9E")
        marker = MODEL_MARKERS.get(model, "o")
        ax.scatter(
            row["tokens_per_sec"],
            row["perplexity"],
            c=color,
            marker=marker,
            s=120,
            edgecolors="black",
            linewidths=0.8,
            label=f"{model} ({attn})",
            zorder=3,
        )
        ax.annotate(
            row["model"],
            xy=(row["tokens_per_sec"], row["perplexity"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Speed vs Perplexity (Final Epoch, lower-left is better)", fontsize=13)
    ax.set_xlabel("Tokens / sec (Training Throughput)")
    ax.set_ylabel("Perplexity (lower is better)")

    # 去重图例
    handles, labels_ = ax.get_legend_handles_labels()
    seen = {}
    dedup_h, dedup_l = [], []
    for h, l in zip(handles, labels_):
        if l not in seen:
            seen[l] = True
            dedup_h.append(h)
            dedup_l.append(l)
    ax.legend(dedup_h, dedup_l, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, "speed_vs_ppl.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"保存: {path}")


def plot_gpu_mem_scatter(df: pd.DataFrame, save_dir: str):
    """散点图：GPU 显存 vs PPL"""
    last_epoch = df.groupby(["model", "dataset"]).apply(
        lambda g: g.sort_values("epoch").iloc[-1]
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in last_epoch.iterrows():
        attn = row["attn_type"]
        model = row["model"]
        color = ATTN_COLORS.get(attn, "#9E9E9E")
        marker = MODEL_MARKERS.get(model, "o")
        ax.scatter(
            row["gpu_mem_mb"],
            row["perplexity"],
            c=color,
            marker=marker,
            s=120,
            edgecolors="black",
            linewidths=0.8,
            label=f"{model} ({attn})",
            zorder=3,
        )
        ax.annotate(
            row["model"],
            xy=(row["gpu_mem_mb"], row["perplexity"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("GPU Memory vs Perplexity (Final Epoch)", fontsize=13)
    ax.set_xlabel("GPU Memory (MB)")
    ax.set_ylabel("Perplexity (lower is better)")

    handles, labels_ = ax.get_legend_handles_labels()
    seen = {}
    dedup_h, dedup_l = [], []
    for h, l in zip(handles, labels_):
        if l not in seen:
            seen[l] = True
            dedup_h.append(h)
            dedup_l.append(l)
    ax.legend(dedup_h, dedup_l, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, "gpu_mem_vs_ppl.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"保存: {path}")


def plot_step_loss_curves(results_dir: str, save_dir: str):
    """折线图：step 级别训练 loss（原始 + 滑动平均），分析收敛平滑度"""
    import glob
    step_files = sorted(glob.glob(os.path.join(results_dir, "step_loss_*.csv")))
    if not step_files:
        print("未找到 step_loss_*.csv，跳过 step 级别分析")
        return

    df_step = pd.concat([pd.read_csv(f) for f in step_files], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 左：原始 step loss
    ax = axes[0]
    for (model, dataset), gdf in df_step.groupby(["model", "dataset"]):
        gdf = gdf.sort_values("step")
        attn = gdf["attn_type"].iloc[0]
        color = ATTN_COLORS.get(attn, "#9E9E9E")
        ax.plot(gdf["step"], gdf["train_loss"], color=color,
                label=f"{model} ({attn})", linewidth=1.0, alpha=0.7)
    ax.set_title("Training Loss vs Step (Raw)", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # 右：滑动平均平滑后的 loss（窗口 = 步数 / 20）
    ax = axes[1]
    for (model, dataset), gdf in df_step.groupby(["model", "dataset"]):
        gdf = gdf.sort_values("step")
        attn = gdf["attn_type"].iloc[0]
        color = ATTN_COLORS.get(attn, "#9E9E9E")
        window = max(5, len(gdf) // 20)
        smoothed = gdf["train_loss"].rolling(window=window, min_periods=1).mean()
        ax.plot(gdf["step"], smoothed, color=color,
                label=f"{model} ({attn}, w={window})", linewidth=2.0)
    ax.set_title("Training Loss vs Step (Smoothed)", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss (smoothed)")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(save_dir, "step_loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"保存: {path}")


def plot_convergence_smoothness(df: pd.DataFrame, save_dir: str):
    """Loss 变化率折线图：分析收敛曲线平滑度"""
    fig, ax = plt.subplots(figsize=(10, 5))

    for (model, dataset), gdf in df.groupby(["model", "dataset"]):
        gdf = gdf.sort_values("epoch")
        if len(gdf) < 2:
            continue
        delta_loss = gdf["eval_loss"].diff().abs().fillna(0)
        attn = gdf["attn_type"].iloc[0]
        color = ATTN_COLORS.get(attn, "#9E9E9E")
        marker = MODEL_MARKERS.get(model, "o")
        ax.plot(
            gdf["epoch"],
            delta_loss,
            marker=marker,
            color=color,
            label=f"{model}({dataset})",
            linewidth=1.8,
            markersize=5,
        )

    ax.set_title("|ΔLoss| vs Epoch (Convergence Smoothness)", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|ΔEval Loss|")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, "loss_delta.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"保存: {path}")


def generate_summary_table(df: pd.DataFrame, save_dir: str):
    """生成汇总统计表并保存为 CSV"""
    last = df.groupby(["model", "attn_type", "dataset"]).apply(
        lambda g: g.sort_values("epoch").iloc[-1]
    ).reset_index(drop=True)

    cols = ["model", "attn_type", "dataset", "epoch", "eval_loss", "perplexity", "tokens_per_sec", "gpu_mem_mb"]
    summary = last[cols].sort_values(["dataset", "perplexity"])

    path = os.path.join(save_dir, "summary_table.csv")
    summary.to_csv(path, index=False)
    print(f"保存汇总表: {path}")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="LM 实验可视化")
    parser.add_argument("--csv", default="./results/results_lm.csv")
    parser.add_argument("--save_dir", default="./results/figures")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    df = load_results(args.csv)
    print(f"加载 {len(df)} 条记录，模型: {df['model'].unique().tolist()}")

    plot_loss_curves(df, args.save_dir)
    plot_ppl_curves(df, args.save_dir)
    plot_ppl_boxplot(df, args.save_dir)
    plot_speed_scatter(df, args.save_dir)
    plot_gpu_mem_scatter(df, args.save_dir)
    plot_convergence_smoothness(df, args.save_dir)
    plot_step_loss_curves(os.path.dirname(args.csv), args.save_dir)
    generate_summary_table(df, args.save_dir)

    print("\n所有图表已保存至:", args.save_dir)


if __name__ == "__main__":
    main()
