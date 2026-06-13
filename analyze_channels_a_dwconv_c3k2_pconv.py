import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ORIGINAL = Path(r"D:\JiQI\MM-experiment\ResTest\A-DWConv-C3k2_PConv\weights\best.pt")
DEFAULT_PRUNED = Path(r"D:\BaiduNetdiskDownload\MutilModel_3398475911\prune_outputs_depgraph_taylor_c3k2\pruned_r0.5_0430_0929.pt")
DEFAULT_OUTPUT = Path(r"D:\BaiduNetdiskDownload\MutilModel_3398475911\channel_compare_outputs")

MAJOR_TYPES = {
    "CSP_MutilScaleEdgeInformationEnhance",
    "C3k2_PConv",
    "TransformerEncoderLayer_DAttention",
    "RTDETRDecoder",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare original/pruned channel widths and export figures.")
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL, help="Original model .pt")
    parser.add_argument("--pruned", type=Path, default=DEFAULT_PRUNED, help="Pruned model .pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--top-convs", type=int, default=40, help="Max changed conv2d modules to plot")
    return parser.parse_args()


def load_model_from_checkpoint(path: Path) -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
    elif hasattr(ckpt, "model"):
        model = ckpt.model
    else:
        model = ckpt
    return model.float().eval()


def get_decoder_hidden_dim(head: nn.Module) -> int | None:
    decoder = getattr(head, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if layers and len(layers):
        first = layers[0]
        linear2 = getattr(first, "linear2", None)
        if linear2 is not None and hasattr(linear2, "out_features"):
            return int(linear2.out_features)
    hidden_dim = getattr(head, "hidden_dim", None)
    if hidden_dim is not None:
        return int(hidden_dim)
    return None


def extract_major_table(model: nn.Module) -> pd.DataFrame:
    rows = []
    core = getattr(model, "model", None)
    if core is None:
        return pd.DataFrame(columns=["layer_idx", "name", "type", "metric", "channels"])

    for idx, layer in enumerate(core):
        cls_name = type(layer).__name__
        metric = None
        value = None
        if cls_name in {"CSP_MutilScaleEdgeInformationEnhance", "C3k2_PConv"} and hasattr(layer, "c"):
            metric = "hidden_c"
            value = int(layer.c)
        elif cls_name == "TransformerEncoderLayer_DAttention" and hasattr(layer, "fc1"):
            metric = "ffn_hidden"
            value = int(layer.fc1.out_channels)
        elif cls_name == "RTDETRDecoder":
            metric = "decoder_hidden"
            value = get_decoder_hidden_dim(layer)

        if cls_name in MAJOR_TYPES and value is not None:
            rows.append(
                {
                    "layer_idx": idx,
                    "name": f"model.{idx}",
                    "type": cls_name,
                    "metric": metric,
                    "channels": value,
                }
            )

    return pd.DataFrame(rows)


def extract_conv2d_table(model: nn.Module) -> pd.DataFrame:
    rows = []
    order = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        rows.append(
            {
                "order": order,
                "name": name,
                "out_channels": int(module.out_channels),
                "in_channels": int(module.in_channels),
                "groups": int(module.groups),
                "kernel_size": f"{module.kernel_size[0]}x{module.kernel_size[1]}",
            }
        )
        order += 1
    return pd.DataFrame(rows)


def merge_compare(
    orig_df: pd.DataFrame,
    pruned_df: pd.DataFrame,
    key_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    merged = orig_df.merge(pruned_df, on=key_cols, how="outer", suffixes=("_orig", "_pruned"))
    merged[f"{value_col}_orig"] = merged[f"{value_col}_orig"].fillna(0).astype(int)
    merged[f"{value_col}_pruned"] = merged[f"{value_col}_pruned"].fillna(0).astype(int)
    orig_vals = merged[f"{value_col}_orig"]
    pruned_vals = merged[f"{value_col}_pruned"]
    merged["reduced"] = orig_vals - pruned_vals
    merged["remain_ratio"] = pruned_vals.where(orig_vals > 0, 0) / orig_vals.where(orig_vals > 0, 1)
    merged["reduced_ratio"] = 1.0 - merged["remain_ratio"]
    return merged


def plot_grouped_bar(
    df: pd.DataFrame,
    label_col: str,
    orig_col: str,
    pruned_col: str,
    title: str,
    output_path: Path,
    figsize: tuple[int, int],
) -> None:
    if df.empty:
        return

    labels = df[label_col].tolist()
    orig_vals = df[orig_col].tolist()
    pruned_vals = df[pruned_col].tolist()

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = list(range(len(labels)))
    bar_h = 0.4

    ax.barh([y - bar_h / 2 for y in y_pos], orig_vals, height=bar_h, label="Original", color="#9fb3c8")
    ax.barh([y + bar_h / 2 for y in y_pos], pruned_vals, height=bar_h, label="Pruned", color="#d66b4d")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Channels")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(loc="best")

    for y, val in zip([y + bar_h / 2 for y in y_pos], pruned_vals):
        ax.text(val + max(orig_vals) * 0.01, y, str(val), va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary(summary_path: Path, major_df: pd.DataFrame, conv_df: pd.DataFrame, original: Path, pruned: Path) -> None:
    lines = [
        f"original: {original}",
        f"pruned:   {pruned}",
        "",
        "[major modules]",
    ]

    if major_df.empty:
        lines.append("no major module rows found")
    else:
        for _, row in major_df.iterrows():
            lines.append(
                f"{row['name']} {row['type']} {row['metric']}: "
                f"{row['channels_orig']} -> {row['channels_pruned']} "
                f"(remain={row['remain_ratio']:.2%})"
            )

    lines.extend(["", "[top changed conv2d]"])
    if conv_df.empty:
        lines.append("no changed conv2d rows found")
    else:
        top_rows = conv_df.sort_values(["reduced", "out_channels_orig"], ascending=[False, False]).head(20)
        for _, row in top_rows.iterrows():
            lines.append(
                f"{row['name']}: out {row['out_channels_orig']} -> {row['out_channels_pruned']} "
                f"(remain={row['remain_ratio']:.2%}, k={row['kernel_size_orig']}, g={row['groups_orig']})"
            )

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    original_model = load_model_from_checkpoint(args.original)
    pruned_model = load_model_from_checkpoint(args.pruned)

    major_orig = extract_major_table(original_model)
    major_pruned = extract_major_table(pruned_model)
    major_cmp = merge_compare(major_orig, major_pruned, ["layer_idx", "name", "type", "metric"], "channels")
    major_cmp = major_cmp.sort_values("layer_idx").reset_index(drop=True)
    major_cmp["label"] = major_cmp["name"] + " " + major_cmp["type"]

    conv_orig = extract_conv2d_table(original_model)
    conv_pruned = extract_conv2d_table(pruned_model)
    conv_cmp = merge_compare(conv_orig, conv_pruned, ["name", "order"], "out_channels")
    conv_cmp = conv_cmp.sort_values("order").reset_index(drop=True)
    changed_conv_cmp = conv_cmp[conv_cmp["out_channels_orig"] != conv_cmp["out_channels_pruned"]].copy()
    changed_conv_cmp["label"] = changed_conv_cmp["name"]

    stem = args.pruned.stem
    major_csv = args.output / f"{stem}_major_modules.csv"
    conv_csv = args.output / f"{stem}_conv2d.csv"
    major_png = args.output / f"{stem}_major_modules.png"
    conv_png = args.output / f"{stem}_conv2d_top_changed.png"
    summary_txt = args.output / f"{stem}_summary.txt"

    major_cmp.to_csv(major_csv, index=False, encoding="utf-8-sig")
    conv_cmp.to_csv(conv_csv, index=False, encoding="utf-8-sig")

    plot_grouped_bar(
        major_cmp,
        label_col="label",
        orig_col="channels_orig",
        pruned_col="channels_pruned",
        title="Major Module Channel Comparison",
        output_path=major_png,
        figsize=(12, max(5, len(major_cmp) * 0.55)),
    )

    if not changed_conv_cmp.empty:
        conv_plot_df = changed_conv_cmp.sort_values(
            ["reduced", "out_channels_orig"], ascending=[False, False]
        ).head(max(1, args.top_convs))
        plot_grouped_bar(
            conv_plot_df,
            label_col="label",
            orig_col="out_channels_orig",
            pruned_col="out_channels_pruned",
            title=f"Top {len(conv_plot_df)} Changed Conv2d Out-Channels",
            output_path=conv_png,
            figsize=(14, max(8, len(conv_plot_df) * 0.35)),
        )

    save_summary(summary_txt, major_cmp, changed_conv_cmp, args.original, args.pruned)

    print(f"Saved: {major_csv}")
    print(f"Saved: {conv_csv}")
    print(f"Saved: {major_png}")
    if not changed_conv_cmp.empty:
        print(f"Saved: {conv_png}")
    print(f"Saved: {summary_txt}")
    print()
    print("Major modules:")
    for _, row in major_cmp.iterrows():
        print(
            f"  {row['name']} {row['type']} {row['metric']}: "
            f"{row['channels_orig']} -> {row['channels_pruned']} "
            f"(remain={row['remain_ratio']:.2%})"
        )
    print()
    print(f"Changed Conv2d rows: {len(changed_conv_cmp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
