import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl
    import torch

    from seqrec.config import DataConfig, DatasetName, ModelName, SplitStrategy
    from seqrec.data import build_databundle, load_dataset_metadata


@app.function
def load_summary_dataframe(data_dir: Path) -> pl.DataFrame:
    """Load preprocessed metadata for all datasets into a Polars DataFrame."""
    rows: list[dict[str, object]] = []
    for ds in ("beauty", "ml-1m"):
        meta = load_dataset_metadata(data_dir, ds)
        rows.append(
            {
                "Dataset": meta.dataset,
                "#Users": meta.num_users,
                "#Items": meta.num_items,
                "#Interactions": meta.num_interactions,
                "Density (%)": meta.density_pct,
                "Avg Len": meta.avg_seq_len,
                "Median Len": meta.median_seq_len,
                "P90 Len": meta.p90_seq_len,
                "Max Len": meta.max_seq_len,
                "<=5": meta.loo_test_len_buckets["<=5"],
                "6-15": meta.loo_test_len_buckets["6-15"],
                "16-50": meta.loo_test_len_buckets["16-50"],
                ">50": meta.loo_test_len_buckets[">50"],
            }
        )
    return pl.DataFrame(rows)


@app.function
def plot_seq_len_distribution(data_dir: Path) -> alt.Chart:
    """Plot LOO test sequence length bucket distributions across both datasets."""
    frames: list[pl.DataFrame] = []
    for ds_name in ("beauty", "ml-1m"):
        df = (
            pl.scan_parquet(data_dir / ds_name / "loo_test.parquet")
            .group_by("len_bucket")
            .agg(pl.len().alias("num_users"))
            .with_columns(
                pl.lit(ds_name).alias("dataset"),
                (pl.col("num_users") / pl.col("num_users").sum() * 100.0)
                .round(2)
                .alias("pct_users"),
            )
            .collect()
        )
        frames.append(df)
    return (
        # Altair's mark_bar generic return type is misinferred by ty (see AGENTS.md).
        alt.Chart(pl.concat(frames))  # ty: ignore[unresolved-attribute]
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "len_bucket:N",
                sort=["<=5", "6-15", "16-50", ">50"],
                title="Sequence Length Bucket |S_u|",
            ),
            y=alt.Y("pct_users:Q", title="Share of Users (%)"),
            color=alt.Color("dataset:N", title="Dataset"),
            xOffset="dataset:N",
            tooltip=[
                "dataset:N",
                "len_bucket:N",
                "num_users:Q",
                "pct_users:Q",
            ],
        )
        .properties(
            title="Sequence Length Bucket Distribution",
            width=380,
            height=250,
        )
    )


@app.function
def plot_item_popularity_curve(data_dir: Path) -> alt.Chart:
    """Plot interaction share across Head (20%), Mid (30%), and Tail (50%) item buckets."""
    frames: list[pl.DataFrame] = []
    for ds_name in ("beauty", "ml-1m"):
        df = (
            pl.scan_parquet(data_dir / ds_name / "loo_item_pop.parquet")
            .group_by("pop_bucket")
            .agg(
                pl.col("train_count").sum().alias("interactions"),
                pl.len().alias("num_items"),
            )
            .with_columns(
                pl.lit(ds_name).alias("dataset"),
                (pl.col("interactions") / pl.col("interactions").sum() * 100.0)
                .round(2)
                .alias("pct_interactions"),
            )
            .collect()
        )
        frames.append(df)
    return (
        # Keep the known Altair/ty workaround local to this chart expression.
        alt.Chart(pl.concat(frames))  # ty: ignore[unresolved-attribute]
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "pop_bucket:N",
                sort=["Head", "Mid", "Tail"],
                title="Item Popularity Bucket",
            ),
            y=alt.Y(
                "pct_interactions:Q",
                title="Share of Training Interactions (%)",
            ),
            color=alt.Color("dataset:N", title="Dataset"),
            xOffset="dataset:N",
            tooltip=[
                "dataset:N",
                "pop_bucket:N",
                "num_items:Q",
                "interactions:Q",
                "pct_interactions:Q",
            ],
        )
        .properties(
            title="Long-Tail Item Popularity Concentration",
            width=380,
            height=250,
        )
    )


@app.function
def inspect_dataloader_sample(
    dataset: DatasetName,
    split_strategy: SplitStrategy,
    model_name: ModelName,
    max_seq_len: int,
) -> pl.DataFrame:
    """Build a DataBundle and inspect one batch from train, val, and test DataLoaders."""
    cfg = DataConfig(
        dataset=dataset,
        split_strategy=split_strategy,
        max_seq_len=max_seq_len,
        batch_size=128,
        eval_batch_size=128,
    )
    bundle = build_databundle(cfg, model_name=model_name)
    rows: list[dict[str, object]] = []
    if bundle.train_loader is not None:
        train_batch = next(iter(bundle.train_loader))
        for k, tensor in train_batch.items():
            assert isinstance(tensor, torch.Tensor)
            rows.append(
                {
                    "split": "train",
                    "batches": len(bundle.train_loader),
                    "key": k,
                    "shape": str(tuple(tensor.shape)),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "min": float(tensor.min().item()),
                    "max": float(tensor.max().item()),
                }
            )
    for split_name, loader in (
        ("val", bundle.val_loader),
        ("test", bundle.test_loader),
    ):
        batch = next(iter(loader))
        for k, tensor in batch.items():
            assert isinstance(tensor, torch.Tensor)
            rows.append(
                {
                    "split": split_name,
                    "batches": len(loader),
                    "key": k,
                    "shape": str(tuple(tensor.shape)),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "min": float(tensor.min().item()),
                    "max": float(tensor.max().item()),
                }
            )
    return pl.DataFrame(rows)


@app.cell
def _():
    data_dir = Path("data/processed")
    summary_df = load_summary_dataframe(data_dir)
    mo.vstack(
        [
            mo.md(
                "# Phase 1 — 序列推荐数据底座探索与验证\n"
                "本笔记本对 **Amazon-Beauty**（短序列高稀疏）与 **MovieLens-1M**（长序列稠密）的 5-core 预处理结果、"
                "序列长度分桶、长尾物品热度分布及三类 PyTorch `Dataset`（Causal / Cloze / Session-Graph）进行交互式验证。"
            ),
            summary_df,
        ]
    )
    return (data_dir,)


@app.cell
def _(data_dir):
    len_chart = plot_seq_len_distribution(data_dir)
    pop_chart = plot_item_popularity_curve(data_dir)
    mo.hstack([len_chart, pop_chart], justify="start")
    return


@app.cell
def _():
    dataset_ui = mo.ui.dropdown(
        options=["beauty", "ml-1m"], value="beauty", label="Dataset"
    )
    split_ui = mo.ui.dropdown(
        options=["loo", "temporal"], value="loo", label="Split Strategy"
    )
    model_ui = mo.ui.dropdown(
        options=["sasrec", "gru4rec", "bert4rec", "srgnn", "pop"],
        value="sasrec",
        label="Model Paradigm",
    )
    seq_len_ui = mo.ui.slider(
        start=10, stop=200, step=10, value=50, label="Max Seq Len (N)"
    )
    mo.hstack([dataset_ui, split_ui, model_ui, seq_len_ui], justify="start")
    return dataset_ui, model_ui, seq_len_ui, split_ui


@app.cell
def _(dataset_ui, model_ui, seq_len_ui, split_ui):
    batch_summary_df = inspect_dataloader_sample(
        dataset=dataset_ui.value,
        split_strategy=split_ui.value,
        model_name=model_ui.value,
        max_seq_len=seq_len_ui.value,
    )
    mo.vstack(
        [
            mo.md(
                f"### DataLoader Batch Tensor Diagnostics (`{dataset_ui.value}` | `{split_ui.value}` | `{model_ui.value}` | $N={seq_len_ui.value}$)"
            ),
            batch_summary_df,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
