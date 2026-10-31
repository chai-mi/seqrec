"""Read independent experiment JSONs and export the three study figure sets.

Run ``uv run python -m seqrec.analysis.plot_results`` after the training sweeps.
Markdown/Typst tables are printed to stdout; only SVG/PDF artifacts are written
to the figure directory. Missing experiments and empty buckets stay missing.
No metric is synthesized, and duplicate configurations are rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import altair as alt
import orjson
import polars as pl
import tyro
import vl_convert as vlc
from rich.console import Console

from seqrec.config import LEN_BUCKETS, POP_BUCKETS, default_max_seq_len

MODELS: dict[str, str] = {
    "pop": "Pop",
    "gru4rec": "GRU4Rec",
    "sasrec": "SASRec",
    "bert4rec": "BERT4Rec",
    "srgnn": "SR-GNN",
}
MODEL_ORDER = list(MODELS.values())
COLORS = ["#7f7f7f", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]
METRICS = ("HR@10", "NDCG@10")
MAIN_BERT_MASK_RATIO = 0.1
WINDOWS = {"beauty": (5, 10, 20, 50), "ml-1m": (10, 20, 50, 100, 200)}
SCHEMA = {
    "dataset": pl.String,
    "model": pl.String,
    "split": pl.String,
    "window": pl.Int64,
    "seed": pl.Int64,
    "mask_ratio": pl.Float64,
    "group": pl.String,
    "bucket": pl.String,
    "metric": pl.String,
    "value": pl.Float64,
    "count": pl.Int64,
    "experiment": pl.String,
    "source": pl.String,
    "partition": pl.String,
    "num_negatives": pl.Int64,
}


@dataclass(slots=True)
class PlotConfig:
    """Select a seed and independently regenerable vector figure/table outputs."""

    metrics_dir: Path = Path("reports/metrics")
    figures_dir: Path = Path("reports/figures")
    seed: int = 42
    table_format: Literal["markdown", "typst"] = "markdown"
    strict: bool = False
    """Require all 45 main model/window cells and all three study figure sets."""


def metric_value(metrics: dict[str, Any], key: str, source: Path) -> float | None:
    """Validate finite numeric values, retaining absent values as null."""
    value = metrics.get(key)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{source}: invalid numeric metric {key}={value!r}")
    if value < 0 or (key != "ARP@10" and not key.endswith("/count") and value > 1):
        raise ValueError(f"{source}: out-of-range metric {key}={value}")
    return float(value)


def read_metrics(metrics_dir: Path, seed: int = 42) -> pl.DataFrame:
    """Normalize trainer JSONs, ignoring summaries and other non-experiment JSONs.

    Select the common d=64, CE, tied-embedding configuration. Refuse multiple
    records for the same model/window/mask ratio rather than silently averaging
    different runs. Counts are carried into the table, including empty buckets.
    """
    rows: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    for source in sorted(metrics_dir.glob("*.json")):
        record = orjson.loads(source.read_bytes())
        if (
            not isinstance(record, dict)
            or "config" not in record
            or "test_metrics" not in record
        ):
            continue
        cfg = record["config"]
        data, model, train = cfg["data"], cfg["model"], cfg["train"]
        name = model["name"]
        if name not in MODELS or train["seed"] != seed:
            continue
        if (
            model["hidden_size"] != 64
            or model["loss_type"] != "ce"
            or not model["share_item_embeddings"]
        ):
            continue
        if cfg["eval"].get("oracle_middle_mask", False):
            continue
        dataset = data["dataset"]
        if dataset not in WINDOWS:
            continue
        window = data["max_seq_len"] or default_max_seq_len(dataset)
        mask_ratio = (
            float(data.get("last_item_mask_ratio", MAIN_BERT_MASK_RATIO))
            if name == "bert4rec"
            else None
        )
        identity = (dataset, name, data["split_strategy"], window, seed, mask_ratio)
        if identity in identities:
            raise ValueError(f"Duplicate experiment configuration {identity}: {source}")
        identities.add(identity)
        base = {
            "dataset": dataset,
            "model": MODELS[name],
            "split": data["split_strategy"],
            "window": window,
            "seed": seed,
            "mask_ratio": mask_ratio,
            "experiment": record["exp_name"],
            "source": source.name,
            "num_negatives": cfg["eval"]["num_neg_samples"],
        }
        for partition, metrics in (
            ("test", record["test_metrics"]),
            ("val", record["best_val_metrics"]),
        ):
            count = int(metrics.get("num_queries", 0))
            if count < 1:
                raise ValueError(
                    f"{source}: {partition} metrics have no evaluated queries"
                )
            for key in (*METRICS, "ARP@10", "Coverage@10"):
                rows.append(
                    dict(
                        base,
                        partition=partition,
                        group="overall",
                        bucket="All",
                        metric=key,
                        value=metric_value(metrics, key, source),
                        count=count,
                    )
                )
            for strategy in ("uniform", "popularity"):
                for key in METRICS:
                    rows.append(
                        dict(
                            base,
                            partition=partition,
                            group="sampled",
                            bucket=strategy,
                            metric=key,
                            value=metric_value(
                                metrics, f"sampled_{strategy}/{key}", source
                            ),
                            count=count,
                        )
                    )
            for group, buckets in (("len", LEN_BUCKETS), ("pop", POP_BUCKETS)):
                total = 0
                for bucket in buckets:
                    bucket_count = int(metrics.get(f"{group}:{bucket}/count", 0))
                    if bucket_count < 0:
                        raise ValueError(f"{source}: negative bucket count")
                    total += bucket_count
                    for key in METRICS:
                        value = (
                            metric_value(metrics, f"{group}:{bucket}/{key}", source)
                            if bucket_count
                            else None
                        )
                        if bucket_count and value is None:
                            raise ValueError(
                                f"{source}: missing {group}:{bucket}/{key}"
                            )
                        rows.append(
                            dict(
                                base,
                                partition=partition,
                                group=group,
                                bucket=bucket,
                                metric=key,
                                value=value,
                                count=bucket_count,
                            )
                        )
                if total != count:
                    raise ValueError(
                        f"{source}: {partition} {group} counts sum to {total}, expected {count}"
                    )
    if not rows:
        raise ValueError(f"No compatible completed experiment JSONs in {metrics_dir}")
    return (
        pl.DataFrame(rows, schema=SCHEMA)
        .lazy()
        .sort(
            "dataset",
            "model",
            "split",
            "window",
            "mask_ratio",
            "partition",
            "group",
            "bucket",
            "metric",
        )
        .collect()
    )


def main_model_filter() -> pl.Expr:
    """Keep every non-mask model; select the default last-item ratio only for BERT4Rec."""
    return (pl.col("model") != "BERT4Rec") | (
        pl.col("mask_ratio") == MAIN_BERT_MASK_RATIO
    )


def select_main(frame: pl.DataFrame, default_window: bool = True) -> pl.LazyFrame:
    """Select standard test LOO results; use prescribed windows, never test-best N."""
    query = frame.lazy().filter(
        (pl.col("partition") == "test")
        & (pl.col("split") == "loo")
        & main_model_filter()
    )
    if default_window:
        query = query.filter(
            ((pl.col("dataset") == "beauty") & (pl.col("window") == 50))
            | ((pl.col("dataset") == "ml-1m") & (pl.col("window") == 200))
        )
    return query


def missing_experiments(frame: pl.DataFrame) -> list[str]:
    """Report unobserved main sweep cells, including SR-GNN before it is trained."""
    observed = set(
        select_main(frame, default_window=False)
        .select("dataset", "model", "window")
        .unique()
        .collect()
        .iter_rows()
    )
    return [
        f"{dataset}/{model}/N={window}"
        for dataset, windows in WINDOWS.items()
        for model in MODEL_ORDER
        for window in windows
        if (dataset, model, window) not in observed
    ]


def table_rows(frame: pl.DataFrame) -> list[list[str]]:
    """Build a rectangular length × model × metric table, with explicit N/A."""
    cells = select_main(frame).filter(pl.col("group") == "len").collect()
    lookup = {
        (r["dataset"], r["bucket"], r["model"], r["metric"]): r
        for r in cells.to_dicts()
    }
    rows = [["Dataset", "N", "History length", "Queries", "Metric", *MODEL_ORDER]]
    for dataset in WINDOWS:
        for bucket in LEN_BUCKETS:
            for metric in METRICS:
                found = [
                    lookup.get((dataset, bucket, model, metric))
                    for model in MODEL_ORDER
                ]
                counts = {r["count"] for r in found if r is not None}
                if len(counts) > 1:
                    raise ValueError(
                        f"Incomparable query populations in {dataset}/{bucket}"
                    )
                values = [
                    f"{r['value']:.4f}"
                    if r is not None and r["value"] is not None
                    else "N/A"
                    for r in found
                ]
                rows.append(
                    [
                        dataset,
                        str(50 if dataset == "beauty" else 200),
                        bucket,
                        str(next(iter(counts))) if counts else "N/A",
                        metric,
                        *values,
                    ]
                )
    return rows


def format_table(
    rows: list[list[str]], table_format: Literal["markdown", "typst"]
) -> str:
    """Render a portable Markdown or Typst table without optional dependencies."""
    if table_format == "markdown":
        lines = [
            "| " + " | ".join(rows[0]) + " |",
            "| " + " | ".join(["---"] * len(rows[0])) + " |",
        ]
        return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows[1:]])
    escape = str.maketrans({char: "\\" + char for char in "\\[]#@$*_`<>"})
    cells = [f"  [{cell.translate(escape)}]," for row in rows for cell in row]
    return f"#table(columns: {len(rows[0])},\n" + "\n".join(cells) + "\n)"


def chart_base(frame: pl.DataFrame) -> alt.Chart:
    """Use inline records, avoiding pandas and process-global Altair settings.

    Local ty suppressions below cover Altair's generic ``mark_*`` return types.
    """
    return alt.Chart(alt.Data(values=frame.to_dicts()))


def model_color(models: list[str] | None = None) -> alt.Color:
    """Keep model identities and accessible colors stable in every figure."""
    order = MODEL_ORDER if models is None else models
    return alt.Color(
        "model:N",
        sort=order,
        scale=alt.Scale(
            domain=order, range=[COLORS[MODEL_ORDER.index(name)] for name in order]
        ),
        title="Model",
    )


def grouped_chart(
    frame: pl.DataFrame, title: str, bucket_order: list[str], subtitle: str
) -> alt.FacetChart:
    """Make paired dataset/metric facets with a stable bucket and model order."""
    base = chart_base(frame)
    bars = base.mark_bar().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("bucket:N", sort=bucket_order, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("model:N", sort=MODEL_ORDER),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=True)),
        color=model_color(),
        tooltip=[
            "dataset:N",
            "model:N",
            "bucket:N",
            "metric:N",
            alt.Tooltip("value:Q", format=".4f"),
            "count:Q",
            "source:N",
        ],
    )
    empty = (
        base.mark_text(text="N/A", color="#555555", fontSize=11)  # ty: ignore[unresolved-attribute]
        .transform_filter("datum.count === 0")
        .transform_aggregate(groupby=["dataset", "metric", "bucket"])
        .encode(x=alt.X("bucket:N", sort=bucket_order), y=alt.value(190))
    )
    return (
        (bars + empty)
        .properties(width=350, height=200)
        .facet(
            column=alt.Column("dataset:N", title=None),
            row=alt.Row("metric:N", title=None),
        )
        .properties(
            title=alt.Title(
                title, subtitle=[subtitle, "N/A indicates a bucket with no queries."]
            )
        )
    )


def length_table_chart(frame: pl.DataFrame) -> alt.FacetChart:
    """Export the numeric three-dimensional comparison table as a vector figure."""
    cells = pl.DataFrame(
        [
            {
                "dataset": row[0],
                "window": row[1],
                "bucket": row[2],
                "count": row[3],
                "metric": row[4],
                "model": model,
                "row": f"{row[2]} / {row[4]}",
                "label": row[index + 5],
            }
            for row in table_rows(frame)[1:]
            for index, model in enumerate(MODEL_ORDER)
        ]
    )
    order = [f"{bucket} / {metric}" for bucket in LEN_BUCKETS for metric in METRICS]
    base = chart_base(cells).encode(
        x=alt.X("model:N", sort=MODEL_ORDER, title=None, axis=alt.Axis(labelAngle=-25)),
        y=alt.Y("row:N", sort=order, title=None),
        tooltip=[
            "dataset:N",
            "window:Q",
            "model:N",
            "bucket:N",
            "metric:N",
            "label:N",
            "count:Q",
        ],
    )
    return (
        (
            base.mark_rect(color="#edf3f7")
            + base.mark_text(fontSize=12).encode(text="label:N")  # ty: ignore[unresolved-attribute]
        )
        .properties(
            width=370,
            height=270,
        )
        .facet(column=alt.Column("dataset:N", title=None))
        .properties(
            title=alt.Title(
                "History length × model × metric",
                subtitle=[
                    "Test Full-Ranking; N=50 (Beauty), N=200 (ML-1M); N/A = empty or missing"
                ],
            )
        )
    )


def mask_ablation_frame(frame: pl.DataFrame, metrics_dir: Path) -> pl.DataFrame:
    """Pair standard validation masks with existing Oracle validation diagnostics.

    Oracle is explicitly diagnostic: it sees one right-side future item and
    may retain one fewer old item at a full window. It never selects a model.
    """
    standard = (
        frame.lazy()
        .filter(
            (pl.col("model") == "BERT4Rec")
            & (pl.col("split") == "loo")
            & (pl.col("partition") == "val")
            & (pl.col("group") == "overall")
            & pl.col("metric").is_in(METRICS)
            & (
                ((pl.col("dataset") == "beauty") & (pl.col("window") == 50))
                | ((pl.col("dataset") == "ml-1m") & (pl.col("window") == 200))
            )
        )
        .with_columns(pl.lit("Standard end mask").alias("protocol"))
        .collect()
    )
    rows = standard.to_dicts()
    for dataset in WINDOWS:
        source = metrics_dir / f"leakage_bert4rec_mismatch_{dataset}.json"
        if not source.exists():
            continue
        summary = orjson.loads(source.read_bytes())
        for row in standard.to_dicts():
            if row["dataset"] != dataset:
                continue
            ratio = row["mask_ratio"]
            key = "pure_cloze" if ratio == 0 else f"last_mask_{ratio:.1f}"
            # Verify the summary belongs to this actual standard run before
            # attaching Oracle scores (do not mix stale sources).
            reference = (
                summary.get("standard_end_mask_loo_val", {})
                .get(key, {})
                .get(row["metric"])
            )
            if reference is None or not math.isclose(
                reference, row["value"], rel_tol=1e-9, abs_tol=1e-12
            ):
                continue
            value = metric_value(
                summary.get("oracle_middle_mask_loo_val", {}).get(key, {}),
                row["metric"],
                source,
            )
            if value is not None:
                rows.append(
                    dict(
                        row,
                        value=value,
                        protocol="Oracle (+1 future item)",
                        source=source.name,
                    )
                )
    return pl.DataFrame(rows, schema={**SCHEMA, "protocol": pl.String})


def length_buckets_chart(frame: pl.DataFrame) -> alt.FacetChart | None:
    """Compare model accuracy across untruncated history buckets."""
    lengths = select_main(frame).filter(pl.col("group") == "len").collect()
    if lengths.is_empty():
        return None
    return grouped_chart(
        lengths,
        "Performance by history length",
        list(LEN_BUCKETS),
        "Buckets use untruncated history; test Full-Ranking, one selected seed",
    )


def popularity_buckets_chart(frame: pl.DataFrame) -> alt.FacetChart | None:
    """Compare retrieval across target-item popularity buckets."""
    popularity = select_main(frame).filter(pl.col("group") == "pop").collect()
    if popularity.is_empty():
        return None
    return grouped_chart(
        popularity,
        "Long-tail retrieval",
        list(POP_BUCKETS),
        "Item-count quantiles: Head 20% / Mid 30% / Tail 50%; train-only frequency",
    )


def window_sensitivity_chart(frame: pl.DataFrame) -> alt.FacetChart | None:
    """Compare the prescribed truncation-window sweeps."""
    windows = (
        select_main(frame, default_window=False)
        .filter(
            (pl.col("group") == "overall")
            & pl.col("metric").is_in(METRICS)
            & pl.col("value").is_not_null()
        )
        .collect()
    )
    if windows.height:
        return (
            chart_base(windows)  # ty: ignore[unresolved-attribute]
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "window:Q",
                    title="Truncation window N",
                    scale=alt.Scale(type="log", base=2),
                    axis=alt.Axis(values=[5, 10, 20, 50, 100, 200], format="d"),
                ),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=True)),
                color=model_color(),
                tooltip=[
                    "dataset:N",
                    "model:N",
                    "window:Q",
                    "metric:N",
                    "value:Q",
                    "source:N",
                ],
            )
            .properties(width=350, height=200)
            .facet(column="dataset:N", row="metric:N")
            .properties(title="Window N sensitivity (test Full-Ranking)")
        )
    return None


def popularity_coverage_chart(frame: pl.DataFrame) -> alt.FacetChart | None:
    """Plot recommendation popularity against catalog coverage."""
    main = select_main(frame)
    bias = main.filter(
        (pl.col("group") == "overall")
        & pl.col("metric").is_in(["ARP@10", "Coverage@10"])
    ).collect()
    if bias.height:
        scatter = (
            bias.lazy()
            .group_by("dataset", "model", "source")
            .agg(
                pl.col("value")
                .filter(pl.col("metric") == "ARP@10")
                .first()
                .alias("ARP"),
                pl.col("value")
                .filter(pl.col("metric") == "Coverage@10")
                .first()
                .alias("coverage"),
            )
            .collect()
        )
        base = chart_base(scatter).encode(
            x=alt.X(
                "ARP:Q", title="ARP@10 (train frequency)", scale=alt.Scale(zero=True)
            ),
            y=alt.Y("coverage:Q", title="Coverage@10", scale=alt.Scale(domain=[0, 1])),
            color=model_color(),
        )
        return (
            base.mark_point(size=140, filled=True)  # ty: ignore[unresolved-attribute]
            .encode(
                shape=alt.Shape(
                    "model:N",
                    sort=MODEL_ORDER,
                    scale=alt.Scale(domain=MODEL_ORDER),
                    title="Model",
                ),
                tooltip=["dataset:N", "model:N", "ARP:Q", "coverage:Q", "source:N"],
            )
            .properties(width=350, height=250)
            .facet(column="dataset:N")
            .resolve_scale(x="independent")
            .properties(title="Popularity bias and catalog coverage")
        )
    return None


def bert4rec_masks_chart(
    frame: pl.DataFrame, metrics_dir: Path
) -> alt.FacetChart | None:
    """Compare BERT4Rec mask strategies and Oracle diagnostics."""
    masks = mask_ablation_frame(frame, metrics_dir)
    if masks.height:
        return (
            chart_base(masks)  # ty: ignore[unresolved-attribute]
            .mark_line(point=True)
            .encode(
                x=alt.X("mask_ratio:O", title="Last-item mask ratio"),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=True)),
                color=alt.Color("protocol:N", title="Validation protocol"),
                strokeDash="protocol:N",
                tooltip=[
                    "dataset:N",
                    "mask_ratio:Q",
                    "protocol:N",
                    "metric:N",
                    "value:Q",
                    "source:N",
                ],
            )
            .properties(width=350, height=200)
            .facet(column="dataset:N", row="metric:N")
            .properties(
                title=alt.Title(
                    "BERT4Rec masking ablation (LOO validation)",
                    subtitle=[
                        "Oracle sees 1 future item; full windows retain 1 fewer old item; diagnostic only"
                    ],
                )
            )
        )
    return None


def split_decay_chart(frame: pl.DataFrame) -> alt.FacetChart | None:
    """Compare paired LOO and temporal experiments at the default windows."""
    splits = (
        frame.lazy()
        .filter(
            (pl.col("partition") == "test")
            & (pl.col("group") == "overall")
            & main_model_filter()
            & pl.col("metric").is_in(METRICS)
            & (
                ((pl.col("dataset") == "beauty") & (pl.col("window") == 50))
                | ((pl.col("dataset") == "ml-1m") & (pl.col("window") == 200))
            )
        )
        .collect()
    )
    paired = (
        splits.lazy()
        .filter(pl.col("split") == "loo")
        .join(
            splits.lazy()
            .filter(pl.col("split") == "temporal")
            .select("dataset", "model", "metric", pl.col("value").alias("temporal")),
            on=["dataset", "model", "metric"],
            how="inner",
        )
        .with_columns(
            pl.when(pl.col("value") > 0)
            .then((1 - pl.col("temporal") / pl.col("value")) * 100)
            .otherwise(None)
            .alias("drop_pct")
        )
        .collect()
    )
    if paired.height:
        paired_models = [
            name for name in MODEL_ORDER if name in set(paired["model"].to_list())
        ]
        return (
            chart_base(paired)  # ty: ignore[unresolved-attribute]
            .mark_bar()
            .encode(
                x=alt.X(
                    "model:N",
                    sort=MODEL_ORDER,
                    title=None,
                    axis=alt.Axis(labelAngle=-25),
                ),
                y=alt.Y("drop_pct:Q", title="Relative decrease (%)"),
                color=model_color(paired_models),
                tooltip=[
                    "dataset:N",
                    "model:N",
                    "metric:N",
                    alt.Tooltip("value:Q", title="LOO"),
                    "temporal:Q",
                    "drop_pct:Q",
                ],
            )
            .properties(width=350, height=200)
            .facet(column="dataset:N", row="metric:N")
            .properties(
                title=alt.Title(
                    "LOO → global temporal performance change",
                    subtitle=[
                        "Different users, targets and histories: differences are not solely evidence of leakage"
                    ],
                )
            )
        )
    return None


def sampled_ranking_chart(frame: pl.DataFrame) -> alt.FacetChart | None:
    """Compare full ranking with both negative-sampling strategies."""
    main = select_main(frame)
    sampled = (
        main.filter(
            pl.col("group").is_in(["overall", "sampled"])
            & pl.col("metric").is_in(METRICS)
        )
        .with_columns(
            pl.when(pl.col("group") == "overall")
            .then(pl.lit("Full"))
            .otherwise(
                pl.lit("Sampled ")
                + pl.col("bucket")
                + pl.lit(" (")
                + pl.col("num_negatives").cast(pl.String)
                + pl.lit(")")
            )
            .alias("protocol")
        )
        .collect()
    )
    if (
        sampled.lazy()
        .filter((pl.col("group") == "sampled") & pl.col("value").is_not_null())
        .collect()
        .height
    ):
        return (
            chart_base(sampled)  # ty: ignore[unresolved-attribute]
            .mark_bar()
            .encode(
                x=alt.X(
                    "model:N",
                    sort=MODEL_ORDER,
                    title=None,
                    axis=alt.Axis(labelAngle=-25),
                ),
                xOffset="protocol:N",
                y=alt.Y("value:Q", title=None),
                color=alt.Color("protocol:N", title="Ranking protocol"),
                tooltip=[
                    "dataset:N",
                    "model:N",
                    "protocol:N",
                    "metric:N",
                    "value:Q",
                    "source:N",
                ],
            )
            .properties(width=350, height=200)
            .facet(column="dataset:N", row="metric:N")
            .properties(
                title=alt.Title(
                    "Full vs. sampled ranking",
                    subtitle=[
                        "Negatives sampled with replacement; draw count in legend; repeats count separately"
                    ],
                )
            )
        )
    return None


def build_charts(frame: pl.DataFrame, metrics_dir: Path) -> dict[str, alt.FacetChart]:
    """Assemble the independently defined study figures that have observations."""
    candidates = {
        "length_table": length_table_chart(frame),
        "length_buckets": length_buckets_chart(frame),
        "popularity_buckets": popularity_buckets_chart(frame),
        "window_sensitivity": window_sensitivity_chart(frame),
        "popularity_coverage": popularity_coverage_chart(frame),
        "bert4rec_masks": bert4rec_masks_chart(frame, metrics_dir),
        "split_decay": split_decay_chart(frame),
        "sampled_ranking": sampled_ranking_chart(frame),
    }
    return {name: chart for name, chart in candidates.items() if chart is not None}


def export_charts(charts: dict[str, alt.FacetChart], figures_dir: Path) -> list[Path]:
    """Convert Vega-Lite directly to SVG/PDF, using local vector rendering."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for name, chart in charts.items():
        spec = (
            chart.configure_axis(labelFontSize=11, titleFontSize=12)
            .configure_legend(
                orient="bottom",
                columns=5,
                labelFontSize=11,
            )
            .configure_title(fontSize=17, subtitleFontSize=11, anchor="start")
            .configure_view(stroke=None)
            .to_dict()
        )
        svg = figures_dir / f"{name}.svg"
        pdf = figures_dir / f"{name}.pdf"
        svg.write_text(vlc.vegalite_to_svg(spec), encoding="utf-8")
        pdf.write_bytes(vlc.vegalite_to_pdf(spec))
        outputs.extend((svg, pdf))
    return outputs


def main(config: PlotConfig | None = None) -> list[Path]:
    """Print the comparison table and regenerate every available study figure."""
    cfg = config if config is not None else tyro.cli(PlotConfig)
    console = Console()
    diagnostics = Console(stderr=True)
    frame = read_metrics(cfg.metrics_dir, cfg.seed)
    missing = missing_experiments(frame)
    charts = build_charts(frame, cfg.metrics_dir)
    if cfg.strict and (missing or len(charts) != 8):
        raise ValueError(
            f"Incomplete report: missing={missing}; available charts={list(charts)}"
        )
    console.print(
        format_table(table_rows(frame), cfg.table_format), markup=False, soft_wrap=True
    )
    if missing:
        diagnostics.print(
            f"Missing sweep cells ({len(missing)}): {', '.join(missing)}", markup=False
        )
    outputs = export_charts(charts, cfg.figures_dir)
    diagnostics.print(f"Exported {len(outputs)} vector files to {cfg.figures_dir}")
    return outputs


if __name__ == "__main__":
    main()
