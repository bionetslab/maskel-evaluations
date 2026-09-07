"""Build two supplement LaTeX tables from benchmark/*_comparison.py's and
benchmark/*_memory.py's CSVs: one runtime-only (with speedup vs. maskel), one
peak-RAM-only.

HRF: mean +/- std across all 45 images (one measurement per image, no repeats).
VESSEL12: mean +/- std across all 20 scans' per-scan median (over 5 repeats) --
matches the dataset's true sample size (n=20 scans), rather than treating repeats
of the same scan as independent samples. read_runtime_csv() takes the median per
`sample` group either way, which is a no-op for HRF's single measurement per image.

Speedup is method_time / maskel_time computed per sample, then mean +/- std across
samples (not a single ratio of the aggregate means) - consistent with how runtime
and peak RAM are already reported as per-sample distributions rather than single
numbers. maskel's own row is therefore always 1.00 $\\times$ $\\pm$ 0.00 by
construction, not a special case.
"""

import ast
import csv
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
HRF_RUNTIME_CSV = _HERE.parent / "results" / "2d_HRF_runtime.csv"
HRF_MEMORY_CSV = _HERE.parent / "results" / "2d_HRF_memory.csv"
LUNG_RUNTIME_CSV = _HERE.parent / "results" / "3d_lung_runtime.csv"
LUNG_MEMORY_CSV = _HERE.parent / "results" / "3d_lung_memory.csv"
RUNTIME_TEX = _HERE.parent / "results" / "runtime_table.tex"
MEMORY_TEX = _HERE.parent / "results" / "memory_table.tex"

HRF_METHODS = ["maskel", "sk_zhang", "sk_lee", "vv_lee"]
LUNG_METHODS = ["maskel", "sk_lee", "vv_lee"]
METHOD_LABELS = {
    "maskel": "maskel",
    "sk_zhang": "scikit-image (zhang)",
    "sk_lee": "scikit-image (lee)",
    "vv_lee": "VesselVio (lee)",
}


def read_runtime_csv(path: Path, methods: list[str]) -> dict[str, list[float]]:
    """Per-sample method runtime, grouped by the CSV's `sample` column and reduced
    to the median across repeats - a no-op for a dataset with one row per sample
    (HRF) and a real median-of-5 for one with repeats (VESSEL12's scans)."""
    grouped: dict[str, dict[str, list[float]]] = {}
    order: list[str] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["sample"]
            if name not in grouped:
                grouped[name] = {m: [] for m in methods}
                order.append(name)
            for m in methods:
                grouped[name][m].append(float(row[f"{m}_time_s"]))
    return {m: [statistics.median(grouped[s][m]) for s in order] for m in methods}


def read_memory_csv(path: Path, methods: list[str]) -> dict[str, list[float]]:
    peaks = {m: [] for m in methods}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        n = 0
        for row in reader:
            for m in methods:
                peaks[m].append(float(row[f"{m}_peak_mb"]))
            n += 1
    return peaks, n


def read_dataset_stats(path: Path) -> tuple[list[tuple[int, ...]], list[int]]:
    """Per-sample (shape, foreground_px) from a *_memory.py CSV - these describe
    the input data itself, not any method, so unlike read_memory_csv's per-method
    columns there's exactly one list of each here regardless of how many methods
    were benchmarked on that dataset."""
    shapes = []
    foreground_px = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shapes.append(ast.literal_eval(row["shape"]))
            foreground_px.append(int(row["foreground_px"]))
    return shapes, foreground_px


def fmt(values: list[float]) -> str:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def fmt_speedup(values: list[float]) -> str:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f} $\\pm$ {std:.2f}$\\times$"


def fmt_image_dim(shapes: list[tuple[int, ...]]) -> str:
    """$d_1 \\times d_2 \\times ...$, per axis - a constant axis (same across every
    sample, e.g. HRF's fixed image size, or VESSEL12's in-plane 512x512) is shown
    directly; an axis that varies across samples (e.g. VESSEL12's scan depth) is
    shown as mean $\\pm$ std instead."""
    ndim = len(shapes[0])
    axes = []
    for axis in range(ndim):
        values = [s[axis] for s in shapes]
        if len(set(values)) == 1:
            axes.append(str(values[0]))
        else:
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            # bare \pm, not $\pm$ - this whole expression is one already-open
            # math region (see the enclosing "$"..."$" below), so wrapping \pm
            # in its own $...$ here would prematurely close/reopen math mode
            axes.append(f"({mean:.0f} \\pm {std:.0f})")
    return "$" + r"\times".join(axes) + "$"


def fmt_foreground_px(values: list[int]) -> str:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:,.0f} $\\pm$ {std:,.0f}".replace(",", "\\,")


def compute_speedups(
    runtimes: dict[str, list[float]], methods: list[str], baseline: str = "maskel"
) -> dict[str, list[float]]:
    """Per-sample method_time / baseline_time, for each method including the
    baseline itself (which is then always exactly 1.0 per sample, not a special
    case) - see module docstring for why this is per-sample rather than a single
    ratio of the aggregate means.
    """
    base_times = runtimes[baseline]
    return {
        m: [t / bt for t, bt in zip(runtimes[m], base_times, strict=True)]
        for m in methods
    }


def build_rows(name: str, methods: list[str], *value_lists) -> list[str]:
    """*value_lists: one or more ({method: [values...]}, fmt_fn) pairs, each
    becoming one per-method formatted column."""
    n = len(methods)
    lines = []
    for i, m in enumerate(methods):
        merged = [f"\\multirow{{{n}}}{{*}}{{{name}}}"] if i == 0 else [""]
        cols = " & ".join(fmt_fn(values[m]) for values, fmt_fn in value_lists)
        lines.append(" & ".join([*merged, METHOD_LABELS[m], cols]) + r" \\")
    return lines


def build_runtime_table(
    hrf_runtime, hrf_speedup, hrf_dims, hrf_fg, lung_runtime, lung_speedup, lung_dims, lung_fg
) -> str:
    # Image dim / foreground px are dataset-level facts, constant across every
    # method row for that dataset - stating them once in the caption instead of
    # as two more \multirow-merged table columns is what actually buys back the
    # width needed for this table to fit a single column in a two-column layout
    # (the two other levers, abbreviating headers and \footnotesize, are applied
    # around this call site / at the \input site respectively - see PR notes).
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Dataset & Method & Runtime (s) & Speedup vs.\\ maskel \\\\",
        "\\midrule",
        *build_rows(
            "HRF ($n=45$)", HRF_METHODS, (hrf_runtime, fmt), (hrf_speedup, fmt_speedup)
        ),
        "\\midrule",
        *build_rows(
            "VESSEL12 ($n=20$)",
            LUNG_METHODS,
            (lung_runtime, fmt),
            (lung_speedup, fmt_speedup),
        ),
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Runtime (mean $\\pm$ std) of \\emph{maskel} compared to "
        "scikit-image and VesselVio's Lee94 thinning implementations. HRF images "
        f"are {hrf_dims} pixels ({hrf_fg} foreground px per image); VESSEL12 "
        f"scans are {lung_dims} voxels ({lung_fg} foreground voxels per scan). "
        "Speedup is each method's per-sample runtime divided by \\emph{maskel}'s "
        "on the same sample (mean $\\pm$ std across samples); \\emph{maskel}'s "
        "own row is the $1.00\\times$ reference by construction.}",
        "\\label{tab:benchmark-runtime}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def build_memory_table(hrf_peak, lung_peak) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{llc}",
        "\\toprule",
        "Dataset & Method & Peak RAM (MB) \\\\",
        "\\midrule",
        *build_rows("HRF ($n=45$)", HRF_METHODS, (hrf_peak, fmt)),
        "\\midrule",
        *build_rows("VESSEL12 ($n=20$)", LUNG_METHODS, (lung_peak, fmt)),
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Peak memory usage (mean $\\pm$ std) of \\emph{maskel} compared "
        "to scikit-image and VesselVio's Lee94 thinning implementations.}",
        "\\label{tab:benchmark-memory}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main():
    if not HRF_RUNTIME_CSV.exists():
        print(
            f"{HRF_RUNTIME_CSV} not found -- run benchmark/2d_HRF_comparison.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not LUNG_RUNTIME_CSV.exists():
        print(
            f"{LUNG_RUNTIME_CSV} not found -- run benchmark/3d_lung_comparison.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    hrf_runtime = read_runtime_csv(HRF_RUNTIME_CSV, HRF_METHODS)
    if len(hrf_runtime["maskel"]) != 45:
        raise RuntimeError(f"Expected 45 HRF timing rows, found {len(hrf_runtime['maskel'])}")
    lung_runtime = read_runtime_csv(LUNG_RUNTIME_CSV, LUNG_METHODS)
    if len(lung_runtime["maskel"]) != 20:
        raise RuntimeError(
            f"Expected 20 VESSEL12 scan medians, found {len(lung_runtime['maskel'])}"
        )
    hrf_speedup = compute_speedups(hrf_runtime, HRF_METHODS)
    lung_speedup = compute_speedups(lung_runtime, LUNG_METHODS)

    if not HRF_MEMORY_CSV.exists() or HRF_MEMORY_CSV.stat().st_size == 0:
        print(
            f"{HRF_MEMORY_CSV} is empty -- run benchmark/2d_HRF_memory.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    hrf_peak, hrf_n = read_memory_csv(HRF_MEMORY_CSV, HRF_METHODS)
    if hrf_n != 45:
        raise RuntimeError(f"Expected 45 HRF memory rows, found {hrf_n}")

    if not LUNG_MEMORY_CSV.exists() or LUNG_MEMORY_CSV.stat().st_size == 0:
        print(
            f"{LUNG_MEMORY_CSV} is empty -- the VESSEL12 memory benchmark job "
            "hasn't finished yet. Re-run this script once it has.",
            file=sys.stderr,
        )
        sys.exit(1)
    lung_peak, lung_n = read_memory_csv(LUNG_MEMORY_CSV, LUNG_METHODS)
    if lung_n != 20:
        raise RuntimeError(f"Expected 20 VESSEL12 memory rows, found {lung_n}")

    hrf_shapes, hrf_fg_px = read_dataset_stats(HRF_MEMORY_CSV)
    lung_shapes, lung_fg_px = read_dataset_stats(LUNG_MEMORY_CSV)
    hrf_dims = fmt_image_dim(hrf_shapes)
    lung_dims = fmt_image_dim(lung_shapes)
    hrf_fg = fmt_foreground_px(hrf_fg_px)
    lung_fg = fmt_foreground_px(lung_fg_px)

    runtime_table = build_runtime_table(
        hrf_runtime, hrf_speedup, hrf_dims, hrf_fg, lung_runtime, lung_speedup, lung_dims, lung_fg
    )
    memory_table = build_memory_table(hrf_peak, lung_peak)

    print(runtime_table)
    print()
    print(memory_table)

    RUNTIME_TEX.parent.mkdir(exist_ok=True)
    RUNTIME_TEX.write_text(runtime_table + "\n")
    MEMORY_TEX.write_text(memory_table + "\n")
    print(f"\nWrote {RUNTIME_TEX}")
    print(f"Wrote {MEMORY_TEX}")


if __name__ == "__main__":
    main()
