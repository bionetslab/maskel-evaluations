"""Build two supplement LaTeX tables from the completed timing SLURM log and
benchmark/*_memory.py's CSVs: one runtime-only (with speedup vs. maskel), one
peak-RAM-only.

Runtime is read from the dedicated timing benchmark's log rather than the memory
benchmark's own (single-shot, no-repeat) wall_time_s column, since the timing
benchmark actually repeats each measurement (5x per VESSEL12 scan) for robustness.

HRF: mean +/- std across all 45 images (one measurement per image, no repeats).
VESSEL12: mean +/- std across all 20 scans' per-scan median (over 5 repeats) --
matches the dataset's true sample size (n=20 scans), rather than treating repeats
of the same scan as independent samples.

Speedup is method_time / maskel_time computed per sample, then mean +/- std across
samples (not a single ratio of the aggregate means) - consistent with how runtime
and peak RAM are already reported as per-sample distributions rather than single
numbers. maskel's own row is therefore always 1.00 $\\times$ $\\pm$ 0.00 by
construction, not a special case.
"""

import ast
import csv
import re
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
TIMING_LOG = _HERE.parent / "slurm" / "logs" / "maskel-benchmark_1498438.out"
HRF_MEMORY_CSV = _HERE.parent / "results" / "2d_memory.csv"
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

_HRF_ROW_RE = re.compile(
    r"^(?P<name>\d{2}_(?:dr|g|h))\s+"
    r"(?P<maskel>[\d.]+)\s+(?P<sk_zhang>[\d.]+)\s+(?P<sk_lee>[\d.]+)\s+(?P<vv_lee>[\d.]+)\s+"
    r"[\d.]+\s+[\d.]+\s+[\d.]+\s*$"
)
_VESSEL12_HEADER_RE = re.compile(r"^--- (VESSEL12_\d+) ---$")
_VESSEL12_MEDIAN_RE = re.compile(
    r"^MEDIAN\s+(?P<maskel>[\d.]+)\s+(?P<sk_lee>[\d.]+)\s+(?P<vv_lee>[\d.]+)\s+[\d.]+\s+[\d.]+\s*$"
)


def parse_hrf_runtime(log_text: str) -> dict[str, list[float]]:
    start = log_text.index("=== 2D comparison (HRF) ===")
    end = log_text.index("=== 3D comparison (VESSAP) ===")
    section = log_text[start:end]

    times = {m: [] for m in HRF_METHODS}
    for line in section.splitlines():
        m = _HRF_ROW_RE.match(line)
        if m:
            for method in HRF_METHODS:
                times[method].append(float(m.group(method)))

    if len(times["maskel"]) != 45:
        raise RuntimeError(f"Expected 45 HRF timing rows, parsed {len(times['maskel'])}")
    return times


def parse_vessel12_runtime(log_text: str) -> dict[str, list[float]]:
    start = log_text.index("=== 3D lung comparison")
    end = log_text.index("AGGREGATE (all runs, all scans)")
    section = log_text[start:end]

    times = {m: [] for m in LUNG_METHODS}
    current_scan = None
    seen_scans = set()
    for line in section.splitlines():
        header_match = _VESSEL12_HEADER_RE.match(line)
        if header_match:
            current_scan = header_match.group(1)
            continue
        median_match = _VESSEL12_MEDIAN_RE.match(line)
        if median_match and current_scan is not None and current_scan not in seen_scans:
            for method in LUNG_METHODS:
                times[method].append(float(median_match.group(method)))
            seen_scans.add(current_scan)

    if len(times["maskel"]) != 20:
        raise RuntimeError(f"Expected 20 VESSEL12 scan medians, parsed {len(times['maskel'])}")
    return times


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


def build_rows(
    name: str, methods: list[str], *value_lists, dataset_cols: list[str] = ()
) -> list[str]:
    """*value_lists: one or more ({method: [values...]}, fmt_fn) pairs, each
    becoming one per-method formatted column. dataset_cols: already-formatted
    strings describing the dataset itself rather than any one method (e.g. image
    dimensions, foreground pixel count) - multirow-merged across the dataset's
    method rows exactly like `name` is, since they're the same for every method."""
    n = len(methods)
    lines = []
    for i, m in enumerate(methods):
        if i == 0:
            merged = [f"\\multirow{{{n}}}{{*}}{{{name}}}"]
            merged += [f"\\multirow{{{n}}}{{*}}{{{c}}}" for c in dataset_cols]
        else:
            merged = [""] * (1 + len(dataset_cols))
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
    log_text = TIMING_LOG.read_text()
    hrf_runtime = parse_hrf_runtime(log_text)
    lung_runtime = parse_vessel12_runtime(log_text)
    hrf_speedup = compute_speedups(hrf_runtime, HRF_METHODS)
    lung_speedup = compute_speedups(lung_runtime, LUNG_METHODS)

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
