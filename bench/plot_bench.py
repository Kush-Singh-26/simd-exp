#!/usr/bin/env python3
"""Generalized Google Benchmark JSON visualizer.

Auto-detects benchmark structure and produces comparison plots.

Usage:
    python bench/plot_bench.py                              # default: matmul
    python bench/plot_bench.py -f bench/results/abs_bench.json
    python bench/plot_bench.py --all                        # all JSON files
    python bench/plot_bench.py --list                       # show data table
    python bench/plot_bench.py -m cpu_time                  # override y-axis
    python bench/plot_bench.py --clean                      # delete all plots
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ── Parsing ───────────────────────────────────────────────────────────────────

def try_num(s):
    for t in (int, float):
        try:
            return t(s)
        except (ValueError, TypeError):
            pass
    return s


def parse_name(name):
    """Parse e.g. 'BM_Gemm/ijk/M:64/N:64/K:64' or 'BM_Abs_Scalar/pos/1048576'."""
    parts = name.split("/")
    raw_suite = parts[0][3:] if parts[0].startswith("BM_") else parts[0]
    # Split suite on first underscore: "Abs_Scalar" → suite="Abs", impl="Scalar"
    if "_" in raw_suite:
        i = raw_suite.index("_")
        suite = raw_suite[:i]
        impl = raw_suite[i + 1:]
    else:
        suite = raw_suite
        impl = ""
    params = {}
    variant_parts = [impl] if impl else []
    for part in parts[1:]:
        if ":" in part:
            k, v = part.split(":", 1)
            params[k] = try_num(v)
        else:
            v = try_num(part)
            if isinstance(v, str):
                variant_parts.append(v)
            else:
                params[f"_{len(params)}"] = v
    return suite, "/".join(variant_parts), params


def extract_metrics(b):
    metrics = {}
    if "real_time" in b and "time_unit" in b:
        scale = 1e-9 if b["time_unit"] == "ns" else 1
        metrics["real_time"] = b["real_time"] * scale
        metrics["cpu_time"] = b["cpu_time"] * scale
    skip = {"name", "family_index", "per_family_instance_index", "run_name",
            "run_type", "repetitions", "repetition_index", "threads",
            "iterations", "real_time", "cpu_time", "time_unit"}
    for k, v in b.items():
        if k not in skip and isinstance(v, (int, float)):
            metrics[k] = v
    if "counters" in b and isinstance(b["counters"], dict):
        for k, v in b["counters"].items():
            if isinstance(v, (int, float)):
                metrics[k] = v
    return metrics


def parse_benchmarks(data):
    entries = []
    for b in data.get("benchmarks", []):
        suite, variant, params = parse_name(b["name"])
        metrics = extract_metrics(b)
        entries.append(dict(suite=suite, variant=variant,
                            params=params, metrics=metrics))
    return entries


# ── Display ───────────────────────────────────────────────────────────────────

def display_table(entries):
    if not entries:
        print("No entries.")
        return
    pk = sorted(set(k for e in entries for k in e["params"]))
    mk = sorted(set(k for e in entries for k in e["metrics"]))
    width = 22 + 11 * len(pk) + 14 * len(mk)
    print(f"\nSuite: {entries[0]['suite']}  ({len(entries)} entries)")
    print("─" * width)
    h = f"{'Variant':<22}"
    for k in pk:
        h += f"  {k:>8}"
    for k in mk:
        h += f"  {k:>12}"
    print(h)
    print("─" * width)
    for e in entries:
        r = f"{e['variant']:<22}"
        for k in pk:
            r += f"  {str(e['params'].get(k, '-')):>8}"
        for k in mk:
            m = e["metrics"].get(k, 0)
            r += f"  {m:>12.4g}" if isinstance(m, float) else f"  {str(m):>12}"
        print(r)
    print("─" * width)


# ── Variant splitting ─────────────────────────────────────────────────────────

def split_variant(variant):
    """'ijk_decode' → ('ijk', 'decode'). Returns (base, suffix)."""
    if "_" in variant:
        i = variant.rfind("_")
        return variant[:i], variant[i + 1:]
    return variant, ""


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6",
          "#f39c12", "#1abc9c", "#e67e22", "#2980b9"]


def plot_on_ax(ax, by_base, y_metric, title):
    all_entries = [e for ee in by_base.values() for e in ee]
    pk = sorted(set(k for e in all_entries for k in e["params"]))
    if not pk:
        ax.text(0.5, 0.5, "No parameters", ha="center", va="center")
        ax.set_title(title)
        return

    plot_bars_on_ax(ax, by_base, y_metric)

    ax.set_title(title)
    ax.set_ylabel(y_metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def plot_lines_on_ax(ax, by_base, x_param, y_metric):
    bases = sorted(by_base)
    for i, base in enumerate(bases):
        pts = sorted(by_base[base], key=lambda e: e["params"].get(x_param, 0))
        xs = [e["params"][x_param] for e in pts]
        ys = [e["metrics"].get(y_metric, 0) for e in pts]
        ax.plot(xs, ys, marker="o", label=base,
                color=COLORS[i % len(COLORS)], linewidth=2)
    ax.set_xlabel(x_param)


def plot_bars_on_ax(ax, by_base, y_metric):
    all_entries = [e for ee in by_base.values() for e in ee]
    pk = sorted(set(k for e in all_entries for k in e["params"]))

    # Group entries by param tuple
    groups = defaultdict(list)
    for e in all_entries:
        groups[tuple(e["params"].get(k, None) for k in pk)].append(e)

    bases = sorted(by_base)
    gkeys = sorted(groups)
    n, m = len(gkeys), len(bases)
    w = 0.8 / m
    x = list(range(n))

    for i, base in enumerate(bases):
        ys = []
        for gk in gkeys:
            match = [e for e in groups[gk] if e["variant"] == base or e["variant"].endswith("_" + base) or e["variant"].startswith(base + "_")]
            ys.append(match[0]["metrics"].get(y_metric, 0) if match else 0)
        off = (i - m / 2 + 0.5) * w
        ax.bar([xi + off for xi in x], ys, w, label=base,
               color=COLORS[i % len(COLORS)])

    ax.set_xticks(x)
    labels = []
    for gk in gkeys:
        uniq = set(gk)
        core = str(next(iter(uniq))) if len(uniq) == 1 else '×'.join(str(v) for v in gk)
        vs = [e['variant'] for e in groups[gk]]
        cats = set()
        for v in vs:
            if '_' in v:
                cats.add(v.rsplit('_', 1)[1])
        tag = '/'.join(sorted(cats)) if cats else ''
        labels.append(f'{tag}\n{core}' if tag else core)
    ax.set_xticklabels(labels, rotation=45, fontsize=6, ha='right')


# ── Auto plot ─────────────────────────────────────────────────────────────────

def auto_plot(entries, y_metric=None, output=None):
    if not entries:
        print("No entries to plot.")
        return

    suite = entries[0]["suite"]

    # Auto-detect metric
    all_metrics = set()
    for e in entries:
        all_metrics.update(e["metrics"].keys())
    if not y_metric:
        for p in ("GFLOPS", "items_per_second", "bytes_per_second",
                  "real_time", "cpu_time"):
            if p in all_metrics:
                y_metric = p
                break
        else:
            y_metric = sorted(all_metrics)[0] if all_metrics else "real_time"

    # Group by variant base (strip suffix like _decode, _nonsquare)
    by_base = defaultdict(list)
    for e in entries:
        base, _ = split_variant(e["variant"])
        by_base[base].append(e)

    n_grps = len(set(
        tuple(e["params"].get(k, None) for k in sorted(e["params"]))
        for ee in by_base.values() for e in ee
    ))

    fig, ax = plt.subplots(figsize=(max(8, n_grps * 1.0), 5))
    title = suite
    plot_on_ax(ax, dict(by_base), y_metric, title)

    plt.tight_layout()

    p = output or f"results/plots/{suite.lower()}_bench.png"
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved to {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

DEFAULT_FILE = "bench/results/matmul_bench.json"


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Google Benchmark JSON results")
    parser.add_argument("-f", "--file", default=DEFAULT_FILE,
                        help="Benchmark JSON file (default: %(default)s)")
    parser.add_argument("-m", "--metric",
                        help="Y-axis metric (default: auto-detect)")
    parser.add_argument("--list", action="store_true",
                        help="Show parsed data table instead of plotting")
    parser.add_argument("--output", help="Output PNG path (default: auto-named)")
    parser.add_argument("--clean", action="store_true",
                        help="Delete all generated plots")
    parser.add_argument("--all", action="store_true",
                        help="Generate plots for all JSON files in bench/results/")
    args = parser.parse_args()

    if plt is None:
        print("Error: matplotlib required. Install: pip install matplotlib",
              file=sys.stderr)
        sys.exit(1)

    if args.clean:
        import glob, os
        for p in glob.glob("results/plots/*.png"):
            os.remove(p)
            print(f"Removed {p}")
        print("All plots cleaned.")
        return

    if args.all:
        import glob
        for json_path in sorted(glob.glob("bench/results/*.json")):
            path = Path(json_path)
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            entries = parse_benchmarks(data)
            if entries:
                auto_plot(entries, y_metric=args.metric)
        return

    path = Path(args.file)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    entries = parse_benchmarks(data)
    if not entries:
        print("No benchmark entries found.", file=sys.stderr)
        sys.exit(1)

    if args.list:
        display_table(entries)
    else:
        auto_plot(entries, y_metric=args.metric, output=args.output)


if __name__ == "__main__":
    main()
