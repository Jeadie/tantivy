"""Summarizes `block_wand_union_bench` A/B results from the `Block-Max WAND bench` workflow.

Input: a directory of `*.tsv` result files and `*.meta` files written by the bench jobs.
Output: a Markdown report. The exit code is 1 if the baseline and the head return different
top-K results for a cell (the bench checksum covers doc addresses and score bits).
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# P99 needs at least 10 samples above it.
MIN_SAMPLES_FOR_P99 = 1_000
# Repetitions of the same binary that differ by more than this are too noisy to compare.
MAX_REPETITION_SPREAD = 0.20
# A verdict needs at least this many runs of each binary.
MIN_REPETITIONS = 2


@dataclass(frozen=True)
class Measurement:
    segments: int
    binary: str
    repetition: int
    mix: str
    terms: int
    k: int
    samples: int
    p99_us: float
    p999_us: float
    p999_ok: bool
    checksum: str

    @property
    def cell(self) -> tuple[str, int, int]:
        return (self.mix, self.terms, self.k)


@dataclass
class Cell:
    segments: int
    mix: str
    terms: int
    k: int
    base: list[Measurement] = field(default_factory=list)
    head: list[Measurement] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.mix}/{self.terms}/{self.k}"

    @property
    def results_differ(self) -> bool:
        checksums = {m.checksum for m in self.base} | {m.checksum for m in self.head}
        return len(checksums) > 1

    @property
    def min_samples(self) -> int:
        return min(m.samples for m in self.base + self.head)

    def change(self) -> float:
        """Median head P99 / median baseline P99 - 1."""
        base = statistics.median(m.p99_us for m in self.base)
        head = statistics.median(m.p99_us for m in self.head)
        return head / base - 1.0

    def verdict(self) -> str:
        if self.results_differ:
            return "**RESULTS DIFFER**"
        if not self.base or not self.head:
            return "missing"
        if min(len(self.base), len(self.head)) < MIN_REPETITIONS:
            return f"fewer than {MIN_REPETITIONS} runs"
        if self.min_samples < MIN_SAMPLES_FOR_P99:
            return f"too few samples (<{MIN_SAMPLES_FOR_P99})"
        if spread(self.base) > MAX_REPETITION_SPREAD or spread(self.head) > MAX_REPETITION_SPREAD:
            return "noisy"
        base = [m.p99_us for m in self.base]
        head = [m.p99_us for m in self.head]
        if max(head) < min(base):
            return "faster"
        if min(head) > max(base):
            return "slower"
        return "no clear change"


@dataclass(frozen=True)
class JobMeta:
    segments: int
    shard: str
    cpu: str
    load_before: str
    load_after: str


def spread(measurements: list[Measurement]) -> float:
    values = [m.p99_us for m in measurements]
    return max(values) / min(values) - 1.0 if len(values) > 1 else 0.0


def read_measurements(results_dir: Path) -> list[Measurement]:
    measurements = []
    for path in sorted(results_dir.glob("*.tsv")):
        for line in path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) != 11:
                continue
            seg, binary, rep, mix, terms, k, samples, p99, p999, p999_ok, checksum = parts
            measurements.append(
                Measurement(
                    segments=int(seg),
                    binary=binary,
                    repetition=int(rep),
                    mix=mix,
                    terms=int(terms),
                    k=int(k),
                    samples=int(samples),
                    p99_us=float(p99),
                    p999_us=float(p999),
                    p999_ok=p999_ok == "true",
                    checksum=checksum,
                )
            )
    return measurements


def read_metas(results_dir: Path) -> list[JobMeta]:
    metas = []
    for path in sorted(results_dir.glob("*.meta")):
        values = dict(
            line.split("=", 1) for line in path.read_text().splitlines() if "=" in line
        )
        metas.append(
            JobMeta(
                segments=int(values.get("segments", "0")),
                shard=values.get("shard", "?"),
                cpu=values.get("cpu", "?"),
                load_before=values.get("load_before", "?"),
                load_after=values.get("load_after", "?"),
            )
        )
    return metas


def group_cells(measurements: list[Measurement]) -> dict[int, list[Cell]]:
    cells: dict[tuple[int, str, int, int], Cell] = {}
    for m in measurements:
        key = (m.segments, *m.cell)
        cell = cells.setdefault(key, Cell(m.segments, m.mix, m.terms, m.k))
        (cell.base if m.binary == "base" else cell.head).append(m)
    by_segments: dict[int, list[Cell]] = defaultdict(list)
    for key in sorted(cells):
        by_segments[key[0]].append(cells[key])
    return by_segments


def fmt_us(value: float) -> str:
    return f"{value / 1000:.2f} ms" if value >= 1000 else f"{value:.0f} µs"


def fmt_reps(measurements: list[Measurement]) -> str:
    ordered = sorted(measurements, key=lambda m: m.repetition)
    return " / ".join(fmt_us(m.p99_us) for m in ordered)


def report(by_segments: dict[int, list[Cell]], metas: list[JobMeta], header: str) -> str:
    lines = [header, ""]
    cpus = sorted({meta.cpu for meta in metas})
    lines.append(f"Runner CPU models: {', '.join(cpus) or 'unknown'}. Each cell compares the two")
    lines.append("binaries on the same runner, in alternating order. Cells on different runners")
    lines.append("can have different absolute times.")
    lines.append("")
    lines.append(
        f"Verdict: **faster** / **slower** only when every repetition of one binary beats every "
        f"repetition of the other, each binary's repetitions agree within "
        f"{MAX_REPETITION_SPREAD:.0%}, each binary has at least {MIN_REPETITIONS} runs, and each "
        f"run has at least {MIN_SAMPLES_FOR_P99} samples."
    )
    for segments, cells in by_segments.items():
        verdicts = defaultdict(int)
        for cell in cells:
            verdicts[cell.verdict()] += 1
        clear = [c for c in cells if c.verdict() in ("faster", "slower", "no clear change")]
        lines += ["", f"### {segments} segment(s)", ""]
        summary = ", ".join(f"{name}: {count}" for name, count in sorted(verdicts.items()))
        lines.append(f"{len(cells)} cells. {summary}.")
        if clear:
            geomean = math.exp(statistics.fmean(math.log(1 + c.change()) for c in clear)) - 1
            lines.append(
                f"Geometric mean of the P99 change over the {len(clear)} cells that are not "
                f"noisy: {geomean:+.1%}."
            )
        lines += [
            "",
            "| Cell | Min samples | Baseline P99 | Head P99 | Change | Verdict |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for cell in cells:
            change = f"{cell.change():+.1%}" if cell.base and cell.head else "-"
            lines.append(
                f"| {cell.label} | {cell.min_samples} | {fmt_reps(cell.base)} | "
                f"{fmt_reps(cell.head)} | {change} | {cell.verdict()} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--header", default="## Block-Max WAND bench")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    measurements = read_measurements(args.results_dir)
    if not measurements:
        args.output.write_text(f"{args.header}\n\nNo results were found.\n")
        return 1
    by_segments = group_cells(measurements)
    args.output.write_text(report(by_segments, read_metas(args.results_dir), args.header))
    differ = [c for cells in by_segments.values() for c in cells if c.results_differ]
    for cell in differ:
        print(f"results differ: {cell.segments} segment(s), {cell.label}", file=sys.stderr)
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
