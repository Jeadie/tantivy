"""Summarizes `term_union_bench` A/B results from the `Term union bench` workflow.

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


@dataclass(frozen=True)
class Percentile:
    name: str
    # A percentile needs at least 10 samples above it.
    min_samples: int


PERCENTILES = [
    Percentile("P90", 100),
    Percentile("P95", 200),
    Percentile("P99", 1_000),
    Percentile("P99.9", 10_000),
]
# The verdict uses P99.
VERDICT_PERCENTILE = PERCENTILES[2]
MIN_SAMPLES_FOR_P99 = VERDICT_PERCENTILE.min_samples
# Repetitions of the same binary that differ by more than this are too noisy to compare.
MAX_REPETITION_SPREAD = 0.20
# A verdict needs at least this many runs of each binary. With 2 runs each and no real difference,
# all runs of one binary beat all runs of the other by chance for 1 cell in 3 (1 in 10 with 3
# runs), so the workflow uses 3 runs by default and MIN_CHANGE also applies.
MIN_REPETITIONS = 2
# A verdict also needs a median P99 change of at least this size.
MIN_CHANGE = 0.05


@dataclass(frozen=True)
class Measurement:
    segments: int
    binary: str
    repetition: int
    mix: str
    terms: int
    k: int
    samples: int
    # Latency in µs for each percentile name. Results from older versions of the bench only
    # have P99 and P99.9.
    latencies_us: dict[str, float]
    checksum: str

    @property
    def cell(self) -> tuple[str, int, int]:
        return (self.mix, self.terms, self.k)

    @property
    def p99_us(self) -> float:
        return self.latencies_us["P99"]

    def has(self, percentile: Percentile) -> bool:
        return percentile.name in self.latencies_us and self.samples >= percentile.min_samples


def percentile_change(
    reference: list[Measurement], other: list[Measurement], percentile: Percentile
) -> float | None:
    """Median `other` / median `reference` - 1, or None if a run lacks valid data."""
    runs = reference + other
    if not reference or not other or not all(m.has(percentile) for m in runs):
        return None
    ref = statistics.median(m.latencies_us[percentile.name] for m in reference)
    oth = statistics.median(m.latencies_us[percentile.name] for m in other)
    return oth / ref - 1.0


def fmt_change(change: float | None) -> str:
    return "-" if change is None else f"{change:+.1%}"


def verdict(reference: list[Measurement], other: list[Measurement]) -> str:
    """Verdict on the P99 of `other` against `reference`."""
    if not reference or not other:
        return "missing"
    if min(len(reference), len(other)) < MIN_REPETITIONS:
        return f"fewer than {MIN_REPETITIONS} runs"
    if min(m.samples for m in reference + other) < MIN_SAMPLES_FOR_P99:
        return f"too few samples (<{MIN_SAMPLES_FOR_P99})"
    if spread(reference) > MAX_REPETITION_SPREAD or spread(other) > MAX_REPETITION_SPREAD:
        return "noisy"
    ref = [m.p99_us for m in reference]
    oth = [m.p99_us for m in other]
    change = statistics.median(oth) / statistics.median(ref) - 1.0
    if max(oth) < min(ref) and change <= -MIN_CHANGE:
        return "faster"
    if min(oth) > max(ref) and change >= MIN_CHANGE:
        return "slower"
    return "no clear change"


CLEAR_VERDICTS = ("faster", "slower", "no clear change")


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

    def change(self, percentile: Percentile = VERDICT_PERCENTILE) -> float | None:
        return percentile_change(self.base, self.head, percentile)

    def verdict(self) -> str:
        if self.results_differ:
            return "**RESULTS DIFFER**"
        return verdict(self.base, self.head)


@dataclass(frozen=True)
class JobMeta:
    segments: int
    shard: str
    cpu: str
    load_before: str
    load_after: str
    # Settings and commits of the run. Empty in results from older versions of the workflow.
    num_docs: str = ""
    min_samples: str = ""
    queries: str = ""
    base_sha: str = ""
    head_sha: str = ""


def spread(measurements: list[Measurement]) -> float:
    values = [m.p99_us for m in measurements]
    return max(values) / min(values) - 1.0 if len(values) > 1 else 0.0


def parse_measurement(parts: list[str]) -> Measurement | None:
    if len(parts) == 12:
        seg, binary, rep, mix, terms, k, samples, p90, p95, p99, p999, checksum = parts
        latencies = {"P90": p90, "P95": p95, "P99": p99, "P99.9": p999}
    elif len(parts) == 11:
        # Older bench output: P99, P99.9 and a P99.9 validity flag.
        seg, binary, rep, mix, terms, k, samples, p99, p999, _p999_ok, checksum = parts
        latencies = {"P99": p99, "P99.9": p999}
    else:
        return None
    return Measurement(
        segments=int(seg),
        binary=binary,
        repetition=int(rep),
        mix=mix,
        terms=int(terms),
        k=int(k),
        samples=int(samples),
        latencies_us={name: float(value) for name, value in latencies.items()},
        checksum=checksum,
    )


def read_measurements(results_dir: Path) -> list[Measurement]:
    measurements = []
    for path in sorted(results_dir.glob("*.tsv")):
        for line in path.read_text().splitlines():
            measurement = parse_measurement(line.split("\t"))
            if measurement is not None:
                measurements.append(measurement)
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
                num_docs=values.get("num_docs", ""),
                min_samples=values.get("min_samples", ""),
                queries=values.get("queries", ""),
                base_sha=values.get("base", ""),
                head_sha=values.get("head", ""),
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


def geomean_changes(changes: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(1 + change) for change in changes)) - 1


def percentile_rules() -> str:
    names = ", ".join(f"{p.name} needs {p.min_samples:,}" for p in PERCENTILES)
    return (
        f"A percentile change is shown only when every run has enough samples ({names}). "
        "Changes are median head / median baseline - 1, over the runs of each binary."
    )


def report(by_segments: dict[int, list[Cell]], metas: list[JobMeta], header: str) -> str:
    lines = [header, ""]
    cpus = sorted({meta.cpu for meta in metas})
    lines.append(f"Runner CPU models: {', '.join(cpus) or 'unknown'}. Each cell compares the two")
    lines.append("binaries on the same runner, in alternating order. Cells on different runners")
    lines.append("can have different absolute times.")
    lines.append("")
    lines.append(
        f"Verdict (on P99): **faster** / **slower** only when every run of one binary beats every "
        f"run of the other, the median changes by at least {MIN_CHANGE:.0%}, each binary's runs "
        f"agree within {MAX_REPETITION_SPREAD:.0%}, each binary has at least {MIN_REPETITIONS} "
        f"runs, and each run has at least {MIN_SAMPLES_FOR_P99:,} samples. {percentile_rules()}"
    )
    for segments, cells in by_segments.items():
        verdicts: dict[str, int] = defaultdict(int)
        for cell in cells:
            verdicts[cell.verdict()] += 1
        clear = [c for c in cells if c.verdict() in CLEAR_VERDICTS]
        lines += ["", f"### {segments} segment(s)", ""]
        summary = ", ".join(f"{name}: {count}" for name, count in sorted(verdicts.items()))
        lines.append(f"{len(cells)} cells. {summary}.")
        for percentile in PERCENTILES:
            changes = [c.change(percentile) for c in clear]
            valid = [change for change in changes if change is not None]
            if valid:
                lines.append(
                    f"- {percentile.name}: geometric mean change {geomean_changes(valid):+.1%} "
                    f"over {len(valid)} cells that are not noisy."
                )
        names = " | ".join(p.name for p in PERCENTILES)
        lines += [
            "",
            f"| Cell | Min samples | Baseline P99 | Head P99 | {names} | Verdict |",
            "| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in PERCENTILES) + " | --- |",
        ]
        for cell in cells:
            changes = " | ".join(fmt_change(cell.change(p)) for p in PERCENTILES)
            lines.append(
                f"| {cell.label} | {cell.min_samples} | {fmt_reps(cell.base)} | "
                f"{fmt_reps(cell.head)} | {changes} | {cell.verdict()} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--header", default="## Term union bench")
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
