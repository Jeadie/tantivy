"""Compares the results of two runs of the `Block-Max WAND bench` workflow.

Example: compare the head of run 222 with the head of run 111 (for example, `main` at two dates,
or two branches that each ran against `main`):

    python3 .github/bench/compare_block_wand_runs.py 111 222 --repo Jeadie/tantivy

The script downloads the `summary` artifact of each run with the GitHub CLI (`gh`), and compares
one side (`head` by default) of each run, cell by cell. Use `--dir-a` / `--dir-b` instead of run
IDs to compare results that are already on disk.

Two runs can use different runner CPU models, so a cell is compared only when both runs measured
it on the same CPU model. A comparison between runs is always less reliable than the A/B
comparison inside one run, where both binaries run on the same runner.

Exit codes: 0 on success, 2 if the runs used different corpus settings.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Do not write __pycache__ next to the scripts when importing the summary module.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_block_wand_bench import (  # noqa: E402
    PERCENTILES,
    JobMeta,
    Percentile,
    Measurement,
    fmt_change,
    fmt_us,
    percentile_change,
    percentile_rules,
    read_measurements,
    read_metas,
    verdict,
)


@dataclass
class RunResults:
    label: str
    measurements: list[Measurement]
    metas: list[JobMeta]

    def meta_for(self, segments: int, mix: str, terms: int) -> JobMeta | None:
        shard = f"{mix}/{terms}/"
        for meta in self.metas:
            if meta.segments == segments and meta.shard == shard:
                return meta
        return None

    def settings(self) -> set[tuple[str, str]]:
        return {(meta.num_docs, meta.min_samples) for meta in self.metas}

    def shas(self, side: str) -> set[str]:
        return {meta.head_sha if side == "head" else meta.base_sha for meta in self.metas} - {""}


@dataclass
class CellComparison:
    segments: int
    mix: str
    terms: int
    k: int
    a: list[Measurement]
    b: list[Measurement]
    cpu_a: str
    cpu_b: str

    @property
    def label(self) -> str:
        return f"{self.mix}/{self.terms}/{self.k}"

    @property
    def same_cpu(self) -> bool:
        return self.cpu_a == self.cpu_b

    def change(self, percentile: Percentile) -> float | None:
        return percentile_change(self.a, self.b, percentile) if self.same_cpu else None

    def verdict(self) -> str:
        if not self.a or not self.b:
            return "missing in one run"
        if not self.same_cpu:
            return "different CPU, not compared"
        return verdict(self.a, self.b)

    def results_differ(self) -> bool:
        return len({m.checksum for m in self.a + self.b}) > 1


def download_results(run_id: str, repo: str, destination: Path) -> Path:
    subprocess.run(
        ["gh", "run", "download", run_id, "--repo", repo, "--name", "summary", "--dir", str(destination)],
        check=True,
    )
    results = destination / "results"
    return results if results.is_dir() else destination


def load_run(label: str, results_dir: Path) -> RunResults:
    return RunResults(label, read_measurements(results_dir), read_metas(results_dir))


def compare(run_a: RunResults, run_b: RunResults, side: str) -> list[CellComparison]:
    def by_cell(run: RunResults) -> dict[tuple[int, str, int, int], list[Measurement]]:
        cells: dict[tuple[int, str, int, int], list[Measurement]] = {}
        for m in run.measurements:
            if m.binary == side:
                cells.setdefault((m.segments, m.mix, m.terms, m.k), []).append(m)
        return cells

    cells_a, cells_b = by_cell(run_a), by_cell(run_b)
    comparisons = []
    for key in sorted(cells_a.keys() | cells_b.keys()):
        segments, mix, terms, k = key
        meta_a = run_a.meta_for(segments, mix, terms)
        meta_b = run_b.meta_for(segments, mix, terms)
        comparisons.append(
            CellComparison(
                segments=segments,
                mix=mix,
                terms=terms,
                k=k,
                a=cells_a.get(key, []),
                b=cells_b.get(key, []),
                cpu_a=meta_a.cpu if meta_a else "?",
                cpu_b=meta_b.cpu if meta_b else "?",
            )
        )
    return comparisons


def report(run_a: RunResults, run_b: RunResults, side: str, comparisons: list[CellComparison]) -> str:
    def shas(run: RunResults) -> str:
        return ", ".join(sha[:8] for sha in sorted(run.shas(side))) or "unknown"

    lines = [
        "## Block-Max WAND bench: comparison of two runs",
        "",
        f"Side: `{side}`. Run A: {run_a.label} ({shas(run_a)}). Run B: {run_b.label} ({shas(run_b)}).",
        "Change = median of run B / median of run A - 1, for each percentile. Cells measured on",
        "different CPU models are not compared. The verdict (on P99) uses the same rules as the",
        f"summary of one run. {percentile_rules()}",
    ]
    for segments in sorted({c.segments for c in comparisons}):
        cells = [c for c in comparisons if c.segments == segments]
        counts: dict[str, int] = {}
        for cell in cells:
            counts[cell.verdict()] = counts.get(cell.verdict(), 0) + 1
        lines += ["", f"### {segments} segment(s)", ""]
        lines.append(", ".join(f"{name}: {count}" for name, count in sorted(counts.items())) + ".")
        names = " | ".join(p.name for p in PERCENTILES)
        lines += [
            "",
            f"| Cell | Run A P99 | Run B P99 | {names} | Verdict | Same results |",
            "| --- | ---: | ---: | " + " | ".join("---:" for _ in PERCENTILES) + " | --- | --- |",
        ]
        for cell in cells:
            a = " / ".join(fmt_us(m.p99_us) for m in sorted(cell.a, key=lambda m: m.repetition))
            b = " / ".join(fmt_us(m.p99_us) for m in sorted(cell.b, key=lambda m: m.repetition))
            changes = " | ".join(fmt_change(cell.change(p)) for p in PERCENTILES)
            same = "-" if not (cell.a and cell.b) else ("no" if cell.results_differ() else "yes")
            lines.append(
                f"| {cell.label} | {a or '-'} | {b or '-'} | {changes} | {cell.verdict()} | {same} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", nargs="?", help="Run ID of the reference run")
    parser.add_argument("run_b", nargs="?", help="Run ID of the run to compare with the reference")
    parser.add_argument("--repo", default="Jeadie/tantivy", help="Repository of the runs (owner/name)")
    parser.add_argument("--side", choices=["head", "base"], default="head", help="Binary to compare")
    parser.add_argument("--dir-a", type=Path, help="Results directory to use instead of run A")
    parser.add_argument("--dir-b", type=Path, help="Results directory to use instead of run B")
    parser.add_argument("--output", type=Path, help="Write the Markdown report here (default: stdout)")
    parser.add_argument(
        "--allow-different-settings",
        action="store_true",
        help="Compare runs that used a different number of documents or minimum samples",
    )
    args = parser.parse_args()
    if not (args.run_a or args.dir_a) or not (args.run_b or args.dir_b):
        parser.error("give two run IDs, or --dir-a / --dir-b")

    with tempfile.TemporaryDirectory() as tmp:
        dir_a = args.dir_a or download_results(args.run_a, args.repo, Path(tmp) / "a")
        dir_b = args.dir_b or download_results(args.run_b, args.repo, Path(tmp) / "b")
        run_a = load_run(f"run {args.run_a}" if args.run_a else str(dir_a), dir_a)
        run_b = load_run(f"run {args.run_b}" if args.run_b else str(dir_b), dir_b)

    for run in (run_a, run_b):
        if not run.measurements:
            print(f"{run.label}: no results were found", file=sys.stderr)
            return 1
        if run.settings() == {("", "")}:
            print(f"{run.label}: the settings were not recorded (older workflow)", file=sys.stderr)
    settings_a, settings_b = run_a.settings() - {("", "")}, run_b.settings() - {("", "")}
    if settings_a and settings_b and settings_a != settings_b and not args.allow_different_settings:
        print(
            f"The runs used different settings (documents, minimum samples): {sorted(settings_a)} "
            f"and {sorted(settings_b)}. Use --allow-different-settings to compare them.",
            file=sys.stderr,
        )
        return 2

    text = report(run_a, run_b, args.side, compare(run_a, run_b, args.side))
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
