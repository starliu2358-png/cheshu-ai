"""Run the local Vehicle Book release gate without network access by default."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from cockpit_agent.vehicle_book.release_gate import (
    BASELINE_REPORTS,
    create_baseline_manifest,
    evaluate_release_gate,
    measure_retrieval_p95_ms,
    render_markdown_report,
)

DEFAULT_CANDIDATE_DIR = Path("outputs/eval")
DEFAULT_BASELINE_DIR = Path("data/baselines/vehicle_book_release_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/release_gate")
BASELINE_ROOT = Path("data/baselines").resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deterministic Vehicle Book release gate.")
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--answer-mode", choices=("replay", "live", "skip"), default="replay")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--initialize-baseline", action="store_true")
    parser.add_argument("--force", action="store_true", help="allow replacing an existing baseline")
    parser.add_argument("--json-only", action="store_true")
    return parser


def _run_command(arguments: list[str]) -> int:
    return subprocess.run(arguments, cwd=Path.cwd(), check=False).returncode


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _initialize_baseline(candidate_dir: Path, baseline_dir: Path, pytest_passed: bool) -> None:
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for name in BASELINE_REPORTS:
        source = candidate_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"candidate report missing for baseline: {source}")
        shutil.copyfile(source, baseline_dir / name)
    _write_json(baseline_dir / "baseline_manifest.json", create_baseline_manifest(candidate_dir, pytest_passed=pytest_passed))


def _safe_baseline_target(path: Path) -> bool:
    target = path.resolve()
    try:
        target.relative_to(BASELINE_ROOT)
    except ValueError:
        return False
    return target != BASELINE_ROOT


def _print_summary(summary: dict[str, object]) -> None:
    print("Gate                                  Status")
    for gate in summary["gates"]:  # type: ignore[index]
        print(f"{gate['name']:<37} {gate['status']}")
    print(f"Decision: {summary['decision']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.initialize_baseline and args.baseline_dir.exists() and not args.force:
        print(
            f"error: baseline already exists: {args.baseline_dir}; pass --force to replace it.",
            file=sys.stderr,
        )
        return 1
    if args.initialize_baseline and args.force and not _safe_baseline_target(args.baseline_dir):
        print("error: --force only permits a baseline directory below data/baselines.", file=sys.stderr)
        return 1
    pytest_passed: bool | None = None
    if not args.skip_pytest:
        pytest_passed = _run_command([sys.executable, "-m", "pytest", "-q"]) == 0

    retrieval_exit = _run_command([sys.executable, "scripts/run_retrieval_eval.py", "--output-dir", str(args.candidate_dir)])
    answer_exit = 0
    live_status = "NOT_RUN"
    live_reason = "Live API acceptance was not explicitly executed."
    if args.answer_mode != "skip":
        answer_exit = _run_command(
            [sys.executable, "scripts/run_answer_eval.py", "--mode", args.answer_mode, "--output-dir", str(args.candidate_dir), "--raw-results", str(args.candidate_dir / "answer_raw_results.json")]
        )
    if args.answer_mode == "live":
        live_status = "PASS" if answer_exit == 0 else "NOT_RUN"
        live_reason = "Live answer evaluation completed." if answer_exit == 0 else "Live answer evaluation did not complete; no PASS is inferred."

    if args.initialize_baseline:
        if args.baseline_dir.exists() and args.force:
            shutil.rmtree(args.baseline_dir)
        try:
            _initialize_baseline(args.candidate_dir, args.baseline_dir, pytest_passed is True)
        except (OSError, ValueError) as exc:
            print(f"error: cannot initialize baseline: {exc}", file=sys.stderr)
            return 1

    summary = evaluate_release_gate(
        args.candidate_dir,
        args.baseline_dir,
        pytest_passed=pytest_passed,
        latency_p95_ms=measure_retrieval_p95_ms(),
        live_status=live_status,
        live_reason=live_reason,
    )
    if retrieval_exit != 0 or answer_exit != 0:
        summary["decision"] = "FAIL"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "release_gate_summary.json", summary)
    (args.output_dir / "release_gate_report.md").write_text(render_markdown_report(summary), encoding="utf-8")
    if not args.json_only:
        _print_summary(summary)
    return 0 if summary["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
