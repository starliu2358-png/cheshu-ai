from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from cockpit_agent.vehicle_book.schemas import (
    AnswerResult,
    Citation,
    DocumentMetadata,
    KnowledgeChunk,
    RetrievedChunk,
)
from scripts import run_answer_eval

POSITIVE_CASE_ID = "x1_v1_battery_capacity_answer_001"
FALLBACK_CASE_ID = "x1_v1_flight_mode_answer_001"
SECOND_POSITIVE_CASE_ID = "x2_v1_battery_capacity_answer_001"

SUMMARY_METRICS = {
    "total_cases",
    "positive_cases",
    "fallback_cases",
    "critical_cases",
    "cited_cases",
    "answer_accuracy",
    "citation_accuracy",
    "fallback_accuracy",
    "grounded_answer_rate",
    "critical_error_rate",
    "cross_vehicle_answer_error_rate",
    "cross_version_answer_error_rate",
    "cross_vehicle_answer_error_cases",
    "cross_version_answer_error_cases",
    "failed_case_ids",
}


def _case_payload(case_id: str) -> dict[str, Any]:
    dataset = Path("data/eval/vehicle_book_answer_eval.jsonl")
    for line in dataset.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload["case_id"] == case_id:
            return payload
    raise AssertionError(f"test fixture case is missing: {case_id}")


def _write_dataset(tmp_path: Path, case_ids: tuple[str, ...]) -> Path:
    path = tmp_path / "cases.jsonl"
    rows = [json.dumps(_case_payload(case_id), ensure_ascii=False) for case_id in case_ids]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _positive_result(
    *,
    vehicle_id: str = "alpha_aster_x1_max_v1",
    chunk_id: str = "oem_alpha_aster_x1_v1_owner_manual:battery-and-charging",
    document_version: str = "1.0",
    answer: str = "该车配备 75 kWh LFP 动力电池。",
    fallback_reason: str | None = None,
) -> AnswerResult:
    metadata = DocumentMetadata(
        document_id=chunk_id.split(":", 1)[0],
        vehicle_id=vehicle_id,
        title="Fake owner manual",
        content_type="owner_manual",
        document_version=document_version,
        source_path=Path("fake/owner-manual.md"),
        status="published",
    )
    evidence = RetrievedChunk(
        chunk=KnowledgeChunk(
            chunk_id=chunk_id,
            section="Battery and charging",
            text=answer,
            metadata=metadata,
        ),
        score=0.875,
        rank=1,
    )
    return AnswerResult(
        answer=answer,
        vehicle_id=vehicle_id,
        citations=(Citation.from_retrieved(evidence),),
        evidence=(evidence,),
        fallback_reason=fallback_reason,
    )


def _fallback_result() -> AnswerResult:
    return AnswerResult(
        answer="未在当前车辆资料中找到足够可靠的依据。",
        vehicle_id="alpha_aster_x1_max_v1",
        fallback_reason="no_retrieval_results",
    )


def _write_raw_results(tmp_path: Path, rows: list[tuple[str, AnswerResult]]) -> Path:
    path = tmp_path / "fake_raw_results.json"
    payload = [
        {
            "case_id": case_id,
            "answer_result": run_answer_eval._raw_result_payload(result),
            "runtime_error": None,
        }
        for case_id, result in rows
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _block_online_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_call(*args: object, **kwargs: object) -> None:
        raise AssertionError("replay attempted to use a live or network path")

    monkeypatch.setattr(run_answer_eval, "_run_live", unexpected_call)
    monkeypatch.setattr(
        run_answer_eval.DeepSeekAnswerGenerator,
        "from_env",
        unexpected_call,
    )
    monkeypatch.setattr(run_answer_eval, "VehicleBookService", unexpected_call)


def _run_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    case_ids: tuple[str, ...] = (POSITIVE_CASE_ID, FALLBACK_CASE_ID),
    extra_args: tuple[str, ...] = (),
) -> tuple[int, Path]:
    dataset = _write_dataset(tmp_path, case_ids)
    results = {
        POSITIVE_CASE_ID: _positive_result(),
        FALLBACK_CASE_ID: _fallback_result(),
        SECOND_POSITIVE_CASE_ID: _positive_result(
            vehicle_id="alpha_aster_x2_max_v1",
            chunk_id="oem_alpha_aster_x2_v1_owner_manual:battery-and-charging",
            answer="该车配备 90 kWh NMC 动力电池。",
        ),
    }
    raw_results = _write_raw_results(
        tmp_path,
        [(case_id, results[case_id]) for case_id in case_ids],
    )
    output_dir = tmp_path / "reports"
    _block_online_paths(monkeypatch)
    exit_code = run_answer_eval.main(
        [
            "--mode",
            "replay",
            "--dataset",
            str(dataset),
            "--raw-results",
            str(raw_results),
            "--output-dir",
            str(output_dir),
            *extra_args,
        ]
    )
    return exit_code, output_dir


def test_replay_cli_is_offline_and_writes_complete_stable_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, output_dir = _run_replay(tmp_path, monkeypatch)

    assert exit_code == 0
    summary = json.loads(
        (output_dir / run_answer_eval.SUMMARY_JSON).read_text(encoding="utf-8")
    )
    assert summary["mode"] == "replay"
    assert summary["total_cases"] == 2
    assert SUMMARY_METRICS <= summary.keys()
    assert set(summary["category_metrics"]) == {"charging", "no_answer"}
    for category_summary in summary["category_metrics"].values():
        assert SUMMARY_METRICS <= category_summary.keys()

    with (output_dir / run_answer_eval.CASE_RESULTS_CSV).open(
        encoding="utf-8", newline=""
    ) as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == run_answer_eval.CASE_FIELDS
    assert [row["case_id"] for row in rows] == [POSITIVE_CASE_ID, FALLBACK_CASE_ID]


def test_answer_result_raw_round_trip_preserves_nested_values(tmp_path: Path) -> None:
    expected = _positive_result(fallback_reason="insufficient_evidence")
    raw_results = _write_raw_results(tmp_path, [(POSITIVE_CASE_ID, expected)])

    actual, runtime_errors = run_answer_eval._load_raw_results(
        raw_results, {POSITIVE_CASE_ID: expected.vehicle_id}
    )

    assert runtime_errors == {}
    assert actual[POSITIVE_CASE_ID] == expected
    assert actual[POSITIVE_CASE_ID].citations == expected.citations
    assert actual[POSITIVE_CASE_ID].evidence == expected.evidence
    assert actual[POSITIVE_CASE_ID].fallback_reason == "insufficient_evidence"


def test_case_id_selects_only_the_requested_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, output_dir = _run_replay(
        tmp_path,
        monkeypatch,
        extra_args=("--case-id", FALLBACK_CASE_ID),
    )

    assert exit_code == 0
    rows = json.loads(
        (output_dir / run_answer_eval.CASE_RESULTS_JSON).read_text(encoding="utf-8")
    )
    assert [row["case_id"] for row in rows] == [FALLBACK_CASE_ID]


def test_max_cases_limits_cases_in_dataset_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, output_dir = _run_replay(
        tmp_path,
        monkeypatch,
        case_ids=(POSITIVE_CASE_ID, FALLBACK_CASE_ID, SECOND_POSITIVE_CASE_ID),
        extra_args=("--max-cases", "2"),
    )

    assert exit_code == 0
    rows = json.loads(
        (output_dir / run_answer_eval.CASE_RESULTS_JSON).read_text(encoding="utf-8")
    )
    assert [row["case_id"] for row in rows] == [POSITIVE_CASE_ID, FALLBACK_CASE_ID]


@pytest.mark.parametrize(
    ("raw_rows", "message"),
    [
        ([], "raw results missing selected case_id"),
        (
            [
                (POSITIVE_CASE_ID, _positive_result()),
                (POSITIVE_CASE_ID, _positive_result()),
            ],
            "duplicate raw result case_id",
        ),
    ],
)
def test_missing_or_duplicate_raw_result_returns_a_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    raw_rows: list[tuple[str, AnswerResult]],
    message: str,
) -> None:
    dataset = _write_dataset(tmp_path, (POSITIVE_CASE_ID,))
    raw_results = _write_raw_results(tmp_path, raw_rows)
    _block_online_paths(monkeypatch)

    exit_code = run_answer_eval.main(
        [
            "--mode",
            "replay",
            "--dataset",
            str(dataset),
            "--raw-results",
            str(raw_results),
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("top_k", ["0", "-1", "not-an-integer"])
def test_invalid_top_k_returns_a_nonzero_exit_code(top_k: str) -> None:
    with pytest.raises(SystemExit) as raised:
        run_answer_eval.main(["--top-k", top_k])

    assert raised.value.code != 0
