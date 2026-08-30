"""The command line, exercised end to end with a stubbed harness.

`run --dry-run` costing nothing is asserted rather than assumed: it is the command the
README tells people to start with, and if it ever opened a connection that advice would be
wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from safety_eval.cli import main


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    """Run the CLI with results and logs under a temp directory."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitignore").write_text("logs/strong_reject/\n")
    return tmp_path


def test_list_benchmarks_prints_ranges_and_directions(capsys) -> None:
    assert main(["list-benchmarks"]) == 0
    out = capsys.readouterr().out
    assert "0-100 %" in out                       # XSTest is a percentage
    assert "0-5" in out                           # StrongREJECT is not
    assert "judge_llm" in out and "scorer_model" in out
    assert "GATED DATASET" in out
    assert "transcripts withheld" in out


def test_dry_run_spends_nothing(capsys, monkeypatch, workdir) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("--dry-run must not reach the harness")

    monkeypatch.setattr("inspect_ai.eval", explode, raising=False)
    assert main(["run", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "samples requested" in out
    assert "Nothing was run and nothing was spent" in out


def test_dry_run_names_the_cells_whose_logs_are_withheld(capsys, workdir) -> None:
    main(["run", "--dry-run"])
    out = capsys.readouterr().out
    assert out.count("WITHHELD") == 3            # one per model on strong_reject


def test_doctor_exits_nonzero_when_something_is_missing(capsys, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert main(["doctor", "--offline"]) == 1
    assert "preflight" in capsys.readouterr().out


def test_report_and_gate_from_a_saved_run(workdir, catalog, capsys) -> None:
    from fixtures.factory import make_results

    results = make_results(catalog=catalog, profile="clean")
    run_dir = workdir / "results" / "run-cli"
    results.save(run_dir / "results.json")

    root = Path(__file__).resolve().parents[1]
    args = ["--config", str(root / "config" / "eval_config.yaml"),
            "--catalog", str(root / "config" / "benchmarks.yaml")]

    assert main(args + ["report", "--run-id", "run-cli"]) == 0
    assert (run_dir / "leaderboard.html").exists()
    assert (run_dir / "report.pdf").exists()
    assert (run_dir / "charts" / "calibration.png").exists()

    assert main(args + ["gate", "--run-id", "run-cli"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_gate_exits_one_on_a_breach(workdir, catalog, capsys) -> None:
    """The command that makes 'release gating' a fact rather than a claim."""
    from fixtures.factory import make_results

    results = make_results(catalog=catalog)          # the realistic profile breaches
    results.save(workdir / "results" / "run-bad" / "results.json")
    root = Path(__file__).resolve().parents[1]
    code = main(["--config", str(root / "config" / "eval_config.yaml"),
                 "--catalog", str(root / "config" / "benchmarks.yaml"),
                 "gate", "--run-id", "run-bad"])
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_report_with_no_runs_is_a_clear_message_not_a_traceback(workdir, capsys) -> None:
    assert main(["report"]) == 1
    assert "safety-eval run" in capsys.readouterr().err


def test_task_subset_scopes_the_run(capsys, workdir) -> None:
    main(["run", "--dry-run", "--tasks", "sycophancy", "--limit", "5"])
    out = capsys.readouterr().out
    assert "cells             3" in out
    assert "samples requested 15" in out
    assert "xstest" not in out.split("gated datasets")[0]
