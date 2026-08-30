"""Preflight checks.

The value of ``doctor`` is that it turns an expensive mid-run discovery into a cheap
pre-run one, so these tests are mostly about it *finding* things — a gated dataset with no
token, a missing key, a log directory that would leak transcripts — rather than about it
being quiet when all is well.
"""

from __future__ import annotations

import pytest

from safety_eval.doctor import Level, diagnose


def find(diagnosis, name: str):
    return next(c for c in diagnosis.checks if c.name == name)


@pytest.fixture()
def clean_env(monkeypatch):
    for var in ("OPENROUTER_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                "S3_MODEL_BUCKET"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_missing_provider_key_is_a_failure_with_the_fix(config, clean_env) -> None:
    d = diagnose(config, check_network=False)
    check = find(d, "creds:openrouter")
    assert check.level is Level.FAIL
    assert "OPENROUTER_API_KEY" in check.detail
    assert ".env" in check.fix
    assert d.exit_code == 1


def test_a_present_key_is_masked_not_echoed(config, clean_env) -> None:
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-v1-abcdefghijklmnop")
    check = find(diagnose(config, check_network=False), "creds:openrouter")
    assert check.level is Level.OK
    assert "abcdefghijklmnop" not in check.detail


def test_gated_dataset_without_a_token_is_a_failure_naming_the_gate_url(
    config, clean_env
) -> None:
    """This is the check that pays for the whole module: it turns a mid-run crash into a
    one-line instruction before anything is spent."""
    check = find(diagnose(config, check_network=False), "dataset:xstest")
    assert check.level is Level.FAIL
    assert "GATED" in check.detail
    assert "huggingface.co/datasets/walledai/XSTest" in check.fix
    assert "blocked" in check.fix


def test_ungated_datasets_pass_without_credentials(config, clean_env) -> None:
    for key in ("dataset:sycophancy", "dataset:strong_reject"):
        assert find(diagnose(config, check_network=False), key).level is Level.OK


def test_gated_dataset_with_a_token_is_not_verified_offline(config, clean_env) -> None:
    clean_env.setenv("HF_TOKEN", "hf_" + "x" * 30)
    check = find(diagnose(config, check_network=False), "dataset:xstest")
    assert check.level is Level.SKIP
    assert "not verified" in check.detail


def test_log_safety_check_catches_an_ungitignored_withheld_task(
    config, clean_env, tmp_path, monkeypatch
) -> None:
    """A repo that publishes a working jailbreak has failed regardless of its scores."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitignore").write_text("__pycache__/\n")
    check = find(diagnose(config, check_network=False), "log safety")
    assert check.level is Level.FAIL
    assert "strong_reject" in check.detail
    assert "forbidden prompts" in check.fix


def test_log_safety_check_passes_when_gitignored(config, clean_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitignore").write_text("logs/strong_reject/\n")
    assert find(diagnose(config, check_network=False), "log safety").level is Level.OK


def test_harness_versions_are_reported(config, clean_env) -> None:
    check = find(diagnose(config, check_network=False), "harness")
    assert "inspect_ai" in check.detail and "inspect_evals" in check.detail


def test_version_drift_from_the_catalog_is_a_warning(config, catalog, clean_env) -> None:
    """An inspect_evals upgrade can change a scorer, which changes the numbers."""
    catalog.verified_against = dict(catalog.verified_against, inspect_evals="0.0.1")
    check = find(diagnose(config, catalog, check_network=False), "harness")
    assert check.level is Level.WARN
    assert "--metrics" in check.fix
    catalog.verified_against = dict(catalog.verified_against, inspect_evals="0.18.0")


def test_an_invalid_config_is_reported_as_a_check_not_a_traceback(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: []\ngrader_model: x\ntasks: []\n")
    d = diagnose(config_path=bad, check_network=False)
    assert find(d, "config").level is Level.FAIL
    assert d.exit_code == 1


def test_render_shows_the_fix_for_failures_only(config, clean_env) -> None:
    text = diagnose(config, check_network=False).render()
    assert "[FAIL]" in text and "->" in text
    assert "failure(s)" in text


def test_aws_is_skipped_when_not_configured(config, clean_env) -> None:
    assert find(diagnose(config, check_network=False), "aws:s3").level is Level.SKIP


@pytest.mark.harness
def test_metric_drift_probe_confirms_the_catalog(config, catalog, clean_env) -> None:
    """Runs the ungated benchmarks against mockllm and diffs the real metric addresses.

    Free, offline apart from the dataset fetch, and the only way to know the catalog still
    describes the harness that is installed.
    """
    d = diagnose(config, catalog, check_metrics=True, check_network=False)
    sr = find(d, "metrics:strong_reject")
    assert sr.level is Level.OK, sr.detail
    assert "confirmed" in sr.detail
    # XSTest cannot be probed without dataset access, and says so rather than passing.
    assert find(d, "metrics:xstest").level is Level.SKIP
