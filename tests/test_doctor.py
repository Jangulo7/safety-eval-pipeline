"""Preflight checks.

The value of ``doctor`` is that it turns an expensive mid-run discovery into a cheap
pre-run one, so these tests are mostly about it *finding* things — a gated dataset with no
token, a missing key, a log directory that would leak transcripts — rather than about it
being quiet when all is well.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from safety_eval.doctor import Level, diagnose

ROOT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "eval_config.yaml"


def find(diagnosis, name: str):
    return next(c for c in diagnosis.checks if c.name == name)


@pytest.fixture()
def clean_env(monkeypatch):
    for var in ("OPENROUTER_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                "S3_MODEL_BUCKET"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_missing_grader_key_is_a_failure_with_the_fix(config, clean_env) -> None:
    """The grader is hosted even when the models under test are local.

    Without its key every judge-graded metric comes back nan, so this is a failure rather
    than a warning.
    """
    d = diagnose(config, check_network=False)
    check = find(d, "creds:grader")
    assert check.level is Level.FAIL
    assert "OPENROUTER_API_KEY" in check.detail
    assert d.exit_code == 1


def test_a_present_key_is_masked_not_echoed(config, clean_env, tmp_path) -> None:
    """A preflight that echoes a secret into a terminal or a CI log is a liability."""
    import yaml

    from safety_eval.config import RunConfig

    data = yaml.safe_load(ROOT_CONFIG.read_text())
    data["models"] = [{"id": "openrouter/openai/gpt-4.1", "family": "openai",
                       "label": "GPT-4.1"}]
    path = tmp_path / "or.yaml"
    path.write_text(yaml.safe_dump(data))

    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-v1-abcdefghijklmnop")
    check = find(diagnose(RunConfig.load(path, config.catalog), check_network=False),
                 "creds:openrouter")
    assert check.level is Level.OK
    assert "abcdefghijklmnop" not in check.detail


def test_a_local_model_needs_a_server_not_a_key(config, clean_env) -> None:
    """Inspect would otherwise try to start its own vLLM, which needs a torch this
    environment cannot host without replacing its own."""
    clean_env.delenv("VLLM_BASE_URL", raising=False)
    check = find(diagnose(config, check_network=False), "vllm server")
    assert check.level is Level.FAIL
    assert "VLLM_BASE_URL" in check.detail
    assert "vllm serve" in check.fix


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

    # XSTest is gated. Whether it can be probed here depends on the machine: `inspect_ai`
    # loads `.env` itself when a model is constructed, so a developer with a granted token
    # sees it confirmed while CI sees it skipped. Both are correct; what must never happen
    # is a silent FAIL, which would mean the catalog was checked against nothing.
    xstest = find(d, "metrics:xstest")
    assert xstest.level in (Level.OK, Level.SKIP), xstest.detail
    if xstest.level is Level.SKIP:
        assert "HF_TOKEN" in xstest.detail
    else:
        assert "confirmed" in xstest.detail


def test_provider_extras_are_checked_not_just_the_key(config, clean_env) -> None:
    """A valid API key is not the same as a usable provider.

    inspect_ai ships its provider integrations as optional extras, so a missing package
    surfaces as a PrerequisiteError at *generation* time — after the credential check has
    passed, after the dataset has downloaded, and once per cell. This check moves it to the
    front, where it costs nothing.
    """
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "x" * 24)
    check = find(diagnose(config, check_network=False), "provider:openrouter")
    assert check.level is Level.OK
    assert "constructs" in check.detail


def test_a_missing_provider_extra_is_a_failure_carrying_the_install_line(
    config, clean_env, monkeypatch
) -> None:
    clean_env.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "x" * 24)

    def unavailable(*args, **kwargs):
        raise RuntimeError("ERROR: OpenRouter API requires optional dependencies. "
                           "Install with: pip install openai")

    monkeypatch.setattr("inspect_ai.model.get_model", unavailable)
    check = find(diagnose(config, check_network=False), "provider:openrouter")
    assert check.level is Level.FAIL
    assert "pip install openai" in check.fix
    assert "generation time" in check.fix


def test_unpinned_nondeterministic_sample_order_fails_a_multi_model_run(
    catalog, clean_env, tmp_path
) -> None:
    """A matrix over an unseeded shuffle is not a comparison.

    `inspect_evals/sycophancy` shuffles its dataset by default with no seed, so under a
    sample cap each cell draws a different subset — two loads of 50 were measured to share
    none. Every model would be scored on different prompts and the leaderboard would rank
    sampling noise. `--seed` does not cover it: it seeds generation, not dataset ordering.
    """
    import yaml

    from safety_eval.config import RunConfig

    data = yaml.safe_load(Path(ROOT_CONFIG).read_text())
    for task in data["tasks"]:
        if task["key"] == "sycophancy":
            task["args"].pop("shuffle", None)
    path = tmp_path / "unpinned.yaml"
    path.write_text(yaml.safe_dump(data))

    check = find(diagnose(RunConfig.load(path, catalog), catalog, check_network=False),
                 "sample order:sycophancy")
    assert check.level is Level.FAIL
    assert "same prompts" in check.detail
    assert "shuffle: False" in check.fix


def test_pinned_sample_order_passes(config, catalog, clean_env) -> None:
    check = find(diagnose(config, catalog, check_network=False), "sample order:sycophancy")
    assert check.level is Level.OK
    assert "same samples" in check.detail


def test_deterministic_benchmarks_are_not_checked(config, catalog, clean_env) -> None:
    """Only tasks that actually have the hazard get a row; the rest stay quiet."""
    names = {c.name for c in diagnose(config, catalog, check_network=False).checks}
    assert "sample order:sycophancy" in names
    assert "sample order:strong_reject" not in names
