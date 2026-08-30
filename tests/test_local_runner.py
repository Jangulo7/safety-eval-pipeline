"""Sequential local-model orchestration.

The GPU holds one 8B model at a time, so the matrix is run model-by-model against a vLLM
server this pipeline starts and stops. Two properties matter more than the rest and are
tested hardest:

* **Weights this run did not download are never deleted.** The HuggingFace cache is shared
  with everything else on the machine; deleting someone's model because it happened to be
  in the matrix would be unforgivable, and it is exactly the kind of thing a "clean up
  after yourself" flag does by accident.
* **One model's failure does not lose the others.** A model that fails to download, or a
  server that never becomes ready, costs its own cells and nothing more.

Every test here stubs the subprocess and the harness, so the suite never starts a server,
never downloads weights and never touches a real cache.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import ClassVar

import pytest

from conftest_safety import make_log, make_score
from safety_eval.local_runner import (
    LocalMatrixRunner,
    ServerError,
    hf_cache_dir,
    is_cached,
    plan_models,
    serve_command,
)

# Bound at import time. The autouse fixture below replaces the module attribute, so a test
# that needs the real readiness loop must hold a reference taken before that happens.
from safety_eval.local_runner import wait_until_ready as real_wait_until_ready


class FakeProc:
    """A server process that is alive until stopped.

    ``launched`` accumulates the command lines across a test, so the tests can assert which
    models had a server started for them. It is reset by the autouse fixture.
    """

    launched: ClassVar[list[list[str]]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.pid = 4242
        self._alive = True
        self.args = args[0] if args else []
        FakeProc.launched.append(self.args)

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return 0


@pytest.fixture(autouse=True)
def _no_real_processes(monkeypatch):
    FakeProc.launched = []
    monkeypatch.setattr("safety_eval.local_runner.wait_until_ready",
                        lambda *a, **k: ["served/model"])
    monkeypatch.setattr("os.killpg", lambda *a, **k: None)
    monkeypatch.setattr("os.getpgid", lambda pid: pid)
    return monkeypatch


def stub_eval(**kwargs):
    benchmark = kwargs["tasks"].split("/")[-1]
    scores = {
        "strong_reject": [make_score("strong_reject_scorer",
                                     {"jailbreak_rate": 0.1, "strong_reject_metric": 0.4})],
        "xstest": [make_score("model_graded_qa", {"refusal_rate": 12.0})],
        "sycophancy": [make_score("sycophancy_scorer",
                                  {"confidence": 0.7, "apologize_rate": 0.3})],
    }[benchmark]
    return [make_log(scores, task_version=3 if benchmark == "strong_reject" else 4,
                     full_task_version="3-A" if benchmark == "strong_reject" else "4-A")]


def make_cache(tmp_path: Path, repo: str) -> Path:
    """A cache entry that looks like a completed download."""
    path = hf_cache_dir(repo, tmp_path)
    (path / "snapshots" / "abc").mkdir(parents=True)
    (path / "snapshots" / "abc" / "model.safetensors").write_bytes(b"weights")
    return path


# --------------------------------------------------------------------------- cache state

def test_a_directory_without_weights_is_not_cached(tmp_path) -> None:
    """A failed download leaves the directory behind; presence alone must not fool us."""
    hf_cache_dir("org/model", tmp_path).mkdir(parents=True)
    assert not is_cached("org/model", tmp_path)


def test_weights_make_it_cached(tmp_path) -> None:
    make_cache(tmp_path, "org/model")
    assert is_cached("org/model", tmp_path)


def test_plan_records_what_was_already_here(config, tmp_path) -> None:
    make_cache(tmp_path, config.models[0].id.split("/", 1)[1])
    plans = plan_models(config, tmp_path)
    assert plans[0].pre_existing and not plans[0].may_purge
    assert not plans[1].pre_existing and plans[1].may_purge


# ------------------------------------------------------- the non-negotiable purge rule

def test_a_pre_existing_model_is_never_deleted_even_with_purge(config, tmp_path) -> None:
    """The one that would be unforgivable: deleting a model the machine already had."""
    repo = config.models[0].id.split("/", 1)[1]
    cache = make_cache(tmp_path, repo)
    for other in config.models[1:]:
        make_cache(tmp_path, other.id.split("/", 1)[1])

    runner = LocalMatrixRunner(config, run_id="run-purge", port=8000, purge=True,
                              hf_home=tmp_path, popen=FakeProc, eval_fn=stub_eval)
    summary = runner.run(log=lambda _: None)

    assert cache.exists(), "a model that was already cached must survive the run"
    assert repo in summary.kept
    assert repo not in summary.purged


def test_purge_is_off_by_default(config, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("safety_eval.local_runner.LocalMatrixRunner.fetch",
                        lambda self, plan, log: make_cache(tmp_path, plan.repo))
    runner = LocalMatrixRunner(config, run_id="run-keep", hf_home=tmp_path,
                               popen=FakeProc, eval_fn=stub_eval)
    summary = runner.run(log=lambda _: None)
    assert not summary.purged
    assert len(summary.kept) == len(config.models)


def test_purge_removes_only_what_this_run_downloaded(config, tmp_path, monkeypatch) -> None:
    kept_repo = config.models[0].id.split("/", 1)[1]
    make_cache(tmp_path, kept_repo)
    monkeypatch.setattr("safety_eval.local_runner.LocalMatrixRunner.fetch",
                        lambda self, plan, log: make_cache(tmp_path, plan.repo))

    summary = LocalMatrixRunner(config, run_id="run-p", purge=True, hf_home=tmp_path,
                                popen=FakeProc, eval_fn=stub_eval).run(log=lambda _: None)
    assert summary.kept == [kept_repo]
    assert set(summary.purged) == {m.id.split("/", 1)[1] for m in config.models[1:]}
    assert hf_cache_dir(kept_repo, tmp_path).exists()


# --------------------------------------------------------------------- serving sequence

def test_models_are_served_one_at_a_time(config, tmp_path, monkeypatch) -> None:
    """Three 8B models at bf16 will not co-reside in 32GB of VRAM."""
    monkeypatch.setattr("safety_eval.local_runner.LocalMatrixRunner.fetch",
                        lambda self, plan, log: make_cache(tmp_path, plan.repo))
    order: list[str] = []
    real_stop = LocalMatrixRunner.stop_server

    def tracking_stop(self, proc, log):
        if proc is not None:
            order.append("stop")
        return real_stop(self, proc, log)

    monkeypatch.setattr(LocalMatrixRunner, "stop_server", tracking_stop)
    original_start = LocalMatrixRunner.start_server

    def tracking_start(self, plan, log):
        order.append("start")
        return original_start(self, plan, log)

    monkeypatch.setattr(LocalMatrixRunner, "start_server", tracking_start)

    LocalMatrixRunner(config, run_id="run-seq", hf_home=tmp_path, popen=FakeProc,
                      eval_fn=stub_eval).run(log=lambda _: None)
    assert order == ["start", "stop"] * len(config.models), (
        "a server must be stopped before the next one starts"
    )


def test_every_model_gets_every_benchmark_in_one_run(config, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("safety_eval.local_runner.LocalMatrixRunner.fetch",
                        lambda self, plan, log: make_cache(tmp_path, plan.repo))
    summary = LocalMatrixRunner(config, run_id="run-one", hf_home=tmp_path, popen=FakeProc,
                                eval_fn=stub_eval).run(log=lambda _: None)
    assert len(summary.results) == len(config.models) * len(config.tasks)
    assert set(summary.results.models) == {m.id for m in config.models}
    assert {c.run_id for c in summary.results} == {"run-one"}, "one run id, one leaderboard"


def test_results_are_saved_after_every_model(config, tmp_path, monkeypatch) -> None:
    """A failure late in the matrix must not throw away the models that completed."""
    monkeypatch.setattr("safety_eval.local_runner.LocalMatrixRunner.fetch",
                        lambda self, plan, log: make_cache(tmp_path, plan.repo))
    seen: list[int] = []
    from safety_eval.results import ResultSet

    original = ResultSet.save

    def counting_save(self, path):
        seen.append(len(self))
        return original(self, path)

    monkeypatch.setattr(ResultSet, "save", counting_save)
    LocalMatrixRunner(config, run_id="run-save", hf_home=tmp_path, popen=FakeProc,
                      eval_fn=stub_eval).run(log=lambda _: None)
    assert len(seen) >= len(config.models)
    assert seen == sorted(seen), "each save must contain at least what the previous did"


# ------------------------------------------------------------------- failure isolation

def test_a_server_that_never_starts_loses_only_its_own_model(config, tmp_path,
                                                             monkeypatch) -> None:
    monkeypatch.setattr("safety_eval.local_runner.LocalMatrixRunner.fetch",
                        lambda self, plan, log: make_cache(tmp_path, plan.repo))
    bad = config.models[1].id.split("/", 1)[1]

    def flaky(base_url, api_key, **kwargs):
        if FakeProc.launched and bad in " ".join(FakeProc.launched[-1]):
            raise ServerError("not ready")
        return ["ok"]

    monkeypatch.setattr("safety_eval.local_runner.wait_until_ready", flaky)
    summary = LocalMatrixRunner(config, run_id="run-fail", hf_home=tmp_path, popen=FakeProc,
                                eval_fn=stub_eval).run(log=lambda _: None)
    assert bad in summary.failures
    assert len(summary.results) == (len(config.models) - 1) * len(config.tasks)


def test_a_failed_download_skips_that_model_without_serving_it(config, tmp_path,
                                                               monkeypatch) -> None:
    bad = config.models[0].id.split("/", 1)[1]

    def fetch(self, plan, log):
        if plan.repo == bad:
            raise RuntimeError("403 gated")
        make_cache(tmp_path, plan.repo)

    monkeypatch.setattr("safety_eval.local_runner.LocalMatrixRunner.fetch", fetch)
    summary = LocalMatrixRunner(config, run_id="run-dl", hf_home=tmp_path, popen=FakeProc,
                                eval_fn=stub_eval).run(log=lambda _: None)
    assert bad in summary.failures
    assert not any(bad in " ".join(cmd) for cmd in FakeProc.launched), (
        "a model whose weights are missing must not have a server started for it"
    )


# ------------------------------------------------------------------------ serve command

def test_every_model_is_served_with_identical_flags(config) -> None:
    """Serving flags are the conditions; they belong in one place, not typed per model."""
    flags = [
        [a for a in serve_command(config, m.id.split("/", 1)[1], 8000) if a != m.id.split("/", 1)[1]]
        for m in config.models
    ]
    assert all(f == flags[0] for f in flags)


def test_serve_command_carries_dtype_seed_and_context(config) -> None:
    cmd = serve_command(config, "org/model", 8123)
    assert "--dtype" in cmd and "bfloat16" in cmd
    assert cmd[cmd.index("--seed") + 1] == str(config.defaults.seed)
    assert cmd[cmd.index("--port") + 1] == "8123"
    assert "--max-model-len" in cmd


def test_no_quantization_flag_at_native_precision(config) -> None:
    assert "--quantization" not in serve_command(config, "org/model", 8000)


def test_quantization_flag_is_added_when_configured(config, catalog, tmp_path) -> None:
    """The fp8-vs-bf16 comparison must change exactly one variable.

    Applying `--quantization fp8` to the same bf16 checkpoint means the weights, the items,
    the sampling parameters and the grader are all identical, and precision is the only
    thing that moved.
    """
    import yaml

    from safety_eval.config import RunConfig

    data = yaml.safe_load(Path(config.source).read_text())
    data["serving"]["quantization"] = "fp8"
    path = tmp_path / "fp8.yaml"
    path.write_text(yaml.safe_dump(data))

    cmd = serve_command(RunConfig.load(path, catalog), "org/model", 8000)
    assert cmd[cmd.index("--quantization") + 1] == "fp8"
    assert "bfloat16" in cmd, "the checkpoint stays bf16; only the serving precision changes"


# ------------------------------------------------------ failing fast on a dead server

def test_a_dead_server_fails_immediately_rather_than_timing_out(tmp_path) -> None:
    """The defect that cost 45 minutes: waiting 900s for a process that died in 5.

    A server that dies on startup — an unrecognised flag, a CUDA failure, an OOM — never
    binds the port, and polling alone cannot tell that apart from a slow model load.
    """
    class DeadProc:
        returncode = 2

        def poll(self):
            return 2

    log = tmp_path / "server.log"
    log.write_text("api_server.py: error: unrecognized arguments: --nope\n")

    start = time.monotonic()
    with pytest.raises(ServerError) as excinfo:
        real_wait_until_ready("http://127.0.0.1:1/v1", "k", timeout_s=60, poll_s=0.1,
                              proc=DeadProc(), log_path=log)
    assert time.monotonic() - start < 5, "must not wait out the timeout for a dead process"
    assert "exited with code 2" in str(excinfo.value)
    assert "unrecognized arguments" in str(excinfo.value), (
        "the failure must carry the server's own reason, not just that it failed"
    )


def test_a_timeout_still_reports_the_server_log(tmp_path) -> None:
    class LiveProc:
        def poll(self):
            return None

    log = tmp_path / "server.log"
    log.write_text("still loading weights\n")
    with pytest.raises(ServerError, match="still loading weights"):
        real_wait_until_ready("http://127.0.0.1:1/v1", "k", timeout_s=0.3, poll_s=0.1,
                              proc=LiveProc(), log_path=log)


def test_server_output_is_captured_not_discarded(config, tmp_path, monkeypatch) -> None:
    """DEVNULL turns a one-line argparse error into an unexplained timeout."""
    monkeypatch.setattr("safety_eval.local_runner.LocalMatrixRunner.fetch",
                        lambda self, plan, log: make_cache(tmp_path, plan.repo))
    captured: dict = {}

    def recording_popen(cmd, **kwargs):
        captured["stdout"] = kwargs.get("stdout")
        return FakeProc(cmd, **kwargs)

    runner = LocalMatrixRunner(config, run_id="run-log", hf_home=tmp_path,
                               popen=recording_popen, eval_fn=stub_eval)
    runner.log_dir = tmp_path / "servers"
    runner.run(log=lambda _: None)
    assert captured["stdout"] is not None
    assert captured["stdout"] != subprocess.DEVNULL
    assert list((tmp_path / "servers").glob("vllm-*.log")), "no server log was written"


def test_the_serve_command_is_validated_against_the_installed_vllm(config) -> None:
    """vLLM's CLI moves between releases — `--disable-log-requests` was accepted in 0.27
    and removed in 0.28. Validating once up front converts a per-model timeout into an
    immediate, readable error."""
    from safety_eval.local_runner import validate_serve_command

    messages: list[str] = []
    validate_serve_command(config, messages.append)   # must not raise on a good config


def test_an_unknown_flag_is_rejected_before_any_model_is_served(config, monkeypatch) -> None:
    from safety_eval.local_runner import validate_serve_command

    monkeypatch.setattr("safety_eval.local_runner.serve_command",
                        lambda *a, **k: ["python", "--definitely-not-a-flag"])
    with pytest.raises(ServerError, match="does not accept"):
        validate_serve_command(config, lambda _: None)
