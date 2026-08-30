"""Sequential local-model orchestration: fetch → serve → evaluate → stop → purge.

Three 8B models at bf16 are ~48 GB of weights and ~48 GB of VRAM. Disk holds all of them;
the GPU does not. So weights are fetched up front — a download failure then costs nothing
but a retry, rather than stranding the run halfway through the matrix — and the models are
*served* one at a time. All three models' cells land in one run directory, so the result is
a single ``results.json`` and a single leaderboard.

**Weights this pipeline did not download are never deleted.** The cache is shared with
whatever else the machine is used for; a model that was already there when the run started
belongs to someone else. Purging is off by default and, when enabled, touches only weights
this run fetched.

The server is started from its own virtualenv. vLLM pins an exact torch build, so installing
it beside ``inspect_ai`` would replace the torch the rest of the pipeline sits on; a
subprocess over HTTP keeps the two apart, and is also how this would work against a model
served on another machine.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

from .config import ModelSpec, RunConfig
from .results import CellResult, ResultSet, RunMetadata, link_latest, new_run_id
from .runner import Runner, harness_versions

DEFAULT_VLLM_PYTHON = Path.home() / "venvs" / "vllm" / "bin" / "python"


def _relative(path: Any) -> str | None:
    """A path relative to the working directory, so records stay portable."""
    if not path:
        return None
    p = Path(str(path))
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return p.name


class ServerError(RuntimeError):
    """The vLLM server could not be started or did not become ready."""


@dataclass
class ModelPlan:
    """One model's turn: whether it must be fetched, and whether it may be purged."""

    model: ModelSpec
    repo: str
    pre_existing: bool
    cache_dir: Path

    @property
    def may_purge(self) -> bool:
        """Only weights this run fetched may be deleted."""
        return not self.pre_existing


@dataclass
class LocalRunSummary:
    """What happened, per model."""

    run_dir: Path
    results: ResultSet
    fetched: list[str] = field(default_factory=list)
    purged: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)


def hf_cache_dir(repo: str, home: Path | None = None) -> Path:
    """The HuggingFace hub directory for a repo id."""
    root = Path(home or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    return root / "hub" / ("models--" + repo.replace("/", "--"))


def is_cached(repo: str, home: Path | None = None) -> bool:
    """Whether a repo's weights are already present locally.

    Presence of the directory is not enough — a failed download leaves one behind — so this
    requires at least one weight shard.
    """
    path = hf_cache_dir(repo, home)
    if not path.is_dir():
        return False
    return any(path.rglob("*.safetensors")) or any(path.rglob("*.bin"))


def plan_models(config: RunConfig, home: Path | None = None) -> list[ModelPlan]:
    """Resolve each model to a repo id and record whether it was already on this machine."""
    plans = []
    for model in config.models:
        repo = model.id.split("/", 1)[1] if "/" in model.id else model.id
        plans.append(ModelPlan(model=model, repo=repo, pre_existing=is_cached(repo, home),
                               cache_dir=hf_cache_dir(repo, home)))
    return plans


def serve_command(config: RunConfig, repo: str, port: int) -> list[str]:
    """The ``vllm serve`` command line, built from the run's serving block.

    Every model is served with the same flags — that is the point of putting them in one
    place rather than typing them per model.
    """
    serving = config.raw.get("serving") or {}
    python = Path(serving.get("vllm_python") or DEFAULT_VLLM_PYTHON)
    cmd = [
        str(python), "-m", "vllm.entrypoints.openai.api_server",
        "--model", repo,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--api-key", os.environ.get("VLLM_API_KEY", "inspectai"),
        "--dtype", str(serving.get("dtype", "bfloat16")),
        "--max-model-len", str(serving.get("max_model_len", 8192)),
        "--gpu-memory-utilization", str(serving.get("gpu_memory_utilization", 0.85)),
        "--seed", str(config.defaults.seed),
    ]
    quantization = serving.get("quantization")
    if quantization and quantization != "none":
        # Applied at load time to the same bf16 checkpoint, so a precision comparison
        # changes exactly one variable and nothing else.
        cmd += ["--quantization", str(quantization)]
    return cmd


def validate_serve_command(config: RunConfig, log: Callable[[str], None]) -> None:
    """Check the serve command parses against the installed vLLM, before serving anything.

    vLLM's CLI moves between releases — `--disable-log-requests` was accepted in 0.27 and
    removed in 0.28. An unrecognised flag kills the server instantly, so validating once up
    front converts a per-model timeout into an immediate, readable error.
    """
    cmd = serve_command(config, "__validate__", 0)
    flags = {a for a in cmd if a.startswith("--")}
    try:
        helped = subprocess.run([cmd[0], "-m", "vllm.entrypoints.openai.api_server",
                                 "--help"], capture_output=True, text=True, timeout=180)
    except Exception as exc:
        log(f"  could not validate the serve command ({type(exc).__name__}); continuing")
        return
    text = helped.stdout + helped.stderr
    unknown = sorted(f for f in flags if f not in text)
    if unknown:
        raise ServerError(
            f"the installed vLLM does not accept {unknown}. The CLI changes between "
            "releases; update `serving:` in config/eval_config.yaml or the flags in "
            "serve_command()."
        )


def wait_until_ready(
    base_url: str,
    api_key: str,
    timeout_s: float = 900.0,
    poll_s: float = 5.0,
    proc: Any = None,
    log_path: Path | None = None,
) -> list[str]:
    """Block until the server answers, returning the model ids it serves.

    Watches the process as well as the port. A server that dies on startup -- an
    unrecognised flag, a CUDA failure, an OOM -- never binds the port, and polling alone
    cannot tell that apart from a slow model load. Without the process check a five-second
    argparse error costs the full timeout, once per model: waiting that looks like progress.
    """
    import requests

    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise ServerError(
                f"server process exited with code {proc.returncode} before becoming ready"
                + _tail(log_path)
            )
        try:
            resp = requests.get(f"{base_url.rstrip('/')}/models",
                                headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
            if resp.status_code == 200:
                return [m["id"] for m in resp.json().get("data", [])]
            last = f"HTTP {resp.status_code}"
        except Exception as exc:
            last = type(exc).__name__
        time.sleep(poll_s)
    raise ServerError(
        f"server at {base_url} not ready after {timeout_s:.0f}s ({last})" + _tail(log_path)
    )


def _tail(path: Path | None, lines: int = 12) -> str:
    """The end of the server log, so a failure says why rather than only that it did."""
    if path is None or not Path(path).exists():
        return ""
    text = Path(path).read_text(errors="replace").splitlines()
    return "\n    server log tail:\n    " + "\n    ".join(text[-lines:])


class LocalMatrixRunner:
    """Runs the matrix one model at a time against a locally served vLLM."""

    def __init__(
        self,
        config: RunConfig,
        *,
        run_id: str | None = None,
        port: int = 8000,
        purge: bool = False,
        hf_home: Path | None = None,
        popen: Callable[..., Any] = subprocess.Popen,
        eval_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id or new_run_id()
        self.port = port
        self.purge = purge
        self.hf_home = hf_home
        self._popen = popen
        self._eval_fn = eval_fn
        self.log_dir = Path(config.output.log_dir) / "servers"
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.api_key = os.environ.get("VLLM_API_KEY", "inspectai")
        self.versions = harness_versions()

    # ------------------------------------------------------------------ lifecycle

    def fetch(self, plan: ModelPlan, log: Callable[[str], None]) -> None:
        """Download weights, unless they are already here."""
        if plan.pre_existing:
            return
        from huggingface_hub import snapshot_download

        snapshot_download(
            plan.repo,
            token=os.environ.get("HF_TOKEN") or None,
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.jinja"],
        )

    def start_server(self, plan: ModelPlan, log: Callable[[str], None]) -> Any:
        """Launch vLLM and wait for it to answer."""
        cmd = serve_command(self.config, plan.repo, self.port)
        log(f"    starting vLLM ({' '.join(cmd[cmd.index('--dtype'):])})")
        # The server's own output is kept: a start-up failure is otherwise invisible, and
        # DEVNULL turns a one-line argparse error into an unexplained timeout.
        log_path = self.log_dir / f"vllm-{plan.repo.replace('/', '-')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        env = dict(os.environ)
        env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        # vLLM disables pinned memory on WSL by default and its V2 GPU runner then fails
        # with "UVA is not available". The gate is a 4.19.121 kernel; this host is far
        # above it, so the opt-in is safe and is what makes local serving work here at all.
        env.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
        proc = self._popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                           env=env, start_new_session=True)
        try:
            served = wait_until_ready(self.base_url, self.api_key, proc=proc,
                                      log_path=log_path)
        except ServerError:
            self.stop_server(proc, log)
            log(f"    server log: {log_path}")
            raise
        log(f"    server ready, serving: {', '.join(served)}")
        return proc

    def stop_server(self, proc: Any, log: Callable[[str], None]) -> None:
        """Shut the server down and wait for the VRAM to come back."""
        if proc is None or proc.poll() is not None:
            return
        log("    stopping server …")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, AttributeError):
            proc.terminate()
        try:
            proc.wait(timeout=120)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()

    def purge_weights(self, plan: ModelPlan, summary: LocalRunSummary,
                      log: Callable[[str], None]) -> None:
        """Delete the weights — but only if this run downloaded them."""
        if not plan.may_purge:
            summary.kept.append(plan.repo)
            log("    keeping weights: they were already on this machine before the run")
            return
        if not self.purge:
            summary.kept.append(plan.repo)
            return
        if plan.cache_dir.is_dir():
            shutil.rmtree(plan.cache_dir, ignore_errors=True)
            summary.purged.append(plan.repo)
            log(f"    purged {plan.cache_dir}")

    # ------------------------------------------------------------------ execution

    def run(self, log: Callable[[str], None] = print) -> LocalRunSummary:
        """Fetch, serve, evaluate, stop and purge each model in turn."""
        from datetime import datetime

        meta = RunMetadata(
            run_id=self.run_id,
            started_utc=datetime.now(UTC).isoformat(),
            config_path=_relative(self.config.source),
            provider=self.config.provider,
            grader_model=self.config.grader_model,
            limit=self.config.defaults.limit or 0,
            inspect_ai_version=self.versions["inspect_ai"],
            inspect_evals_version=self.versions["inspect_evals"],
            pipeline_version=self.versions["safety_eval"],
        )
        results = ResultSet(meta)
        summary = LocalRunSummary(
            run_dir=Path(self.config.output.results_dir) / self.run_id, results=results)

        validate_serve_command(self.config, log)
        plans = plan_models(self.config, self.hf_home)

        # Fetch everything before serving anything. A model that fails to download then
        # costs a retry rather than stranding the run with two of three models evaluated.
        missing = [p for p in plans if not p.pre_existing]
        if missing:
            log(f"  fetching {len(missing)} model(s) before serving any of them")
            for plan in missing:
                log(f"    {plan.repo}")
                try:
                    self.fetch(plan, log)
                    summary.fetched.append(plan.repo)
                except Exception as exc:
                    summary.failures[plan.repo] = f"download failed: {exc}"
                    log(f"    FAILED to download: {exc}")
        already = [p.repo for p in plans if p.pre_existing]
        if already:
            log(f"  already cached, will not be deleted: {', '.join(already)}")

        os.environ["VLLM_BASE_URL"] = self.base_url
        os.environ.setdefault("VLLM_API_KEY", self.api_key)

        for index, plan in enumerate(plans, start=1):
            log(f"\n  [{index}/{len(plans)}] {plan.model.label}  ({plan.repo})")
            if plan.repo in summary.failures:
                log("    skipped: weights are not available")
                continue
            proc = None
            try:
                proc = self.start_server(plan, log)
                for cell in self._evaluate(plan, log):
                    results.add(cell)
            except Exception as exc:
                summary.failures[plan.repo] = f"{type(exc).__name__}: {exc}"
                log(f"    FAILED: {type(exc).__name__}: {exc}")
            finally:
                self.stop_server(proc, log)
                self.purge_weights(plan, summary, log)
                # Persist after every model, so a failure late in the matrix does not throw
                # away the models that already completed.
                results.save(summary.run_dir / "results.json")

        meta.finished_utc = datetime.now(UTC).isoformat()
        results.save(summary.run_dir / "results.json")
        link_latest(summary.run_dir)
        return summary

    def _evaluate(self, plan: ModelPlan, log: Callable[[str], None]) -> Iterator[CellResult]:
        """Run every benchmark against the model currently being served."""
        scoped = self._scoped_config(plan.model)
        runner = Runner(scoped, run_id=self.run_id, eval_fn=self._eval_fn)

        def progress(cell_id: str, result: CellResult | None) -> None:
            if result is None:
                log(f"      {cell_id.split('::')[0]} …")
            elif result.ok:
                primary = next((m for m in result.metrics if m.primary), None)
                value = (f"{primary.label} {primary.value:.4g}"
                         f"{'%' if primary.unit == 'percent' else ''}" if primary else "ok")
                log(f"      -> {value}  ({result.wall_clock_s:.0f}s, "
                    f"{result.total_tokens:,} tok)")
            else:
                log(f"      -> {result.status.value.upper()}: "
                    f"{(result.error_message or '')[:100]}")

        yield from runner.run(progress=progress).cells

    def _scoped_config(self, model: ModelSpec) -> RunConfig:
        """A copy of the configuration restricted to one model."""
        from .catalog import Catalog

        data = dict(self.config.raw)
        data["models"] = [m for m in data["models"] if m["id"] == model.id]
        return RunConfig(data, self.config.catalog or Catalog.load())
