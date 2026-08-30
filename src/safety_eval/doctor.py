"""Preflight checks — everything that can go wrong, found before any money is spent.

A 3x4 matrix at n=50 is 600 graded samples and a real bill. Discovering afterwards that the
XSTest dataset is gated, that a model id is not on OpenRouter, or that an ``inspect_evals``
upgrade renamed a metric the catalog addresses, is an expensive way to learn it. ``doctor``
finds all of that first.

The metric-drift check is the one worth pointing at: it runs each configured benchmark
against Inspect's built-in ``mockllm`` provider with two samples, reads the metric addresses
out of the resulting log, and diffs them against ``config/benchmarks.yaml``. That costs
nothing, needs no credentials, and is the only way to know that the catalog still describes
the harness that is installed.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .catalog import BenchmarkSpec, Catalog
from .config import ConfigError, RunConfig
from .runner import harness_versions


class Level(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class Check:
    """One preflight finding, with the fix spelled out."""

    name: str
    level: Level
    detail: str
    fix: str = ""

    @property
    def icon(self) -> str:
        return {Level.OK: "PASS", Level.WARN: "WARN", Level.FAIL: "FAIL",
                Level.SKIP: "SKIP"}[self.level]


@dataclass
class Diagnosis:
    """The full preflight result."""

    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, level: Level, detail: str, fix: str = "") -> None:
        self.checks.append(Check(name, level, detail, fix))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.level is Level.FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.level is Level.WARN]

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def render(self) -> str:
        width = max((len(c.name) for c in self.checks), default=10)
        lines = []
        for c in self.checks:
            lines.append(f"  [{c.icon}] {c.name.ljust(width)}  {c.detail}")
            if c.fix and c.level in (Level.FAIL, Level.WARN):
                for i, part in enumerate(c.fix.split("\n")):
                    lines.append(f"         {' ' * width}  {'-> ' if i == 0 else '   '}{part}")
        lines.append("")
        lines.append(
            f"  {len(self.checks) - len(self.failures) - len(self.warnings)} ok, "
            f"{len(self.warnings)} warning(s), {len(self.failures)} failure(s)"
        )
        return "\n".join(lines)


def diagnose(
    config: RunConfig | None = None,
    catalog: Catalog | None = None,
    *,
    check_metrics: bool = False,
    check_network: bool = True,
    config_path: str | Path | None = None,
) -> Diagnosis:
    """Run every preflight check.

    Args:
        config: an already-loaded config, or ``None`` to load and validate one here (so that
            a config error is itself reported as a check rather than a traceback).
        check_metrics: additionally run each benchmark against ``mockllm`` and diff the real
            metric addresses against the catalog. Costs nothing but takes ~30s.
        check_network: allow outbound calls (OpenRouter catalogue, HF dataset heads).
    """
    d = Diagnosis()
    catalog = catalog or Catalog.load()

    _check_versions(d, catalog)

    if config is None:
        try:
            config = RunConfig.load(config_path)
            d.add("config", Level.OK,
                  f"{len(config.models)} models x {len(config.tasks)} tasks = "
                  f"{len(config.cells)} cells, {config.estimated_samples()} samples requested")
        except (ConfigError, FileNotFoundError) as exc:
            d.add("config", Level.FAIL, str(exc),
                  "fix config/eval_config.yaml; every gate bound is checked against the "
                  "metric's catalog range, so a unit mistake is reported here")
            return d
    else:
        d.add("config", Level.OK,
              f"{len(config.cells)} cells, {config.estimated_samples()} samples requested")

    _check_credentials(d, config)
    _check_datasets(d, config, catalog, check_network=check_network)
    if check_network:
        _check_openrouter(d, config)
    _check_aws(d)
    _check_log_safety(d, config, catalog)
    if check_metrics:
        _check_metric_drift(d, config, catalog)
    return d


# ------------------------------------------------------------------------------ checks

def _check_versions(d: Diagnosis, catalog: Catalog) -> None:
    versions = harness_versions()
    if versions["inspect_ai"] == "unknown":
        d.add("harness", Level.FAIL, "inspect_ai is not installed",
              "pip install -r requirements.txt")
        return

    expected = catalog.verified_against
    drift = [
        f"{name} {versions[name]} (catalog verified against {expected[name]})"
        for name in ("inspect_ai", "inspect_evals")
        if expected.get(name) and expected[name] != versions.get(name)
    ]
    detail = (f"inspect_ai {versions['inspect_ai']}, "
              f"inspect_evals {versions['inspect_evals']}, "
              f"pipeline {versions['safety_eval']}")
    if drift:
        d.add("harness", Level.WARN, detail + " — " + "; ".join(drift),
              "run `safety-eval doctor --metrics` to confirm the catalog still describes the "
              "installed harness; an inspect_evals upgrade can rename a score or change a "
              "scorer, which changes the numbers")
    else:
        d.add("harness", Level.OK, detail + " (matches the catalog)")


def _check_credentials(d: Diagnosis, config: RunConfig) -> None:
    providers = {m.provider for m in config.models}
    env_for = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "together": "TOGETHER_API_KEY",
        "bedrock": None,  # boto3 credential chain
        "mockllm": None,
    }
    for provider in sorted(providers):
        var = env_for.get(provider, f"{provider.upper()}_API_KEY")
        if var is None:
            d.add(f"creds:{provider}", Level.OK, "uses the ambient credential chain")
            continue
        value = os.environ.get(var, "")
        if not value:
            d.add(f"creds:{provider}", Level.FAIL, f"{var} is not set",
                  f"add {var}=... to .env (copy .env.example), or export it")
        elif len(value) < 12:
            d.add(f"creds:{provider}", Level.WARN,
                  f"{var} is set but looks too short ({len(value)} chars)",
                  "check the value was pasted in full")
        else:
            d.add(f"creds:{provider}", Level.OK, f"{var} set ({_mask(value)})")

    # The grader is a separate spend and a separate failure mode: it is what turns a
    # completion into a number, so a missing grader key produces a full matrix of nans.
    grader_provider = config.grader_model.split("/", 1)[0]
    if grader_provider not in providers:
        var = env_for.get(grader_provider, f"{grader_provider.upper()}_API_KEY")
        level = Level.OK if (var is None or os.environ.get(var)) else Level.FAIL
        d.add("creds:grader", level,
              f"grader {config.grader_model} needs {var or 'ambient credentials'}",
              "without it every judge-graded metric comes back nan")


def _check_datasets(
    d: Diagnosis, config: RunConfig, catalog: Catalog, *, check_network: bool
) -> None:
    """Gated datasets are the single most common blocker, and the cheapest to detect."""
    for bench in _benchmarks_in(config, catalog):
        name = f"dataset:{bench.key}"
        source = bench.dataset.get("source", "unknown")
        if not bench.gated:
            d.add(name, Level.OK, f"{source} (not gated)")
            continue

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            d.add(name, Level.FAIL, f"{source} is GATED and no HF_TOKEN is set",
                  f"1. request access at {bench.dataset.get('gate_url', source)}\n"
                  "2. create a read token at https://huggingface.co/settings/tokens\n"
                  "3. add HF_TOKEN=hf_... to .env\n"
                  f"Without it, every {bench.key} cell is recorded as 'blocked'.")
            continue
        if not check_network:
            d.add(name, Level.SKIP, f"{source} is gated; HF_TOKEN set, access not verified")
            continue

        ok, message = _hf_access(bench.dataset.get("source", ""), token)
        if ok:
            d.add(name, Level.OK, f"{source} — access granted")
        else:
            d.add(name, Level.FAIL, f"{source} — {message}",
                  f"the token is set but this account may not have accepted the terms at "
                  f"{bench.dataset.get('gate_url', source)}")


def _hf_access(source: str, token: str) -> tuple[bool, str]:
    """Ask the Hub whether this token can actually read the dataset."""
    repo = source.split()[-1] if " " in source else source
    repo = repo.strip()
    if "/" not in repo:
        return False, f"cannot parse a repo id out of {source!r}"
    try:
        import requests

        resp = requests.get(
            f"https://huggingface.co/api/datasets/{repo}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
    except Exception as exc:
        return False, f"could not reach the Hub ({type(exc).__name__})"
    if resp.status_code == 200:
        return True, "ok"
    if resp.status_code in (401, 403):
        return False, f"HTTP {resp.status_code}: access not granted for this token"
    if resp.status_code == 404:
        return False, "HTTP 404: repo not found (or invisible to this token)"
    return False, f"HTTP {resp.status_code}"


def _check_openrouter(d: Diagnosis, config: RunConfig) -> None:
    """Confirm every configured model id actually exists on OpenRouter.

    A typo in a model id costs a full matrix of errors and is invisible until the run starts.
    """
    wanted = [m for m in config.models if m.provider == "openrouter"]
    grader_is_or = config.grader_model.startswith("openrouter/")
    if not wanted and not grader_is_or:
        d.add("openrouter", Level.SKIP, "no OpenRouter models configured")
        return
    if not os.environ.get("OPENROUTER_API_KEY"):
        d.add("openrouter", Level.SKIP, "no key; model ids not verified")
        return

    available = list_openrouter_models()
    if available is None:
        d.add("openrouter", Level.WARN, "could not reach the OpenRouter model catalogue",
              "model ids were not verified; a typo will surface as an errored cell")
        return

    ids = {m["id"] for m in available}
    missing = [m.id for m in wanted if m.id.split("/", 1)[1] not in ids]
    if grader_is_or and config.grader_model.split("/", 1)[1] not in ids:
        missing.append(config.grader_model + " (grader)")

    if missing:
        d.add("openrouter", Level.FAIL,
              f"{len(missing)} model id(s) not in the OpenRouter catalogue: "
              f"{', '.join(missing)}",
              "run `safety-eval list-models --search <name>` to find the exact id")
    else:
        d.add("openrouter", Level.OK,
              f"all {len(wanted) + (1 if grader_is_or else 0)} model id(s) exist "
              f"({len(ids)} available)")


def _check_aws(d: Diagnosis) -> None:
    """The S3 artefact store is optional; report it as configured or not, never as broken."""
    bucket = os.environ.get("S3_MODEL_BUCKET", "")
    if not bucket:
        d.add("aws:s3", Level.SKIP, "S3_MODEL_BUCKET not set; the model watcher is off")
        return
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError

        boto3.client("s3").head_bucket(Bucket=bucket)
        d.add("aws:s3", Level.OK, f"bucket {bucket} reachable")
    except ImportError:
        d.add("aws:s3", Level.WARN, "boto3 not installed", "pip install boto3")
    except (BotoCoreError, ClientError) as exc:
        d.add("aws:s3", Level.WARN, f"bucket {bucket}: {type(exc).__name__}",
              "check AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION")
    except Exception as exc:
        d.add("aws:s3", Level.WARN, f"bucket {bucket}: {exc}")


def _check_log_safety(d: Diagnosis, config: RunConfig, catalog: Catalog) -> None:
    """The one check that is about publication, not about running.

    Any benchmark whose catalog entry withholds transcripts must have its log directory
    gitignored. A repo that publishes a working jailbreak has failed regardless of its scores.
    """
    withheld = [b for b in _benchmarks_in(config, catalog) if not b.publish_logs]
    if not withheld:
        d.add("log safety", Level.OK, "no benchmark withholds transcripts")
        return

    gitignore = Path(".gitignore")
    text = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    unprotected = []
    for bench in withheld:
        keys = [t.key for t in config.tasks if t.benchmark == bench.key]
        if not any(f"logs/{k}" in text or f"logs/{bench.key}" in text for k in keys + [bench.key]):
            unprotected.append(bench.key)

    if unprotected:
        d.add("log safety", Level.FAIL,
              f"{', '.join(unprotected)} withholds transcripts but its log directory is not "
              "gitignored",
              "add `logs/<task>/` to .gitignore — these logs contain model responses to "
              "forbidden prompts and must never be committed")
    else:
        d.add("log safety", Level.OK,
              f"transcripts withheld and gitignored for: "
              f"{', '.join(b.key for b in withheld)}")


def _check_metric_drift(d: Diagnosis, config: RunConfig, catalog: Catalog) -> None:
    """Diff the catalog's metric addresses against what the installed harness really emits.

    Runs each benchmark against Inspect's ``mockllm`` provider with two samples. No
    credentials, no cost. This is the check that catches an ``inspect_evals`` upgrade having
    renamed a score under the catalog's feet, which would otherwise show up as a full matrix
    of "metric absent from the log".
    """
    for bench in _benchmarks_in(config, catalog):
        name = f"metrics:{bench.key}"
        if bench.gated and not (os.environ.get("HF_TOKEN")
                                or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            d.add(name, Level.SKIP, "gated dataset; cannot probe without HF_TOKEN")
            continue
        try:
            found, version = probe_metrics(bench, config)
        except Exception as exc:
            d.add(name, Level.WARN, f"probe failed: {type(exc).__name__}: {exc}",
                  "the catalog could not be verified against the installed harness")
            continue

        expected = set(bench.metrics)
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        version_note = ""
        if bench.task_version_expected and version and version != bench.task_version_expected:
            version_note = (f" · task version {version}, catalog expects "
                            f"{bench.task_version_expected}")

        if missing:
            d.add(name, Level.FAIL,
                  f"catalog addresses {len(missing)} metric(s) the harness does not emit: "
                  f"{', '.join(missing)}{version_note}",
                  f"the log actually carries: {', '.join(sorted(found))}\n"
                  "update config/benchmarks.yaml to match")
        elif extra or version_note:
            d.add(name, Level.WARN,
                  f"{len(expected)} catalog metric(s) confirmed"
                  + (f"; harness also emits {', '.join(extra)}" if extra else "")
                  + version_note,
                  "unlisted metrics are simply not reported; a version mismatch means the "
                  "numbers may not be comparable with an earlier run")
        else:
            d.add(name, Level.OK,
                  f"all {len(expected)} catalog metric(s) confirmed against a real log "
                  f"(task version {version or '?'})")


def probe_metrics(bench: BenchmarkSpec, config: RunConfig) -> tuple[set[str], str | None]:
    """Run a benchmark against ``mockllm`` and return the metric addresses it really emits."""
    import tempfile

    from inspect_ai import eval as inspect_eval

    task = next(t for t in config.tasks if t.benchmark == bench.key)
    args = dict(task.args)
    args[bench.grader_kwarg] = "mockllm/model"

    with tempfile.TemporaryDirectory() as tmp:
        logs = inspect_eval(
            tasks=bench.task, task_args=args, model="mockllm/model", limit=2,
            log_dir=tmp, display="none", fail_on_error=False,
        )
    if not logs:
        raise RuntimeError("mock eval produced no log")
    log = logs[0]
    results = getattr(log, "results", None)
    if results is None:
        raise RuntimeError(f"mock eval produced no results (status={log.status})")

    found = {f"{s.name}/{k}" for s in results.scores for k in s.metrics}
    version = (log.eval.metadata or {}).get("full_task_version")
    return found, version


def list_openrouter_models(timeout: float = 20.0) -> list[dict[str, Any]] | None:
    """Fetch the OpenRouter model catalogue, or ``None`` if it cannot be reached.

    Used by ``doctor`` to validate configured ids and by the Streamlit sidebar to populate
    the model picker, so the UI offers only models that actually exist.
    """
    try:
        import requests

        key = os.environ.get("OPENROUTER_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        resp = requests.get("https://openrouter.ai/api/v1/models", headers=headers,
                            timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception:
        return None


def _benchmarks_in(config: RunConfig, catalog: Catalog) -> Iterable[BenchmarkSpec]:
    seen: set[str] = set()
    for task in config.tasks:
        if task.benchmark not in seen:
            seen.add(task.benchmark)
            yield catalog[task.benchmark]


def _mask(value: str) -> str:
    return f"{value[:6]}…{value[-4:]}" if len(value) > 14 else "…"
