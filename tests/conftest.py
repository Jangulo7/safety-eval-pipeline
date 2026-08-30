"""Pytest fixtures.

The pre-existing fixtures below serve the S3 watcher and collector tests, which the
Inspect harness swap left untouched. Fixtures for the new pipeline live in
``conftest_safety`` and are re-exported here so both suites share one conftest.
"""

from conftest_safety import *
from conftest_safety import make_log, make_sample, make_score  # noqa: F401

"""Pytest configuration and shared fixtures."""

import tempfile
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_s3_client():
    """Provide a mock S3 client."""
    mock_client = MagicMock()

    # Mock common S3 operations
    mock_client.get_paginator.return_value = MagicMock()
    mock_client.download_file = MagicMock()
    mock_client.upload_file = MagicMock()
    mock_client.get_object = MagicMock()

    return mock_client


@pytest.fixture
def mock_redis_client():
    """Provide a mock Redis client."""
    mock_client = MagicMock()

    mock_client.sismember.return_value = False
    mock_client.sadd.return_value = 1

    return mock_client


@pytest.fixture
def sample_gguf_metadata():
    """Provide sample GGUF metadata for testing."""
    return {
        "base_model_family": "llama",
        "param_count_billions": 8.0,
        "max_context_window": 4096,
        "has_chat_template": True,
        "chat_template_preview": "<|im_start|>system...",
        "quant_file_type_id": 2,
        "bits_per_weight": 4.5,
        "file_size_gb": 4.8,
    }


@pytest.fixture
def sample_system_metadata():
    """Provide sample system metadata for testing."""
    return {
        "os_version": "Linux-5.15.0",
        "python_version": "3.8.10",
        "gpu_count": 1,
        "gpu_model": "NVIDIA H100",
        "driver_version": "535.104.05",
        "gpu_vram_total_gb": 80.0,
    }


@pytest.fixture
def sample_tool_metadata():
    """Provide sample tool metadata for testing."""
    return {
        "eval_tool_commit_hash": "abc123def456",
        "eval_tool_branch": "main",
        "python_package_versions": {
            "lighteval": "0.4.0",
            "torch": "2.1.0",
            "transformers": "4.35.0",
            "accelerate": "0.24.0",
        },
        "eval_tool_version": "0.4.0",
    }


@pytest.fixture
def sample_eval_config():
    """Provide sample evaluation configuration."""
    return {
        "run_id": "test-model_1234567890",
        "model_path": "/tmp/test-model.gguf",
        "model_id": "test-model",
        "benchmark_category": "General",
        "use_case_tags": ["chat"],
        "temperature": 0.0,
        "num_threads": 4,
        "random_seed": 42,
        "batch_size": 1,
    }


@pytest.fixture
def mock_config():
    """Provide a mock config module."""
    config = Mock()

    config.S3_BUCKET_NAME = "test-bucket"
    config.S3_RESULTS_PREFIX = "results/"
    config.TEMP_MODEL_DIR = "/tmp/eval_models"
    config.GPU_IDS = "0"
    config.BENCHMARK_SUITE = ["task1|subtask|5|0", "task2|subtask|0|0"]
    config.TRUST_REMOTE_CODE = False
    config.USE_CHAT_TEMPLATE = False

    return config


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up environment variables for all tests."""
    monkeypatch.setenv("S3_MODEL_BUCKET", "test-bucket")
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.com/webhook")


@pytest.fixture
def mock_gguf_reader():
    """Provide a mock GGUF reader."""
    mock_reader = MagicMock()

    # Mock field structure
    mock_field = Mock()
    mock_field.key = "general.architecture"
    mock_field.parts = ["llama"]

    mock_reader.fields = {"general.architecture": mock_field}

    return mock_reader
