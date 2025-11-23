"""
Unit tests for collectors module - Metadata extraction.
"""
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.collectors.master_collector import collect_evaluation_parameters


class TestMasterCollector:
    """Tests for master metadata collector."""

    @patch("collectors.master_collector.get_tool_metadata")
    @patch("collectors.master_collector.get_system_metadata")
    @patch("collectors.master_collector.get_gguf_metadata")
    @patch("collectors.master_collector.calculate_sha256")
    def test_collect_full_metadata(
        self,
        mock_hash,
        mock_gguf,
        mock_system,
        mock_tool,
        sample_eval_config,
        sample_gguf_metadata,
        sample_system_metadata,
        sample_tool_metadata,
    ):
        """Test full metadata collection with all data sources."""
        # Setup mocks
        mock_gguf.return_value = sample_gguf_metadata
        mock_system.return_value = sample_system_metadata
        mock_tool.return_value = sample_tool_metadata
        mock_hash.return_value = "abc123" * 10  # Valid SHA256 length

        result = collect_evaluation_parameters(sample_eval_config)

        # Verify all data sources were called
        mock_gguf.assert_called_once()
        mock_system.assert_called_once()
        mock_tool.assert_called_once()
        mock_hash.assert_called_once()

        # Verify result contains expected keys
        assert "run_id" in result
        assert "timestamp_utc" in result
        assert "base_model_family" in result  # From GGUF
        assert "gpu_model" in result  # From system
        assert "eval_tool_version" in result  # From tool
        assert "file_hash_sha256" in result

    @patch("collectors.master_collector.get_tool_metadata")
    @patch("collectors.master_collector.get_system_metadata")
    @patch("collectors.master_collector.get_gguf_metadata")
    def test_collect_without_model_path(
        self,
        mock_gguf,
        mock_system,
        mock_tool,
        sample_system_metadata,
        sample_tool_metadata,
    ):
        """Test metadata collection without model path."""
        mock_system.return_value = sample_system_metadata
        mock_tool.return_value = sample_tool_metadata

        config = {
            "run_id": "test_123",
            "model_id": "test-model",
        }

        result = collect_evaluation_parameters(config)

        # GGUF metadata should not be called
        mock_gguf.assert_not_called()

        # But system and tool should be
        mock_system.assert_called_once()
        mock_tool.assert_called_once()

        # Basic fields should exist
        assert result["run_id"] == "test_123"
        assert result["model_id"] == "test-model"

    @patch("collectors.master_collector.get_tool_metadata")
    @patch("collectors.master_collector.get_system_metadata")
    @patch("collectors.master_collector.get_gguf_metadata")
    @patch("collectors.master_collector.calculate_sha256")
    def test_collect_with_chat_template_missing(
        self,
        mock_hash,
        mock_gguf,
        mock_system,
        mock_tool,
        sample_eval_config,
        sample_gguf_metadata,
        sample_system_metadata,
        sample_tool_metadata,
    ):
        """Test risk flag when chat template is missing for chat use case."""
        # Mock GGUF without chat template
        gguf_no_chat = sample_gguf_metadata.copy()
        gguf_no_chat["has_chat_template"] = False

        mock_gguf.return_value = gguf_no_chat
        mock_system.return_value = sample_system_metadata
        mock_tool.return_value = sample_tool_metadata
        mock_hash.return_value = "abc" * 21

        # Config with chat use case
        config = sample_eval_config.copy()
        config["use_case_tags"] = ["chat"]

        result = collect_evaluation_parameters(config)

        # Should have risk flag
        assert "RISK_FLAG" in result
        assert result["RISK_FLAG"] == "MISSING_CHAT_TEMPLATE"

    @patch("collectors.master_collector.get_tool_metadata")
    @patch("collectors.master_collector.get_system_metadata")
    @patch("collectors.master_collector.get_gguf_metadata")
    @patch("collectors.master_collector.calculate_sha256")
    def test_collect_with_reproducibility_defaults(
        self,
        mock_hash,
        mock_gguf,
        mock_system,
        mock_tool,
        sample_gguf_metadata,
        sample_system_metadata,
        sample_tool_metadata,
    ):
        """Test that reproducibility defaults are applied."""
        mock_gguf.return_value = sample_gguf_metadata
        mock_system.return_value = sample_system_metadata
        mock_tool.return_value = sample_tool_metadata
        mock_hash.return_value = "abc" * 21

        config = {
            "run_id": "test_123",
            "model_path": "/tmp/model.gguf",
            "model_id": "test",
        }

        result = collect_evaluation_parameters(config)

        # Check reproducibility fields have defaults
        assert result["num_threads"] == 4
        assert result["temperature"] == 0.0
        assert result["random_seed"] == 42
        assert result["batch_size"] == 1

    @patch("collectors.master_collector.calculate_sha256")
    def test_collect_with_hash_failure(
        self,
        mock_hash,
        sample_eval_config,
    ):
        """Test behavior when file hashing fails."""
        # Mock hash returning None (file not found)
        mock_hash.return_value = None

        with patch("collectors.master_collector.get_gguf_metadata") as mock_gguf:
            with patch("collectors.master_collector.get_system_metadata"):
                with patch("collectors.master_collector.get_tool_metadata"):
                    mock_gguf.return_value = {"valid_gguf": True}
                    
                    result = collect_evaluation_parameters(sample_eval_config)

                    # Should not have hash in result
                    assert "file_hash_sha256" not in result
