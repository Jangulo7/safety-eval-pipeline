"""
Unit tests for tasks.py - Celery task orchestrator.
"""
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock config before importing tasks
with patch.dict(os.environ, {"S3_MODEL_BUCKET": "test-bucket"}):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from src.tasks import (
        clear_gpu_cache,
        detect_model_type_and_path,
        download_s3_folder,
        run_lighteval_subprocess,
    )


class TestDownloadS3Folder:
    """Tests for S3 folder download functionality."""

    @patch("tasks.s3")
    @patch("tasks.os.makedirs")
    def test_download_success(self, mock_makedirs, mock_s3):
        """Test successful S3 folder download."""
        # Mock paginator
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "models/test/model.bin"},
                    {"Key": "models/test/config.json"},
                ]
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            download_s3_folder("test-bucket", "models/test/", tmpdir)
            
            # Verify download was called for each file
            assert mock_s3.download_file.call_count == 2

    @patch("tasks.s3")
    def test_download_no_files_raises_error(self, mock_s3):
        """Test that empty S3 prefix raises ValueError."""
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No files found"):
                download_s3_folder("test-bucket", "empty/", tmpdir)


class TestDetectModelTypeAndPath:
    """Tests for model type detection."""

    def test_detect_gguf(self):
        """Test GGUF model detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy GGUF file
            gguf_path = os.path.join(tmpdir, "model.gguf")
            with open(gguf_path, "wb") as f:
                f.write(b"dummy gguf content")

            model_path, model_type = detect_model_type_and_path(tmpdir)
            
            assert model_type == "gguf"
            assert model_path.endswith("model.gguf")

    def test_detect_huggingface(self):
        """Test HuggingFace model detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy config.json
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                f.write("{}")

            model_path, model_type = detect_model_type_and_path(tmpdir)
            
            assert model_type == "hf"
            assert model_path == tmpdir

    def test_detect_unknown_raises_error(self):
        """Test that unknown model type raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Could not identify model type"):
                detect_model_type_and_path(tmpdir)


class TestRunLightevalSubprocess:
    """Tests for LightEval subprocess execution."""

    @patch("tasks.subprocess.run")
    @patch("tasks.config")
    def test_lighteval_success(self, mock_config, mock_run):
        """Test successful LightEval execution."""
        mock_config.BENCHMARK_SUITE = ["task1", "task2"]
        mock_config.TRUST_REMOTE_CODE = True
        mock_config.GPU_IDS = "0"
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            output = run_lighteval_subprocess("/tmp/model", tmpdir, "gguf")
            assert output == "Success"
            mock_run.assert_called_once()

    @patch("tasks.subprocess.run")
    @patch("tasks.config")
    def test_lighteval_failure(self, mock_config, mock_run):
        """Test LightEval execution failure."""
        mock_config.BENCHMARK_SUITE = ["task1"]
        mock_config.TRUST_REMOTE_CODE = False
        mock_config.GPU_IDS = "0"
        
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error occurred"
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="Evaluation process failed"):
                run_lighteval_subprocess("/tmp/model", tmpdir, "hf")


class TestClearGPUCache:
    """Tests for GPU cache clearing."""

    @patch("tasks.torch")
    def test_clear_cache_with_cuda(self, mock_torch):
        """Test GPU cache clearing when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        
        clear_gpu_cache()
        
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("tasks.torch")
    def test_clear_cache_without_cuda(self, mock_torch):
        """Test GPU cache clearing when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False
        
        clear_gpu_cache()
        
        mock_torch.cuda.empty_cache.assert_not_called()

    def test_clear_cache_no_torch(self):
        """Test GPU cache clearing when PyTorch is not installed."""
        with patch.dict("sys.modules", {"torch": None}):
            # Should not raise an error
            clear_gpu_cache()


@pytest.mark.integration
class TestEvaluateModelTask:
    """Integration tests for the main evaluation task."""

    @patch("tasks.evaluate_model_task.retry")
    @patch("tasks.send_alert")
    @patch("tasks.s3")
    @patch("tasks.collect_evaluation_parameters")
    @patch("tasks.run_lighteval_subprocess")
    @patch("tasks.detect_model_type_and_path")
    @patch("tasks.download_s3_folder")
    def test_full_evaluation_success(
        self,
        mock_download,
        mock_detect,
        mock_lighteval,
        mock_collect,
        mock_s3,
        mock_alert,
        mock_retry,
    ):
        """Test successful full evaluation workflow."""
        # Setup mocks
        mock_detect.return_value = ("/tmp/model.gguf", "gguf")
        mock_collect.return_value = {"run_id": "test_123", "model_id": "test"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # This would need more setup for a full integration test
            # Simplified here for demonstration
            pass
