"""Unit tests for scheduler.py - S3 scanning and task triggering."""

import os
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@patch.dict(os.environ, {"S3_MODEL_BUCKET": "test-bucket"})
class TestScheduler:
    """Tests for scheduler functionality."""

    @patch("src.scheduler.get_s3_client")
    @patch("src.scheduler.config")
    def test_get_new_model_folders_finds_models(self, mock_config, mock_get_s3):
        """Test finding new model folders in S3."""
        from src.scheduler import get_new_model_folders

        mock_config.S3_RESULTS_PREFIX = "results/"

        # Mock S3 response
        recent_time = datetime.now(UTC) - timedelta(hours=12)
        old_time = datetime.now(UTC) - timedelta(hours=48)

        mock_s3 = MagicMock()
        mock_get_s3.return_value = mock_s3
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = iter(
            [
                {
                    "Contents": [
                        # Recent model file
                        {
                            "Key": "models/llama-3-8b/model.safetensors",
                            "LastModified": recent_time,
                        },
                        # Old model file
                        {
                            "Key": "models/old-model/model.bin",
                            "LastModified": old_time,
                        },
                        # Results file (should be ignored)
                        {
                            "Key": "results/some-result.json",
                            "LastModified": recent_time,
                        },
                    ]
                }
            ]
        )

        result = get_new_model_folders("test-bucket", hours=24)

        # Should only return the recent model folder
        assert len(result) == 1
        assert "models/llama-3-8b/" in result

    @patch("src.scheduler.get_s3_client")
    @patch("src.scheduler.config")
    def test_get_new_model_folders_skips_root_files(self, mock_config, mock_get_s3):
        """Test that root-level model files are skipped for safety."""
        from src.scheduler import get_new_model_folders

        mock_config.S3_RESULTS_PREFIX = "results/"

        recent_time = datetime.now(UTC) - timedelta(hours=12)

        mock_s3 = MagicMock()
        mock_get_s3.return_value = mock_s3
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = iter(
            [
                {
                    "Contents": [
                        # Root-level file (dangerous)
                        {
                            "Key": "model.gguf",
                            "LastModified": recent_time,
                        },
                    ]
                }
            ]
        )

        result = get_new_model_folders("test-bucket", hours=24)

        # Should not include root-level files
        assert len(result) == 0

    @patch("src.scheduler.get_s3_client")
    @patch("src.scheduler.config")
    def test_get_new_model_folders_empty_bucket(self, mock_config, mock_get_s3):
        """Test handling of empty S3 bucket."""
        from src.scheduler import get_new_model_folders

        mock_config.S3_RESULTS_PREFIX = "results/"

        mock_s3 = MagicMock()
        mock_get_s3.return_value = mock_s3
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = iter([{}])

        result = get_new_model_folders("test-bucket", hours=24)

        assert result == []

    @patch("src.scheduler.evaluate_model_task")
    @patch("src.scheduler.redis_client")
    @patch("src.scheduler.get_new_model_folders")
    @patch("src.scheduler.config")
    def test_trigger_nightly_queue_enqueues_new_models(
        self, mock_config, mock_get_folders, mock_redis, mock_task
    ):
        """Test that new models are enqueued correctly."""
        from src.scheduler import trigger_nightly_queue

        mock_config.S3_BUCKET_NAME = "test-bucket"

        # Mock finding 2 new model folders
        mock_get_folders.return_value = [
            "models/model-a/",
            "models/model-b/",
        ]

        # Mock Redis - neither model processed yet
        mock_redis.sismember.return_value = False

        trigger_nightly_queue()

        # Should enqueue both models
        assert mock_task.delay.call_count == 2

        # Should mark both as processed
        assert mock_redis.sadd.call_count == 2

    @patch("src.scheduler.evaluate_model_task")
    @patch("src.scheduler.redis_client")
    @patch("src.scheduler.get_new_model_folders")
    @patch("src.scheduler.config")
    def test_trigger_nightly_queue_skips_processed(
        self, mock_config, mock_get_folders, mock_redis, mock_task
    ):
        """Test that already-processed models are skipped."""
        from src.scheduler import trigger_nightly_queue

        mock_config.S3_BUCKET_NAME = "test-bucket"

        mock_get_folders.return_value = ["models/model-a/"]

        # Mock Redis - model already processed
        mock_redis.sismember.return_value = True

        trigger_nightly_queue()

        # Should not enqueue
        mock_task.delay.assert_not_called()

    @patch("src.scheduler.get_new_model_folders")
    @patch("src.scheduler.config")
    def test_trigger_nightly_queue_handles_s3_error(self, mock_config, mock_get_folders):
        """Test handling of S3 errors."""
        from src.scheduler import trigger_nightly_queue

        mock_config.S3_BUCKET_NAME = "test-bucket"

        # Mock S3 error
        mock_get_folders.side_effect = Exception("S3 connection failed")

        # Should not raise, just log
        trigger_nightly_queue()

    @patch("src.scheduler.evaluate_model_task")
    @patch("src.scheduler.redis_client")
    @patch("src.scheduler.get_new_model_folders")
    @patch("src.scheduler.config")
    def test_trigger_nightly_queue_no_new_models(
        self, mock_config, mock_get_folders, mock_redis, mock_task
    ):
        """Test behavior when no new models are found."""
        from src.scheduler import trigger_nightly_queue

        mock_config.S3_BUCKET_NAME = "test-bucket"

        mock_get_folders.return_value = []

        trigger_nightly_queue()

        # Should not enqueue anything
        mock_task.delay.assert_not_called()
        mock_redis.sadd.assert_not_called()
