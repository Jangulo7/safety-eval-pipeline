"""
Scheduler for LLM evaluation pipeline.

Scans S3 for new models and triggers evaluation tasks.
Uses shared S3 client for better connection pooling.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List

import redis

# Local imports
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from aws_clients import get_s3_client  # noqa: E402
from tasks import evaluate_model_task  # noqa: E402
import config  # noqa: E402

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Use the same Redis container as Celery
redis_client = redis.Redis(host="redis", port=6379, db=1)
PROCESSED_SET_KEY = "processed_models_history"


def get_new_model_folders(bucket_name: str, hours: int = 24) -> List[str]:
    """
    Scan S3 for 'folders' (prefixes) modified in the last N hours.

    Uses shared S3 client for better performance with connection pooling.

    Args:
        bucket_name: S3 bucket to scan
        hours: Look back window in hours

    Returns:
        List of S3 prefixes (folder paths) containing new models

    Example:
        >>> folders = get_new_model_folders('my-bucket', hours=24)
        >>> print(folders)
        ['models/llama-3-8b/', 'models/mistral-7b/']
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    candidates = set()

    # Use shared S3 client
    s3 = get_s3_client()

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name)

    for page in pages:
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            key = obj["Key"]

            # Ignore benchmark results to prevent infinite loops
            if config.S3_RESULTS_PREFIX in key:
                continue

            if obj["LastModified"] > cutoff:
                # Logic: If a file ends with model weights, get its parent folder
                if key.endswith((".safetensors", ".bin", ".gguf")):
                    # Safety check: avoid root-level files
                    folder_prefix = os.path.dirname(key)

                    # If folder_prefix is empty, the file is at bucket root.
                    # Downloading root ('/') would download the ENTIRE bucket.
                    if not folder_prefix:
                        logger.warning(f"Skipping root-level file '{key}' to prevent bucket dump.")
                        continue

                    folder_prefix = folder_prefix + "/"
                    candidates.add(folder_prefix)

    return list(candidates)


def trigger_nightly_queue():
    """
    Find new models and enqueue them if not already processed.

    Uses shared S3 client for better connection pooling performance.
    """
    logger.info("Starting nightly queue population...")

    try:
        model_folders = get_new_model_folders(config.S3_BUCKET_NAME)
    except Exception as e:
        logger.error(f"Failed to scan S3: {e}")
        return

    if not model_folders:
        logger.info("No modified model folders found in the last 24h.")
        return

    logger.info(f"Found {len(model_folders)} candidate folders.")

    count = 0
    for folder_prefix in model_folders:
        # Deduplication
        if not redis_client.sismember(PROCESSED_SET_KEY, folder_prefix):
            # Enqueue the FOLDER path
            evaluate_model_task.delay(folder_prefix)

            # Design Choice: We mark as processed HERE (Circuit Breaker).
            # If we move this to the worker and the worker SegFaults (crashes),
            # the scheduler will infinite-loop retry this bad model every hour.
            redis_client.sadd(PROCESSED_SET_KEY, folder_prefix)
            logger.info(f"Queued: {folder_prefix}")
            count += 1
        else:
            logger.debug(f"Skipping {folder_prefix} (Already Processed)")

    logger.info(f"Successfully added {count} new jobs to the queue.")


if __name__ == "__main__":
    trigger_nightly_queue()
