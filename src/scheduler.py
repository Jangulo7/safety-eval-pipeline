import logging
import os 
import boto3
import redis
from typing import List  # Fixed: Compatible with Py3.8+
from datetime import datetime, timedelta, timezone
from tasks import evaluate_model_task
import config

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

s3 = boto3.client('s3')
# Use the same Redis container as Celery
redis_client = redis.Redis(host='redis', port=6379, db=1)
PROCESSED_SET_KEY = "processed_models_history"

def get_new_model_folders(bucket_name: str, hours: int = 24) -> List[str]:
    """
    Scans S3 for 'folders' (prefixes) modified in the last N hours.
    Returns a list of prefixes (e.g., 'models/llama-3-8b-quantized/').
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    candidates = set()

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name)

    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            # Ignore benchmark results to prevent infinite loops
            if config.S3_RESULTS_PREFIX in key:
                continue

            if obj['LastModified'] > cutoff:
                # Logic: If a file ends with model weights, get its parent folder
                if key.endswith(('.safetensors', '.bin', '.gguf')):
                    # Fixed: Root-level file safety check
                    folder_prefix = os.path.dirname(key)
                    
                    # If folder_prefix is empty, the file is at bucket root.
                    # Downloading root ('/') would download the ENTIRE bucket.
                    if not folder_prefix:
                        logger.warning(f"Skipping root-level file '{key}' to prevent bucket dump.")
                        continue
                        
                    folder_prefix = folder_prefix + '/'
                    candidates.add(folder_prefix)
    
    return list(candidates)

def trigger_nightly_queue():
    """Finds new models and enqueues them if not processed."""
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