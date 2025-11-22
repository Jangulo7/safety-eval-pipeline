import logging
from datetime import datetime, timedelta, timezone
import boto3
from tasks import evaluate_model_task
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
s3 = boto3.client('s3')

def get_recent_models(hours: int = 24):
    """Scans S3 for models updated in the last N hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    candidates = []
    
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=config.S3_BUCKET_NAME):
        if 'Contents' not in page: continue
        for obj in page['Contents']:
            if obj['LastModified'] > cutoff:
                # Logic: Identify model folders or GGUF/SafeTensors files
                key = obj['Key']
                # Filter out results folder to avoid infinite loops
                if "benchmarks/" in key: continue 
                if key.endswith(('.safetensors', '.bin', '.gguf')):
                    # If it's a file in a folder, get the folder path (virtual HF repo)
                    # Or strictly pass the file if using GGUF
                    candidates.append(key)
    return list(set(candidates))

if __name__ == "__main__":
    logger.info("Running nightly scheduler...")
    models = get_recent_models()
    for m in models:
        evaluate_model_task.delay(m)
        logger.info(f"Queued: {m}")