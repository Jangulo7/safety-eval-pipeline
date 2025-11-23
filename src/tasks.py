"""
Celery task orchestrator for LLM model evaluation pipeline.

This module handles:
- Downloading models from S3 (using shared client for connection pooling)
- Running LightEval benchmarks
- Collecting metadata
- Uploading results
- Cleanup operations
"""
import glob
import json
import logging
import os
import shutil
import subprocess
import time
from typing import Optional, Tuple

from celery import Celery

# Local imports - adjusted for src/ structure
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from aws_clients import get_s3_client  # Use shared client
from collectors.master_collector import collect_evaluation_parameters
from notifications import send_alert

app = Celery("ai_evaluator", broker="redis://redis:6379/0")

# Configure logging specific to worker
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_s3_folder(bucket: str, s3_folder_prefix: str, local_dir: str) -> None:
    """
    Recursively download an S3 'folder' to a local directory.

    Essential for HuggingFace models that rely on config.json + tokenizer files.
    Uses shared S3 client for better connection pooling.

    Args:
        bucket: S3 bucket name
        s3_folder_prefix: S3 prefix (folder path)
        local_dir: Local directory to download to

    Raises:
        ValueError: If no files found at S3 prefix
    """
    logger.info(f"Downloading folder {s3_folder_prefix} to {local_dir}")
    os.makedirs(local_dir, exist_ok=True)

    # Use shared S3 client instead of creating new one
    s3 = get_s3_client()

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=s3_folder_prefix)

    files_downloaded = 0
    for page in pages:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith("/"):
                continue

            rel_path = key[len(s3_folder_prefix) :]
            if rel_path.startswith("/"):
                rel_path = rel_path[1:]

            local_file_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            s3.download_file(bucket, key, local_file_path)
            files_downloaded += 1

    if files_downloaded == 0:
        raise ValueError(f"No files found at S3 prefix: {s3_folder_prefix}")

    logger.info(f"Downloaded {files_downloaded} files.")


def detect_model_type_and_path(local_dir: str) -> Tuple[str, str]:
    """
    Determine if we are dealing with a GGUF file or a HuggingFace directory.

    Args:
        local_dir: Local directory containing model files

    Returns:
        Tuple of (model_path, backend_type) where backend_type is 'gguf' or 'hf'

    Raises:
        ValueError: If model type cannot be determined
    """
    gguf_files = glob.glob(os.path.join(local_dir, "*.gguf"))

    if gguf_files:
        main_gguf = max(gguf_files, key=os.path.getsize)
        logger.info(f"Detected GGUF format: {main_gguf}")
        return main_gguf, "gguf"

    if os.path.exists(os.path.join(local_dir, "config.json")):
        logger.info("Detected Standard HuggingFace directory format.")
        return local_dir, "hf"

    raise ValueError(
        f"Could not identify model type in {local_dir} (No .gguf or config.json)"
    )


def run_lighteval_subprocess(
    model_path: str, output_dir: str, model_type: str
) -> str:
    """
    Run LightEval evaluation. Adapts arguments based on model type.

    Args:
        model_path: Path to model file or directory
        output_dir: Directory for output results
        model_type: Type of model ('gguf' or 'hf')

    Returns:
        stdout from LightEval process

    Raises:
        RuntimeError: If evaluation process fails
    """
    tasks_str = ",".join(config.BENCHMARK_SUITE)

    cmd = ["lighteval", "accelerate"]

    # Both GGUF and HF use same model_args format in newer versions
    trust_code = "True" if config.TRUST_REMOTE_CODE else "False"
    model_args = f"pretrained={model_path},trust_remote_code={trust_code}"

    cmd.extend([
        "--model_args", model_args,
        "--tasks", tasks_str,
        "--output_dir", output_dir,
        "--save_details"
    ])

    logger.info(f"Executing LightEval: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": config.GPU_IDS},
    )

    if result.stdout:
        logger.info(f"LightEval STDOUT (Truncated): {result.stdout[:1000]}")

    if result.returncode != 0:
        logger.error(f"LightEval STDERR: {result.stderr}")
        raise RuntimeError(
            f"Evaluation process failed with code {result.returncode}"
        )

    return result.stdout


def clear_gpu_cache() -> None:
    """Clear GPU cache to prevent OOM errors on subsequent runs."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    except ImportError:
        logger.debug("PyTorch not available, skipping GPU cache clear")


@app.task(bind=True, max_retries=1)
def evaluate_model_task(self, s3_folder_prefix: str) -> None:
    """
    Main Celery task for model evaluation.

    Downloads model from S3, runs benchmarks, collects metadata, and uploads results.
    Uses shared S3 client for better connection pooling performance.

    Args:
        s3_folder_prefix: S3 prefix (folder path) containing the model

    Raises:
        Exception: Any error during evaluation (will trigger Celery retry)
    """
    start_time = time.time()
    model_name = s3_folder_prefix.strip("/").split("/")[-1]
    run_id = f"{model_name}_{int(start_time)}"

    local_model_dir = os.path.join(config.TEMP_MODEL_DIR, model_name)
    local_output_dir = os.path.join(config.TEMP_MODEL_DIR, "results", run_id)

    # Create output dir early for metadata
    os.makedirs(local_output_dir, exist_ok=True)

    # Get shared S3 client
    s3 = get_s3_client()

    try:
        # 1. Download Full Directory
        download_s3_folder(
            config.S3_BUCKET_NAME, s3_folder_prefix, local_model_dir
        )

        # 2. Detect Type (GGUF vs HF)
        target_path, model_type = detect_model_type_and_path(local_model_dir)

        # 3. Metadata Collection
        logger.info("Running Metadata Collection...")

        collector_config = {
            "run_id": run_id,
            "model_path": target_path,
            "model_id": model_name,
            "benchmark_category": "General",
            "use_case_tags": ["chat"] if config.USE_CHAT_TEMPLATE else [],
            "temperature": 0.0,
        }

        eval_meta = collect_evaluation_parameters(collector_config)

        meta_path = os.path.join(local_output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(eval_meta, f, indent=4)

        # CRITICAL: Safety Gate
        if "RISK_FLAG" in eval_meta:
            risk = eval_meta["RISK_FLAG"]
            logger.error(f"Aborting evaluation due to Risk Flag: {risk}")

            # Upload the metadata so we know WHY it failed
            s3.upload_file(
                meta_path,
                config.S3_BUCKET_NAME,
                f"{config.S3_RESULTS_PREFIX}{run_id}/metadata.json",
            )
            send_alert(model_name, "SKIPPED", error_msg=f"Risk Flag: {risk}")
            return  # Exit task early

        # 4. Evaluate (LightEval)
        run_lighteval_subprocess(target_path, local_output_dir, model_type)

        # Clear GPU cache after evaluation
        clear_gpu_cache()

        # 5. Upload Results
        if not os.path.exists(local_output_dir):
            logger.warning(
                f"Output directory {local_output_dir} was not created by LightEval."
            )
            send_alert(model_name, "FAILURE", error_msg="Output directory missing")
            return

        files_uploaded = 0
        for root, _, files in os.walk(local_output_dir):
            for file in files:
                if file.endswith((".json", ".parquet")):
                    local_file = os.path.join(root, file)

                    # Calculate relative path to maintain folder structure in S3
                    rel_path = os.path.relpath(local_file, local_output_dir)
                    s3_dest = f"{config.S3_RESULTS_PREFIX}{run_id}/{rel_path}"

                    s3.upload_file(local_file, config.S3_BUCKET_NAME, s3_dest)
                    files_uploaded += 1

        if files_uploaded > 0:
            elapsed = f"{(time.time() - start_time)/60:.1f}m"
            send_alert(model_name, "SUCCESS", run_time=elapsed)
        else:
            logger.warning("No JSON/Parquet results found to upload.")
            send_alert(
                model_name, "FAILURE", error_msg="No results generated"
            )

    except Exception as e:
        logger.error(f"Task failed for {model_name}: {str(e)}")
        send_alert(model_name, "FAILURE", error_msg=str(e))
        raise self.retry(exc=e)

    finally:
        # 6. Aggressive Cleanup
        if os.path.exists(local_model_dir):
            shutil.rmtree(local_model_dir)
            logger.info(f"Purged model files: {local_model_dir}")

        if os.path.exists(local_output_dir):
            shutil.rmtree(local_output_dir)
            logger.info(f"Purged result files: {local_output_dir}")
