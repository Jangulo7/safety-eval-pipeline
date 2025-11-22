import os
import shutil
import subprocess
import logging
import time
import json
import glob
import boto3
from celery import Celery
import config
from notifications import send_alert

app = Celery('ai_evaluator', broker='redis://redis:6379/0')

# Configure logging specific to worker
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')

def download_s3_folder(bucket: str, s3_folder_prefix: str, local_dir: str):
    """
    Recursively downloads an S3 'folder' to a local directory.
    Essential for HF models that rely on config.json + tokenizer files.
    """
    logger.info(f"Downloading folder {s3_folder_prefix} to {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=s3_folder_prefix)
    
    files_downloaded = 0
    for page in pages:
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            key = obj['Key']
            # Skip the folder key itself if it exists
            if key.endswith('/'):
                continue
                
            # Calculate relative path
            # s3: models/llama/config.json -> local: /tmp/llama/config.json
            rel_path = key[len(s3_folder_prefix):]
            if rel_path.startswith('/'):
                rel_path = rel_path[1:]
                
            local_file_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            s3.download_file(bucket, key, local_file_path)
            files_downloaded += 1
            
    if files_downloaded == 0:
        raise ValueError(f"No files found at S3 prefix: {s3_folder_prefix}")
    
    logger.info(f"Downloaded {files_downloaded} files.")

def detect_model_type_and_path(local_dir: str):
    """
    Determines if we are dealing with a GGUF file or a HF Directory.
    Returns: (final_path_arg, backend_type)
    """
    # Check for GGUF files
    gguf_files = glob.glob(os.path.join(local_dir, "*.gguf"))
    
    if gguf_files:
        # GGUF Strategy: Point directly to the .gguf file
        # We assume the largest GGUF file is the model if multiple exist
        main_gguf = max(gguf_files, key=os.path.getsize)
        logger.info(f"Detected GGUF format: {main_gguf}")
        return main_gguf, "gguf"
    
    # Standard HF Strategy: Point to the directory
    # Check for config.json to confirm validity
    if os.path.exists(os.path.join(local_dir, "config.json")):
        logger.info("Detected Standard HuggingFace directory format.")
        return local_dir, "hf"
        
    raise ValueError(f"Could not identify model type in {local_dir} (No .gguf or config.json)")

def run_lighteval_subprocess(model_path: str, output_dir: str, model_type: str):
    """
    Runs LightEval. Adapts arguments based on model type.
    """
    tasks_str = ",".join(config.BENCHMARK_SUITE)
    
    cmd = ["lighteval", "accelerate"]
    
    # Argument construction based on type
    if model_type == "gguf":
        # Note: LightEval GGUF support varies by backend. 
        # Assuming we use a backend that supports GGUF loading or the 'nanotron' adaptation.
        # Alternatively, if using `text-generation-inference` backend:
        # This example assumes a backend that accepts the file path directly.
        model_args = f"pretrained={model_path},trust_remote_code={config.TRUST_REMOTE_CODE}"
    else:
        # Standard HF Directory
        model_args = f"pretrained={model_path},trust_remote_code={config.TRUST_REMOTE_CODE}"

    cmd.extend([
        "--model_args", model_args,
        "--tasks", tasks_str,
        "--output_dir", output_dir,
        "--save_details"
    ])
    
    logger.info(f"Executing LightEval: {' '.join(cmd)}")
    
    # Capture output to log it (Fixing Minor Error #7)
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": config.GPU_IDS}
    )
    
    # Log the output for debugging
    if result.stdout:
        logger.info(f"LightEval STDOUT (Truncated): {result.stdout[:1000]}")
    
    if result.returncode != 0:
        logger.error(f"LightEval STDERR: {result.stderr}")
        raise RuntimeError(f"Evaluation process failed with code {result.returncode}")
        
    return result.stdout

@app.task(bind=True, max_retries=1)
def evaluate_model_task(self, s3_folder_prefix: str):
    start_time = time.time()
    
    # Fix Logic Error #3: Name extraction
    # "models/llama-3/" -> "llama-3"
    model_name = s3_folder_prefix.strip('/').split('/')[-1]
    
    run_id = f"{model_name}_{int(start_time)}"
    local_model_dir = os.path.join(config.TEMP_MODEL_DIR, model_name)
    local_output_dir = os.path.join(config.TEMP_MODEL_DIR, "results", run_id)

    try:
        # 1. Download Full Directory
        download_s3_folder(config.S3_BUCKET_NAME, s3_folder_prefix, local_model_dir)

        # 2. Detect Type (GGUF vs HF)
        target_path, model_type = detect_model_type_and_path(local_model_dir)

        # 3. Evaluate
        run_lighteval_subprocess(target_path, local_output_dir, model_type)

        # 4. Upload Results (Fix Error #4: Existence Check)
        if not os.path.exists(local_output_dir):
            logger.warning(f"Output directory {local_output_dir} was not created by LightEval.")
            send_alert(model_name, "FAILURE", error_msg="Output directory missing")
            return

        files_uploaded = 0
        for root, _, files in os.walk(local_output_dir):
            for file in files:
                if file.endswith(".json"):
                    local_file = os.path.join(root, file)
                    s3_dest = f"{config.S3_RESULTS_PREFIX}{run_id}/{file}"
                    s3.upload_file(local_file, config.S3_BUCKET_NAME, s3_dest)
                    files_uploaded += 1
        
        if files_uploaded > 0:
            elapsed = f"{(time.time() - start_time)/60:.1f}m"
            send_alert(model_name, "SUCCESS", run_time=elapsed)
        else:
            logger.warning("No JSON results found to upload.")
            send_alert(model_name, "FAILURE", error_msg="No JSON results generated")

    except Exception as e:
        logger.error(f"Task failed for {model_name}: {str(e)}")
        send_alert(model_name, "FAILURE", error_msg=str(e))
        raise self.retry(exc=e)

    finally:
        # 5. Aggressive Cleanup (Fetch-Eval-Purge)
        if os.path.exists(local_model_dir):
            shutil.rmtree(local_model_dir)
            logger.info(f"Purged model files: {local_model_dir}")
        
        if os.path.exists(local_output_dir):
            shutil.rmtree(local_output_dir)
            logger.info(f"Purged result files: {local_output_dir}")