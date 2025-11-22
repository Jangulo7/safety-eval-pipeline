import os
import shutil
import subprocess
import logging
import time
import boto3
from celery import Celery
import config
from notifications import send_alert

app = Celery('ai_evaluator', broker='redis://redis:6379/0')
logger = logging.getLogger(__name__)
s3 = boto3.client('s3')

def run_lighteval_subprocess(model_path, output_dir):
    """
    Executes LightEval in a separate process to guarantee VRAM reclamation.
    """
    tasks_str = ",".join(config.BENCHMARK_SUITE)
    
    # Command optimized for Quantized/Base model evaluation
    cmd = [
        "lighteval", "accelerate",
        f"--model_args", f"pretrained={model_path},trust_remote_code={config.TRUST_REMOTE_CODE}",
        f"--tasks", tasks_str,
        f"--output_dir", output_dir,
        "--save_details"
    ]
    
    # Note: If evaluating Instruction models, append: ["--use_chat_template"]
    
    logger.info(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": config.GPU_IDS}
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Stderr: {result.stderr}")
    return result.stdout

@app.task(bind=True, max_retries=1)
def evaluate_model_task(self, s3_key):
    start_time = time.time()
    local_path = None
    local_out = None
    model_name = os.path.basename(s3_key)

    try:
        # 1. Setup paths
        run_id = f"{model_name}_{int(start_time)}"
        local_path = os.path.join(config.TEMP_MODEL_DIR, model_name)
        local_out = os.path.join(config.TEMP_MODEL_DIR, "results", run_id)
        
        # 2. Download
        logger.info(f"Downloading {s3_key}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(config.S3_BUCKET_NAME, s3_key, local_path)

        # 3. Evaluate
        run_lighteval_subprocess(local_path, local_out)

        # 4. Upload Results
        for root, _, files in os.walk(local_out):
            for f in files:
                if f.endswith(".json"):
                    src = os.path.join(root, f)
                    dst = f"{config.S3_RESULTS_PREFIX}{run_id}/{f}"
                    s3.upload_file(src, config.S3_BUCKET_NAME, dst)

        # 5. Success Alert
        elapsed = f"{(time.time() - start_time)/60:.1f}m"
        send_alert(model_name, "SUCCESS", run_time=elapsed)

    except Exception as e:
        logger.error(f"Failed {model_name}: {e}")
        send_alert(model_name, "FAILURE", error_msg=str(e))
        raise self.retry(exc=e)

    finally:
        # 6. Aggressive Cleanup
        if local_path and os.path.exists(local_path):
            # Handle both file and folder cases
            if os.path.isdir(local_path): shutil.rmtree(local_path)
            else: os.remove(local_path)
        
        if local_out and os.path.exists(local_out):
            shutil.rmtree(local_out)