# EvalPipelineAA
A fully automated pipeline designed to benchmark new models uploaded to our S3 artifact store.

This repository hosts the Overnight Evaluation Queue, a fully automated pipeline designed to benchmark new quantized Large Language Models (LLMs) uploaded to our S3 artifact store. It executes Artificial Analysis compliant benchmarks (MMLU, GSM8K, HumanEval) to assess the degradation of Quality vs. Quantization Efficiency.

Overview
The pipeline operates on a "Just-In-Time" architecture to maximize resource efficiency. It wakes up nightly, scans for fresh models, isolates them in ephemeral GPU processes, and publishes the results to a visual dashboard.

Key Features:

Memory Safe: Uses subprocess isolation to prevent CUDA VRAM fragmentation and OOM crashes.

Storage Optimized: Implements a Fetch-Eval-Purge cycle to keep disk usage low.

Leaderboard Compliant: Uses lighteval with leaderboard task definitions to ensure metrics match public Open LLM Leaderboards.

Passive Observability: Silent success/failure alerts sent directly to Slack/Teams.

Architecture
The system is containerized using Docker Compose and consists of four primary services:

Component

1. Watcher (scheduler): A lightweight Cron script that scans S3 hourly for new .gguf or .safetensors artifacts.
2. Broker (redis): A Redis instance that manages the FIFO job queue.
3. Worker (worker): "The heavy lifter. It claims the GPU, downloads the model, runs the eval, and uploads JSON results."
4. Dashboard (dashboard): "A Streamlit app that visualizes ""Quality vs. Quantization"" trade-offs."

Benchmark Coverage (Artificial Analysis)
We adhere to the Artificial Analysis Intelligence Index methodology to ensure our internal metrics map to external standards.
It supports the following bechmarks: MMLU (5-shot), GSM8K (5-shot), HumanEval (0-shot), HellaSwag (10-shot).

Getting Started
Prerequisites
Docker & Docker Compose

NVIDIA Container Toolkit (The worker requires access to the host GPU).

An AWS S3 Bucket containing your models.

1. Clone & Configure
Clone the repository and create your environment file.

git clone https://github.com/your-org/llm-nightly-eval.git
cd llm-nightly-eval

# Create the environment config (DO NOT COMMIT THIS FILE)
touch .env

2. Environment VariablesPopulate .env with your credentials:

# AWS Credentials (S3 Access)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1
S3_MODEL_BUCKET=acme-llm-quantized-models

# Alerting (Optional)
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/T000/B000/XXXX

# Create the environment config (DO NOT COMMIT THIS FILE)
touch .env

3. Launch the StackDeploy the pipeline in detached mode.

docker-compose up --build -d

Usage & Monitoring
Viewing the Dashboard
Access the metrics dashboard at: http://localhost:8501

The dashboard provides:

Scatter Plot: Quantization Level (x-axis) vs. Quality Score (y-axis).

Leaderboard Table: Filterable list of all evaluated models.

Drill-down: Click any model to see raw JSON outputs.

Manual Trigger
If you need to evaluate a specific model immediately (bypassing the cron schedule):

# Enter the scheduler container
docker-compose exec scheduler python

# Inside Python shell:
from tasks import evaluate_model_task
evaluate_model_task.delay("path/to/model_folder_in_s3")

Logs & Debugging
Check the worker logs to see real-time evaluation progress:

docker-compose logs -f worker

Development & Optimization
Code Standards
This project strictly adheres to PEP8. Run the linter before pushing:

flake8 src/ --exclude=__init__.py

Profiling
Performance is critical. We use cProfile to audit the evaluation loop.

Profile dumps are saved to /tmp/profile_*.stats in the worker container.

To analyze a dump: snakeviz profile_modelname.stats

Adding New Benchmarks
To add a new benchmark (e.g., ARC Challenge), modify src/config.py:

BENCHMARK_SUITE = [
    # ... existing benchmarks ...
    "leaderboard|arc:challenge|25|0"  # Added ARC 25-shot
]





