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

### Create the environment config (DO NOT COMMIT THIS FILE)
touch .env

2. Environment VariablesPopulate .env with your credentials:

### AWS Credentials (S3 Access)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1
S3_MODEL_BUCKET=acme-llm-quantized-models

### Alerting (Optional)
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/T000/B000/XXXX

### Create the environment config (DO NOT COMMIT THIS FILE)
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

### Enter the scheduler container
docker-compose exec scheduler python

### Inside Python shell:
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


# Parameter Register for Automated Evaluation

I've updated the parameter tables, automation keys, and code accordingly. This ensures full reproducibility, compliance, and utility for multi-stakeholder analysis of quantized models.

## Automation Keys (Updated)

- **[AUTO-Header]**: Extracted automatically from GGUF/Model file headers.
- **[AUTO-Env]**: Scraped from the runtime environment (Python, OS, GPU drivers).
- **[AUTO-Plat]**: Pulled via API from your Model/Quantization Platform (with fallback to header/manual if API unavailable).
- **[AUTO-Tool]**: Scraped from the evaluation tool itself (e.g., git commit, package versions). *New*
- **[Manual/Config]**: Defined in the job config (YAML/JSON).
- **[Manual/Default]**: Fallback values hardcoded in code if missing from config.

## A. Model Identity & Quantization (Crucial for Model Team)

**Justification**: Unchanged—essential for debugging quantization impacts like accuracy drops. Added group_size (common in schemes like Q4_K_M/GPTQ) and model_release_date (for daily quantized releases).

| Parameter            | Description                                      | Needed By                  | Automation                  |
|----------------------|--------------------------------------------------|----------------------------|-----------------------------|
| model_id             | Unique ID (e.g., llama-3.1-8b-q4_k_m)            | All                        | [Manual/Config]             |
| base_model_family    | e.g., Llama, Mistral, Qwen                       | Research                   | [AUTO-Header]               |
| param_count_billions | e.g., 8, 70, 405                                 | Pre-Sales (Sizing)         | [AUTO-Header]               |
| quant_scheme         | e.g., Q4_K_M, IQ2_XXS, GPTQ                      | Model Team                 | [AUTO-Header]               |
| bits_per_weight      | Average bits (e.g., 4.65)                        | Research                   | [AUTO-Header]               |
| group_size           | Grouping for quantization (e.g., 128 for GPTQ)   | Model Team                 | [AUTO-Header or AUTO-Plat]  |
| pruning_method       | e.g., Wanda, Magnitude, SparseGPT                | Model Team                 | [AUTO-Plat]                 |
| sparsity_ratio       | % of zeros (e.g., 0.3 for 30%)                   | Model Team                 | [AUTO-Plat]                 |
| healing_applied      | Boolean (Was retraining applied?)                | Model Team                 | [AUTO-Plat]                 |
| calibration_dataset  | Dataset used for quant calibration               | Research (Bias check)      | [AUTO-Plat]                 |
| model_release_date   | Date the quantized model was released (e.g., 2025-11-22) | All (Daily tracking)       | [AUTO-Plat or Manual/Config]|
| file_hash_sha256     | Checksum of the .gguf file                       | ISO 27001 (Integrity)      | [AUTO-Env]                  |

## B. Inference Configuration (Crucial for Research)

**Justification**: Unchanged—key for reproducing scores. Added random_seed (even at temp 0.0, affects sampling) and repetition_penalty (common in benchmarks to reduce verbosity).

| Parameter           | Description                                      | Needed By                  | Automation                             |
|---------------------|--------------------------------------------------|----------------------------|----------------------------------------|
| temperature         | Randomness (usually 0.0 for bench)               | Research                   | [Manual/Default: 0.0]                  |
| top_p / top_k       | Nucleus sampling params                          | Research                   | [Manual/Default: top_p=1.0, top_k=50]  |
| random_seed         | Seed for reproducibility                         | Research                   | [Manual/Default: 42]                   |
| repetition_penalty  | Penalty for repeated tokens (e.g., 1.1)          | Research                   | [Manual/Default: 1.0]                  |
| max_context_window  | Model's limit (e.g., 128k)                       | Pre-Sales (RAG)            | [AUTO-Header]                          |
| batch_size          | Concurrency level (e.g., 1, 8, 32)               | Pre-Sales (Throughput)     | [Manual/Config]                        |
| rope_scaling        | Scaling factor for long context                  | Research                   | [AUTO-Header]                          |
| gpu_split_strategy  | How layers are split across GPUs                 | Ops/Eng                    | [AUTO-Env]                             |
| system_prompt_id    | ID of the system prompt used                     | Research (Prompt Eng)      | [Manual/Config]                        |

## C. Hardware & Environment (Crucial for Pre-Sales & ISO)

**Justification**: Unchanged—vital for performance promises. Added cpu_model and system_ram_gb (for CPU fallbacks or mixed workloads), and handled heterogeneous GPUs.

| Parameter              | Description                                      | Needed By                  | Automation                  |
|------------------------|--------------------------------------------------|----------------------------|-----------------------------|
| gpu_model              | List if heterogeneous (e.g., ['H200 SXM', 'A100-80GB']) | Pre-Sales                  | [AUTO-Env] (nvidia-smi)     |
| gpu_count              | Number of GPUs used                              | Sales (Cost calc)          | [AUTO-Env]                  |
| cpu_model              | CPU details (e.g., Intel Xeon)                   | Ops (Fallback)             | [AUTO-Env]                  |
| system_ram_gb          | Total system RAM                                 | Ops                        | [AUTO-Env]                  |
| driver_version         | CUDA/Driver version                              | Ops (Debugging)            | [AUTO-Env]                  |
| inference_backend      | llama.cpp, vLLM, TGI                             | Research                   | [Manual/Config]             |
| quant_backend_version  | Version of the quant runtime                     | Model Team                 | [AUTO-Env]                  |

## D. Business & Metrics Classification (Crucial for Reporting)

**Justification**: Unchanged—good for filtering. Added use_case_tags (e.g., ["chat", "code-gen"]) for finer granularity.

| Parameter            | Description                                      | Needed By                  | Automation                  |
|----------------------|--------------------------------------------------|----------------------------|-----------------------------|
| benchmark_category   | General, RAG, Agentic, Coding                    | Sales (Grouping)           | [Manual/Config]             |
| industry_vertical    | Medical, Legal, Finance                          | Sales (Pitching)           | [Manual/Config]             |
| use_case_tags        | Tags like "chat", "summarization"                 | Sales                      | [Manual/Config]             |
| metric_type          | Accuracy, Hallucination, Bias                    | ISO 27001 (Safety)         | [Manual/Config]             |
| dataset_version      | Version of MMLU/DeepEval dataset                 | Research                   | [Manual/Config]             |

## E. Tool & Pipeline Metadata (New Category: Crucial for Reproducibility & Auditing)

**Justification**: The evaluation tool itself evolves; logging its state ensures benchmarks are traceable over time. This is key for overnight pipelines where code might update daily.

| Parameter                | Description                                      | Needed By                  | Automation                             |
|--------------------------|--------------------------------------------------|----------------------------|----------------------------------------|
| eval_tool_version        | Version of the benchmarking framework (e.g., DeepEval 1.2.3) | Research                   | [AUTO-Tool]                            |
| eval_tool_commit_hash    | Git commit of the evaluation code                | ISO 27001 (Audit)          | [AUTO-Tool]                            |
| python_package_versions  | Key deps (e.g., {'torch': '2.1.0'})              | Ops (Debugging)            | [AUTO-Tool]                            |
| run_timestamp            | UTC ISO time of run start                        | All                        | [AUTO-Env]                             |
| run_duration_seconds     | Total runtime                                    | Pre-Sales (Perf)           | [AUTO-Env] (Calculated post-run)       |


