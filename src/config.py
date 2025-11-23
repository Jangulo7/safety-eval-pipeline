import os

# S3 Settings
S3_BUCKET_NAME = os.getenv("S3_MODEL_BUCKET", "acme-llm-models")
S3_RESULTS_PREFIX = "benchmarks/artificial_analysis/"

# Local Storage
TEMP_MODEL_DIR = "/tmp/eval_models"
os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

# GPU Settings
GPU_IDS = "0"

# --- BENCHMARK SUITE ---
# Exact task strings for reproducibility (Leaderboard V1/Standard definitions)
# Format: suite|task|few_shot|truncate_few_shot
BENCHMARK_SUITE = [
    # 1. MMLU (General Knowledge) - 5-shot is standard
    "leaderboard|mmlu:major|5|0",
    
    # 2. GSM8K (Math Reasoning) - 5-shot is standard
    "leaderboard|gsm8k|5|0",
    
    # 3. HumanEval (Coding) - 0-shot is standard
    "leaderboard|humaneval|0|0",
    
    # 4. HellaSwag (Common Sense/Reasoning) - 10-shot is standard
    "leaderboard|hellaswag|10|0",
    
    # 5. ARC Challenge (Reasoning) - 25-shot is standard
    "leaderboard|arc:challenge|25|0"
]

# Flags
TRUST_REMOTE_CODE = False
USE_CHAT_TEMPLATE = False # Set True if evaluating Instruct models as Chatbots