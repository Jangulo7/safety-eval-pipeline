from .gguf_parser import get_gguf_metadata
from .system_profiler import get_system_metadata
from .tool_profiler import get_tool_metadata
from ..utils.hashing import calculate_sha256
import datetime

def collect_evaluation_parameters(config: dict):
    
    # 1. [Manual/Config] - Enforce Reproducibility Defaults
    eval_meta = {
        "run_id": config.get("run_id"),
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        # CRITICAL FOR REPRODUCIBILITY IN QUANTIZED MODELS:
        "num_threads": config.get("num_threads", 4),  # Must be fixed per run
        "temperature": config.get("temperature", 0.0),
        "random_seed": config.get("random_seed", 42),
        "batch_size": config.get("batch_size", 1),
    }

    # 2. [AUTO-Header]
    if config.get("model_path"):
        gguf_data = get_gguf_metadata(config["model_path"])
        eval_meta.update(gguf_data)
        
        # SAFETY GATE: If benchmark is 'Agentic' or 'Chat', ensure template exists
        if "chat" in config.get("use_case_tags", []) and not gguf_data.get("has_chat_template"):
            eval_meta["RISK_FLAG"] = "MISSING_CHAT_TEMPLATE"

        # 3. [ISO 27001] Integrity (only hash if file exists)
        eval_meta["file_hash_sha256"] = calculate_sha256(config["model_path"])

    # 4. [AUTO-Env] & [AUTO-Tool]
    eval_meta.update(get_system_metadata())
    eval_meta.update(get_tool_metadata())

    return eval_meta