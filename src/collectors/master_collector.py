"""
Master metadata collector that aggregates all profiling data.

Combines GGUF metadata, system information, and tool versions into a
comprehensive evaluation parameters dictionary.
"""
import datetime
from typing import Any, Dict, Optional

from .gguf_parser import get_gguf_metadata
from .system_profiler import get_system_metadata
from .tool_profiler import get_tool_metadata
from utils.hashing import calculate_sha256


def collect_evaluation_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect all evaluation parameters and metadata.

    This is the main entry point for metadata collection. It aggregates:
    - Run configuration (ID, timestamp, settings)
    - Model metadata (GGUF headers if applicable)
    - System information (GPU, OS, driver versions)
    - Tool versions (LightEval, PyTorch, etc.)
    - File integrity (SHA256 hash)

    Args:
        config: Configuration dictionary with keys:
            - run_id: Unique identifier for this evaluation run
            - model_path: Path to model file (optional)
            - model_id: Model identifier/name
            - benchmark_category: Category of benchmarks (e.g., 'General')
            - use_case_tags: List of use case tags (e.g., ['chat'])
            - temperature: Sampling temperature (default: 0.0)
            - num_threads: Number of threads for inference
            - random_seed: Random seed for reproducibility
            - batch_size: Batch size for evaluation

    Returns:
        Dictionary containing all collected metadata

    Example:
        >>> config = {
        ...     "run_id": "llama-3-8b_1234567890",
        ...     "model_path": "/tmp/models/llama-3-8b.gguf",
        ...     "model_id": "llama-3-8b",
        ... }
        >>> metadata = collect_evaluation_parameters(config)
        >>> print(metadata.keys())
    """
    # 1. [Manual/Config] - Enforce Reproducibility Defaults
    eval_meta: Dict[str, Any] = {
        "run_id": config.get("run_id"),
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        # CRITICAL FOR REPRODUCIBILITY IN QUANTIZED MODELS:
        "num_threads": config.get("num_threads", 4),  # Must be fixed per run
        "temperature": config.get("temperature", 0.0),
        "random_seed": config.get("random_seed", 42),
        "batch_size": config.get("batch_size", 1),
        "model_id": config.get("model_id", "unknown"),
        "benchmark_category": config.get("benchmark_category", "General"),
        "use_case_tags": config.get("use_case_tags", []),
    }

    # 2. [AUTO-Header] - Model-specific metadata
    model_path = config.get("model_path")
    if model_path:
        gguf_data = get_gguf_metadata(model_path)
        eval_meta.update(gguf_data)

        # SAFETY GATE: If benchmark is 'Agentic' or 'Chat', ensure template exists
        use_case_tags = config.get("use_case_tags", [])
        if "chat" in use_case_tags and not gguf_data.get("has_chat_template"):
            eval_meta["RISK_FLAG"] = "MISSING_CHAT_TEMPLATE"

        # 3. [ISO 27001] Integrity (only hash if file exists)
        file_hash = calculate_sha256(model_path)
        if file_hash:
            eval_meta["file_hash_sha256"] = file_hash

    # 4. [AUTO-Env] & [AUTO-Tool]
    eval_meta.update(get_system_metadata())
    eval_meta.update(get_tool_metadata())

    return eval_meta
