"""
Master metadata collector that aggregates all profiling data.

Combines GGUF metadata, system information, and tool versions into a
comprehensive evaluation parameters dictionary.

UPDATED VERSION: Now includes ALL parameters from register table.
"""
import datetime
from typing import Any, Dict, Optional

from .gguf_parser import get_gguf_metadata
from .system_profiler import get_system_metadata
from .tool_profiler import get_tool_metadata
from utils.hashing import calculate_sha256


def collect_evaluation_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect all evaluation parameters and metadata - COMPLETE VERSION.

    This is the main entry point for metadata collection. It aggregates:
    - Run configuration (ID, timestamp, settings)
    - Model metadata (GGUF headers if applicable)
    - System information (GPU, OS, driver versions)
    - Tool versions (LightEval, PyTorch, etc.)
    - File integrity (SHA256 hash)
    - Business & Research parameters

    Args:
        config: Configuration dictionary with keys:
            - run_id: Unique identifier for this evaluation run
            - model_path: Path to model file (optional)
            - model_id: Model identifier/name
            
            # Inference Parameters
            - temperature: Sampling temperature (default: 0.0)
            - num_threads: Number of threads for inference (default: 4)
            - random_seed: Random seed for reproducibility (default: 42)
            - batch_size: Batch size for evaluation (default: 1)
            - top_p: Nucleus sampling parameter (default: 1.0)
            - top_k: Top-K sampling parameter (default: 50)
            - repetition_penalty: Penalty for repetition (default: 1.0)
            - system_prompt_id: System prompt identifier (default: "default")
            
            # Hardware
            - inference_backend: Backend used (default: "lighteval")
            
            # Business & Metrics
            - benchmark_category: Category of benchmarks (default: 'General')
            - use_case_tags: List of use case tags (default: [])
            - industry_vertical: Industry sector (default: "General")
            - metric_type: Primary metric type (default: "Accuracy")
            - dataset_version: Dataset version (default: "latest")
            
            # Model Identity (Platform API - optional)
            - pruning_method: Pruning technique used (default: "Unknown")
            - sparsity_ratio: Sparsity ratio (default: 0.0)
            - healing_applied: Whether healing was applied (default: False)
            - calibration_dataset: Dataset used for calibration (default: "Unknown")
            - model_release_date: Model release date (default: current date)

    Returns:
        Dictionary containing all collected metadata

    Example:
        >>> config = {
        ...     "run_id": "llama-3-8b_1234567890",
        ...     "model_path": "/tmp/models/llama-3-8b.gguf",
        ...     "model_id": "llama-3-8b",
        ...     "top_p": 0.9,
        ...     "top_k": 40,
        ... }
        >>> metadata = collect_evaluation_parameters(config)
        >>> print(metadata.keys())
    """
    # ========================================================================
    # SECTION A: MODEL IDENTITY & RUN METADATA
    # ========================================================================
    eval_meta: Dict[str, Any] = {
        "run_id": config.get("run_id"),
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "model_id": config.get("model_id", "unknown"),
    }
    
    # Platform API Fields (Placeholders until API is connected)
    # These come from the model platform API when available,
    # otherwise can be manually set in config, or default to "Unknown"
    eval_meta.update({
        "pruning_method": config.get("pruning_method", "Unknown"),
        "sparsity_ratio": config.get("sparsity_ratio", 0.0),
        "healing_applied": config.get("healing_applied", False),
        "calibration_dataset": config.get("calibration_dataset", "Unknown"),
        "model_release_date": config.get(
            "model_release_date", 
            datetime.datetime.utcnow().strftime('%Y-%m-%d')
        ),
    })
    
    # ========================================================================
    # SECTION B: INFERENCE CONFIGURATION
    # ========================================================================
    # CRITICAL FOR REPRODUCIBILITY IN QUANTIZED MODELS
    eval_meta.update({
        # Core inference settings (required for reproducibility)
        "num_threads": config.get("num_threads", 4),  # Must be fixed per run
        "temperature": config.get("temperature", 0.0),
        "random_seed": config.get("random_seed", 42),
        "batch_size": config.get("batch_size", 1),
        
        # Sampling parameters (previously missing)
        "top_p": config.get("top_p", 1.0),
        "top_k": config.get("top_k", 50),
        "repetition_penalty": config.get("repetition_penalty", 1.0),
        "system_prompt_id": config.get("system_prompt_id", "default"),
    })
    
    # ========================================================================
    # SECTION C: HARDWARE CONFIGURATION
    # ========================================================================
    eval_meta["inference_backend"] = config.get("inference_backend", "lighteval")
    
    # ========================================================================
    # SECTION D: BUSINESS & METRICS METADATA
    # ========================================================================
    eval_meta.update({
        "benchmark_category": config.get("benchmark_category", "General"),
        "use_case_tags": config.get("use_case_tags", []),
        "industry_vertical": config.get("industry_vertical", "General"),
        "metric_type": config.get("metric_type", "Accuracy"),
        "dataset_version": config.get("dataset_version", "latest"),
    })

    # ========================================================================
    # SECTION E: AUTO-EXTRACTED MODEL METADATA (GGUF)
    # ========================================================================
    model_path = config.get("model_path")
    if model_path:
        gguf_data = get_gguf_metadata(model_path)
        eval_meta.update(gguf_data)

        # SAFETY GATE: If benchmark is 'Agentic' or 'Chat', ensure template exists
        use_case_tags = config.get("use_case_tags", [])
        if "chat" in use_case_tags and not gguf_data.get("has_chat_template"):
            eval_meta["RISK_FLAG"] = "MISSING_CHAT_TEMPLATE"

        # ========================================================================
        # SECTION F: FILE INTEGRITY (ISO 27001)
        # ========================================================================
        file_hash = calculate_sha256(model_path)
        if file_hash:
            eval_meta["file_hash_sha256"] = file_hash

    # ========================================================================
    # SECTION G: SYSTEM & TOOL METADATA
    # ========================================================================
    eval_meta.update(get_system_metadata())
    eval_meta.update(get_tool_metadata())

    return eval_meta


# ============================================================================
# HELPER FUNCTION: Validate Required Fields
# ============================================================================
def validate_metadata(metadata: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate that all critical fields are present in metadata.
    
    Args:
        metadata: Metadata dictionary to validate
        
    Returns:
        Tuple of (is_valid, list_of_missing_fields)
        
    Example:
        >>> meta = collect_evaluation_parameters(config)
        >>> is_valid, missing = validate_metadata(meta)
        >>> if not is_valid:
        ...     print(f"Missing fields: {missing}")
    """
    required_fields = [
        "run_id",
        "timestamp_utc",
        "model_id",
        "num_threads",
        "temperature",
        "random_seed",
        "batch_size",
    ]
    
    missing_fields = [field for field in required_fields if field not in metadata]
    is_valid = len(missing_fields) == 0
    
    return is_valid, missing_fields


# ============================================================================
# HELPER FUNCTION: Get Parameter Summary
# ============================================================================
def get_parameter_summary(metadata: Dict[str, Any]) -> Dict[str, int]:
    """
    Get a summary count of parameters by category.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Dictionary with counts per category
        
    Example:
        >>> meta = collect_evaluation_parameters(config)
        >>> summary = get_parameter_summary(meta)
        >>> print(f"Total parameters collected: {sum(summary.values())}")
    """
    categories = {
        "model_identity": ["model_id", "pruning_method", "sparsity_ratio", "healing_applied"],
        "inference": ["temperature", "top_p", "top_k", "repetition_penalty"],
        "hardware": ["inference_backend", "gpu_name", "driver_version"],
        "business": ["benchmark_category", "industry_vertical", "metric_type"],
        "gguf": ["base_model_family", "param_count_billions", "max_context_window"],
        "system": ["os_name", "python_version"],
        "tools": ["lighteval_version", "torch_version"],
    }
    
    summary = {}
    for category, fields in categories.items():
        count = sum(1 for field in fields if field in metadata)
        summary[category] = count
    
    return summary