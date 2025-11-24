"""
GGUF metadata parser - ENHANCED VERSION.

Robustly extracts metadata from GGUF files, handling inconsistent naming
conventions across different model architectures (Llama, Qwen, etc.).

UPDATED: Now includes rope_scaling and group_size extraction.
"""

import gguf
import os
import logging
from typing import Dict, Any


def get_gguf_metadata(file_path: str) -> Dict[str, Any]:
    """
    Robustly extracts metadata from GGUF files.

    Handles inconsistent naming conventions (Llama vs Qwen) and extracts:
    - Architecture and parameters
    - Context window and quantization details
    - RoPE scaling configuration (for long-context research)
    - Group size (for quantization analysis)
    - Chat template validation

    Args:
        file_path: Path to GGUF model file

    Returns:
        Dictionary containing extracted metadata, or error dict if extraction fails

    Example:
        >>> metadata = get_gguf_metadata("/path/to/model.gguf")
        >>> print(f"Model: {metadata['base_model_family']}")
        >>> print(f"Params: {metadata['param_count_billions']}B")
    """
    try:
        reader = gguf.GGUFReader(file_path)
        # Convert all keys to a standard dict for easier searching
        raw_meta: Dict[str, Any] = {}
        for field in reader.fields.values():
            if hasattr(field, "parts") and field.parts:
                # Access key through the field's name attribute
                key = str(list(reader.fields.keys())[list(reader.fields.values()).index(field)])
                raw_meta[key] = field.parts[-1]

        # ====================================================================
        # HELPER: Robust Key Search (Suffix Method)
        # ====================================================================
        # Finds 'llama.context_length' OR 'qwen.context_length' OR 'context_length'
        def find_val(suffix: str, default: Any = None) -> Any:
            """
            Find value by key suffix, handling architecture-specific prefixes.

            Args:
                suffix: Key suffix to search for (e.g., '.context_length')
                default: Default value if not found

            Returns:
                Found value or default
            """
            for key, val in raw_meta.items():
                if key.endswith(suffix):
                    # Handle list values (some GGUF fields return lists)
                    return val[0] if isinstance(val, list) else val
            return default

        # ====================================================================
        # SECTION 1: ARCHITECTURE & BASIC PARAMETERS
        # ====================================================================
        arch = raw_meta.get("general.architecture", "Unknown")
        param_count = find_val(".parameter_count", 0) / 1e9
        context_length = find_val(".context_length", 4096)

        # ====================================================================
        # SECTION 2: ROPE SCALING (Critical for Long Context Research)
        # ====================================================================
        # RoPE (Rotary Position Embedding) configuration affects long-context
        # performance. Key parameters:
        # - rope.freq_base: Base frequency for position encoding
        # - rope.scale_linear: Linear scaling factor
        # - rope.dimension_count: Number of dimensions

        rope_freq_base = find_val(".rope.freq_base")
        rope_scale_linear = find_val(".rope.scale_linear")
        rope_dimension_count = find_val(".rope.dimension_count")

        # Calculate effective RoPE scaling factor
        # If rope_scale_linear exists, that's the scaling
        # Otherwise, if freq_base is non-standard, that indicates scaling
        if rope_scale_linear is not None:
            rope_scaling_factor = float(rope_scale_linear)
        elif rope_freq_base is not None and rope_freq_base != 10000:
            # Standard RoPE uses freq_base=10000
            # If different, calculate implied scaling
            rope_scaling_factor = float(rope_freq_base) / 10000.0
        else:
            rope_scaling_factor = 1.0  # No scaling

        rope_metadata = {
            "rope_freq_base": rope_freq_base if rope_freq_base else 10000,
            "rope_scaling_factor": round(rope_scaling_factor, 2),
            "rope_dimension_count": rope_dimension_count,
        }

        # ====================================================================
        # SECTION 3: GROUP SIZE (Critical for Quantization Analysis)
        # ====================================================================
        # Group size is tricky in GGUF - it's often implicit in the file type
        # or stored in quantization metadata. Common patterns:
        # - general.quantization_version
        # - Tensor-specific metadata (requires deeper inspection)
        # - File type code implies group size

        quant_version = raw_meta.get("general.quantization_version", "Unknown")
        file_type_code_raw = raw_meta.get("general.file_type", [0])
        file_type_code: int = (
            file_type_code_raw[0]
            if isinstance(file_type_code_raw, list)
            else int(file_type_code_raw)
        )

        # Infer group size from file type code
        # Common GGUF file types and their group sizes:
        # Type 2-5: Various quantizations with different group sizes
        # This is a heuristic - ideally would inspect tensor metadata
        group_size_map = {
            2: 32,  # Q4_0
            3: 32,  # Q4_1
            7: 128,  # Q5_0
            8: 128,  # Q5_1
            10: 32,  # Q8_0
        }

        group_size = group_size_map.get(file_type_code, None)

        # Try to find explicit group_size in metadata
        explicit_group_size = find_val(".group_size")
        if explicit_group_size:
            group_size = int(explicit_group_size)

        # ====================================================================
        # SECTION 4: CHAT TEMPLATE VALIDATION
        # ====================================================================
        # ISO 27001 Safety: Ensures we aren't testing a broken model interface
        chat_template = find_val(".chat_template", None)
        has_chat_template = bool(chat_template)

        # ====================================================================
        # SECTION 5: QUANTIZATION DETAILS
        # ====================================================================
        # Calculate bits per weight from file size
        file_size_bytes = os.path.getsize(file_path)
        bits_per_weight = (file_size_bytes * 8) / (param_count * 1e9) if param_count > 0 else 0

        # ====================================================================
        # RETURN: COMPLETE METADATA DICTIONARY
        # ====================================================================
        return {
            # Architecture & Capacity
            "base_model_family": arch,
            "param_count_billions": round(param_count, 2),
            "max_context_window": int(context_length),
            # Chat Capability
            "has_chat_template": has_chat_template,
            "chat_template_preview": (str(chat_template)[:50] + "..." if chat_template else "None"),
            # Quantization Details
            "quant_file_type_id": file_type_code,
            "quant_version": str(quant_version),
            "bits_per_weight": round(bits_per_weight, 2),
            "group_size": group_size if group_size else "Unknown",
            # RoPE Configuration (NEW)
            **rope_metadata,
            # File Info
            "file_size_gb": round(file_size_bytes / (1024**3), 2),
            # Validation
            "valid_gguf": True,
        }

    except FileNotFoundError:
        logging.error(f"GGUF file not found: {file_path}")
        return {"error": f"File not found: {file_path}", "valid_gguf": False}
    except Exception as e:
        logging.error(f"GGUF Extraction Failed for {file_path}: {e}")
        return {"error": str(e), "valid_gguf": False}


def get_tensor_info(file_path: str) -> Dict[str, Any]:
    """
    Extract detailed tensor information from GGUF file.

    This is more expensive than get_gguf_metadata() as it inspects
    individual tensors, but provides more accurate group_size and
    quantization details.

    Args:
        file_path: Path to GGUF model file

    Returns:
        Dictionary with tensor-level metadata

    Example:
        >>> tensor_info = get_tensor_info("/path/to/model.gguf")
        >>> print(f"Tensors: {tensor_info['num_tensors']}")
    """
    try:
        reader = gguf.GGUFReader(file_path)

        # Get tensor information
        tensors = reader.tensors
        num_tensors = len(tensors)

        # Analyze quantization types
        quant_types: Dict[str, int] = {}
        for tensor in tensors:
            tensor_type = str(tensor.tensor_type)
            quant_types[tensor_type] = quant_types.get(tensor_type, 0) + 1

        return {
            "num_tensors": num_tensors,
            "quantization_types": quant_types,
            "dominant_quant_type": (
                max(quant_types, key=lambda k: quant_types[k]) if quant_types else "Unknown"
            ),
        }

    except Exception as e:
        logging.error(f"Tensor inspection failed for {file_path}: {e}")
        return {
            "error": str(e),
            "num_tensors": 0,
        }


def validate_gguf_file(file_path: str) -> tuple[bool, str]:
    """
    Quick validation that a file is a valid GGUF file.

    Args:
        file_path: Path to file to validate

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> is_valid, msg = validate_gguf_file("/path/to/model.gguf")
        >>> if not is_valid:
        ...     print(f"Invalid GGUF: {msg}")
    """
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"

    if not file_path.endswith(".gguf"):
        return False, "File does not have .gguf extension"

    try:
        # Try to open and read basic metadata
        reader = gguf.GGUFReader(file_path)
        arch = None
        for key, field in reader.fields.items():
            if key == "general.architecture" and hasattr(field, "parts") and field.parts:
                arch = field.parts[-1]
                break

        if not arch:
            return False, "No architecture found in GGUF metadata"

        return True, "Valid GGUF file"

    except Exception as e:
        return False, f"Failed to read GGUF: {str(e)}"
