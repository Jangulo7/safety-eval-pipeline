import gguf
import os
import logging

def get_gguf_metadata(file_path: str):
    """
    Robustly extracts metadata, handling inconsistent naming conventions (Llama vs Qwen).
    """
    try:
        reader = gguf.GGUFReader(file_path)
        # Convert all keys to a standard dict for easier searching
        raw_meta = {field.key: field.parts[-1] for field in reader.fields.values() if field.parts}
        
        # 1. Robust Key Search (Suffix Method)
        # Finds 'llama.context_length' OR 'qwen.context_length' OR 'context_length'
        def find_val(suffix, default):
            for key, val in raw_meta.items():
                if key.endswith(suffix):
                    return val[0] if isinstance(val, list) else val
            return default

        # 2. Architecture & Params
        arch = raw_meta.get("general.architecture", "Unknown")
        param_count = find_val(".parameter_count", 0) / 1e9
        context_length = find_val(".context_length", 4096)
        
        # 3. Critical: Chat Template Validation
        # ISO 27001 Safety: Ensures we aren't testing a broken model interface
        chat_template = find_val(".chat_template", None)
        has_chat_template = bool(chat_template)
        
        # 4. Quantization Details
        quant_version = raw_meta.get("general.quantization_version", "Unknown")
        # specific tensor quant type usually found in 'general.file_type' or inferred
        file_type_code = raw_meta.get("general.file_type", [0])[0]
        
        # 5. Bits Per Weight (Calculated)
        file_size_bytes = os.path.getsize(file_path)
        bits_per_weight = (file_size_bytes * 8) / (param_count * 1e9) if param_count > 0 else 0

        return {
            "base_model_family": arch,
            "param_count_billions": round(param_count, 2),
            "max_context_window": int(context_length),
            "has_chat_template": has_chat_template, # FAIL pipeline if False for Chat Benchmarks
            "chat_template_preview": str(chat_template)[:50] + "..." if chat_template else "None",
            "quant_file_type_id": file_type_code, 
            "bits_per_weight": round(bits_per_weight, 2),
            "file_size_gb": round(file_size_bytes / (1024**3), 2)
        }
    except Exception as e:
        logging.error(f"GGUF Extraction Failed: {e}")
        return {"error": str(e), "valid_gguf": False}