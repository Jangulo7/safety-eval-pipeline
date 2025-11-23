import platform
import pynvml

def get_system_metadata():
    # OS Info
    meta = {
        "os_version": platform.platform(),
        "python_version": platform.python_version(),
    }

    # GPU Info (Crucial for H200/L40S distinction)
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assuming homogeneous GPUs
        
        meta["gpu_count"] = device_count
        meta["gpu_model"] = pynvml.nvmlDeviceGetName(handle) # e.g. 'NVIDIA H200'
        meta["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
        
        # Memory Info for Extrapolation
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        meta["gpu_vram_total_gb"] = mem_info.total / (1024**3)
        
        pynvml.nvmlShutdown()
    except Exception as e:
        meta["gpu_error"] = str(e)
        meta["gpu_model"] = "CPU_ONLY"

    return meta