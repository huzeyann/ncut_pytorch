import torch


def auto_device(existing_device="", user_input_device="auto"):
    """
    Automatically select the best available device or use user-specified device.

    Args:
        existing_device: The device where the features are currently located
        user_input_device: User-specified device or 'auto' for automatic selection

    Returns:
        str: Selected device as a string
    """
    # If user specified a device and it's not 'auto', try to use it
    if user_input_device is not None and str(user_input_device) != "auto":
        try:
            torch.device(str(user_input_device))
            return str(user_input_device)
        except RuntimeError:
            raise ValueError(f"Invalid device: {user_input_device}")

    def prioritize_existing_device(existing_device, new_device):
        existing_device = str(existing_device)
        if new_device in existing_device:
            return existing_device
        return new_device

    # Check for CUDA (NVIDIA GPUs)
    is_cuda_available = torch.cuda.is_available()
    if is_cuda_available:
        return prioritize_existing_device(existing_device, "cuda")

    # Check for MPS (Apple Silicon GPUs)
    try:
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            return prioritize_existing_device(existing_device, "mps")
    except (ImportError, AttributeError):
        pass

    # Check for TPU (Google TPUs)
    try:
        import torch_xla.core.xla_model as xm
        return prioritize_existing_device(existing_device, xm.xla_device())
    except (ImportError, AttributeError):
        pass

    # Check for AMD GPUs
    try:
        if hasattr(torch, 'hip') and torch.hip.is_available():
            return prioritize_existing_device(existing_device, "hip")
    except (ImportError, AttributeError):
        pass

    # Check for Intel GPUs
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return prioritize_existing_device(existing_device, "xpu")
    except (ImportError, AttributeError):
        pass

    # Fallback to CPU
    return "cpu"
