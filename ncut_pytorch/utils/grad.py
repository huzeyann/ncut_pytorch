"""Utilities for managing PyTorch gradient computation state."""

import torch
from contextlib import contextmanager


@contextmanager
def set_grad_enabled(enabled: bool):
    """Context manager to temporarily set gradient computation mode.
    
    This context manager allows you to control gradient computation for a block
    of code, and automatically restores the previous gradient state when exiting
    the context.
    
    Args:
        enabled (bool): If True, enables gradient tracking within the context.
                        If False, disables gradient tracking within the context.
    
    Yields:
        None
        
    Examples:
        >>> import torch
        >>> from ncut_pytorch.utils.grad import set_grad_enabled
        >>> 
        >>> # Disable gradients for inference
        >>> with set_grad_enabled(False):
        ...     result = model(input_tensor)
        >>> 
        >>> # Enable gradients for training
        >>> with set_grad_enabled(True):
        ...     loss = criterion(model(input_tensor), target)
        ...     loss.backward()
    """
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(enabled)
    try:
        yield
    finally:
        torch.set_grad_enabled(prev_grad_state)

