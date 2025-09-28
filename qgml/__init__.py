"""
QGML Backend Abstraction Layer

Provides unified interface for JAX and PyTorch implementations,
allowing seamless switching between computational backends.
"""

import os
from typing import Optional, Union, Any
from enum import Enum

class Backend(Enum):
    """Available computational backends."""
    PYTORCH = "pytorch"
    JAX = "jax"

class BackendManager:
    """Manages backend selection and switching."""
    
    _current_backend: Optional[Backend] = None
    _backend_modules = {}
    
    @classmethod
    def set_backend(cls, backend: Union[str, Backend]) -> None:
        """Set the computational backend."""
        if isinstance(backend, str):
            backend = Backend(backend.lower())
        
        cls._current_backend = backend
        
        # Clear cached modules to force reload
        cls._backend_modules.clear()
        
        print(f"QGML backend set to: {backend.value}")
    
    @classmethod
    def get_backend(cls) -> Backend:
        """Get current backend."""
        if cls._current_backend is None:
            # Default to PyTorch
            cls.set_backend(Backend.PYTORCH)
        return cls._current_backend
    
    @classmethod
    def get_backend_module(cls, module_name: str) -> Any:
        """Get backend-specific module."""
        backend = cls.get_backend()
        
        if module_name not in cls._backend_modules:
            if backend == Backend.PYTORCH:
                from .pytorch_backend import get_module
                cls._backend_modules[module_name] = get_module(module_name)
            elif backend == Backend.JAX:
                from .jax_backend import get_module
                cls._backend_modules[module_name] = get_module(module_name)
            else:
                raise ValueError(f"Unknown backend: {backend}")
        
        return cls._backend_modules[module_name]

# Convenience functions
def set_backend(backend: Union[str, Backend]) -> None:
    """Set computational backend."""
    BackendManager.set_backend(backend)

def get_backend() -> Backend:
    """Get current backend."""
    return BackendManager.get_backend()

def get_matrix_ops():
    """Get matrix operations for current backend."""
    return BackendManager.get_backend_module("matrix_ops")

def get_optimizer():
    """Get optimizer for current backend."""
    return BackendManager.get_backend_module("optimizer")

# Auto-detect backend from environment
if "QGML_BACKEND" in os.environ:
    set_backend(os.environ["QGML_BACKEND"])
elif "JAX_ENABLE_X64" in os.environ:
    set_backend(Backend.JAX)
else:
    set_backend(Backend.PYTORCH)
