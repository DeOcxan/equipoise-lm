"""
Module Loader Utility

Handles loading modules from folders with non-Python-compliant names (dashes).

This is necessary because Python identifiers cannot contain dashes,
but our folder structure uses them (EBT_IMG, EBT_NLP, Simple_EBM).

Author: DeOcxan
Date: October 2025
"""

import importlib.util
from pathlib import Path
from typing import Any


def load_module_from_path(module_name: str, file_path: str) -> Any:
    """
    Load a Python module from a file path.
    
    This is useful when the folder name contains characters that
    are not valid in Python identifiers (like dashes).
    
    Args:
        module_name: Name to give the loaded module
        file_path: Path to the .py file ("EBT_IMG/ebt_img_denoise.py")
        
    Returns:
        Loaded module object
        
    Example:
        >>> ebt_img = load_module_from_path("ebt_img", "EBT_IMG/ebt_img_denoise.py")
        >>> Model = ebt_img.EBT_IMG_Denoise
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Module file not found: {file_path}")
    
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module


def load_ebt_img_module():
    """Load EBT_IMG module."""
    return load_module_from_path("ebt_img_denoise", "EBT_IMG/ebt_img_denoise.py")


def load_ebt_nlp_module():
    """Load EBT_NLP module."""
    return load_module_from_path("ebt_nlp", "EBT_NLP/ebt_nlp.py")


def load_simple_ebm_module():
    """Load Simple_EBM module."""
    return load_module_from_path("simple_ebm", "Simple_EBM/simple_ebm.py")


# Convenience functions for common imports
def get_ebt_img_denoise_class():
    """Get EBT_IMG_Denoise class."""
    module = load_ebt_img_module()
    return module.EBT_IMG_Denoise


def get_ebt_nlp_class():
    """Get EBT_NLP class."""
    module = load_ebt_nlp_module()
    return module.EBT_NLP


def get_simple_ebm_class():
    """Get SimpleEBM class."""
    module = load_simple_ebm_module()
    return module.SimpleEBM

