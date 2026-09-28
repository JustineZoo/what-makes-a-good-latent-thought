"""Latent-concept diffusion LM.

A controlled study of *representational granularity* (compression ratio R) x
*generation paradigm* (autoregressive vs diffusion) x *denoising steps* T, and
their effect on multi-step reasoning.

The top-level package deliberately imports nothing that requires torch, so the
data / oracle / analysis layers are usable on a login node with no GPU. Import
the torch-backed pieces (``dlm.models``, ``dlm.concept``, ``dlm.generators``,
``dlm.train``) explicitly; each imports torch lazily.
"""

__version__ = "0.0.1"

# Convenience re-exports for the torch-free layers only.
from dlm.data import make_task, TASKS  # noqa: E402,F401

__all__ = ["make_task", "TASKS", "__version__"]
