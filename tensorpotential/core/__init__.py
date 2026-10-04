"""Code shared by the TensorFlow and PyTorch backends.

Nothing in this package may import ``tensorflow``, ``tf_keras`` or ``torch``, directly or through
another module of ``tensorpotential`` (enforced by the import-linter contract in ``pyproject.toml``
and by ``tests/test_core_imports.py``).
"""
