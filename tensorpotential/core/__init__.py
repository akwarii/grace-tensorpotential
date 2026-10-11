"""Code shared by the TensorFlow and PyTorch backends.

Nothing in this package may import ``tensorflow``, ``keras`` or ``torch``, directly or through
another module of ``tensorpotential`` (enforced by the import-linter contract in ``pyproject.toml``
and by the gates in ``tests/test_import_gates.py``).
"""
