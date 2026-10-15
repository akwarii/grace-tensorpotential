"""Weight files of a converted model.

``npz`` arrays (pickle disabled) and JSON metadata keyed by ``<instruction>/<attribute>`` (rule R6).

Layer: next to the plans, above :mod:`~tensorpotential.torch_backend.spec`. It reads and checks files only, in
numpy, and imports neither torch nor TensorFlow (import contracts); the TF-side extractor that writes the files
lives in ``tensorpotential/scripts``.
"""
