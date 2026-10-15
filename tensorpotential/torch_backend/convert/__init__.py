"""Assembly of a torch model from a ``model.yaml`` and its weight files.

The spec, the twin of each instruction, the weights placed into them, the executor.

Layer: above the model and the instructions, below :mod:`~tensorpotential.torch_backend.torchsim`; the one layer
that knows both the concrete twins and the executor.
"""
