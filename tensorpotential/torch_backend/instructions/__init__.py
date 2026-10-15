"""The twins: one torch module per supported TF instruction class.

Each implements :class:`~tensorpotential.torch_backend.protocol.Twin`.

Layer: above the ops and the plans, beside :mod:`~tensorpotential.torch_backend.model`, which never imports a twin
(import contract: the executor depends on the protocol only).
"""
