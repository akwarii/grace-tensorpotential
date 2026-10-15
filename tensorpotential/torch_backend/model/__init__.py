"""The executor: runs the twins of a model in file order and derives forces and stress from the energy.

Layer: above the ops, beside :mod:`~tensorpotential.torch_backend.instructions`. It depends on
:class:`~tensorpotential.torch_backend.protocol.Twin` and receives the twins it runs; it imports no concrete twin
(import contract), so a twin is added without changing the executor.
"""
