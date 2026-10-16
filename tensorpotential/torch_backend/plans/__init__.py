"""Index tables ("plans") of the twins, built from the spec in pure numpy.

Layer: above :mod:`~tensorpotential.torch_backend.spec`, below the ops. Plan builders are pure functions (no I/O,
no global state) and import neither torch nor TensorFlow (import contracts); they own a copy of the TF table logic,
guarded by table-equality tests against tables dumped from TensorFlow (rule R1, decision D7). A plan whose
TensorFlow counterpart no golden stores (the spherical harmonics, whose coefficients differ from TF's) is checked
through the outputs of its op instead.
"""
