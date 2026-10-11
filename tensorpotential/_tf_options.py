"""Import this module before the first ``import tensorflow`` of a TensorFlow-side module.

Importing it applies the package-wide TensorFlow options (numpy-style type promotion, no
TensorFloat-32) once, because a module body runs only on the first import. ``tensorpotential`` itself
no longer imports TensorFlow, so the options are applied here, by the first TensorFlow-side module
that is imported, instead of by the package ``__init__``.
"""

import tensorpotential
from tensorpotential.core.backends import require_backend

# the one place where every console script and TensorFlow-side module learns that the 'tf' extra is missing
require_backend("tf")
tensorpotential._configure_tf_options(verbose=True)
