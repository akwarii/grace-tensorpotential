"""Uncertainty quantification with Gaussian mixture models.

The public names are imported on first access (PEP 562, :mod:`tensorpotential.core.lazy`): the submodules import
this package for ``tensorpotential.uq.constants``, so an eager import here would make the package and its submodules
import each other.
"""

from __future__ import annotations

from tensorpotential.core.lazy import lazy_exports

_FEATURES = "tensorpotential.uq.feature_extraction"
_COMPUTE = "tensorpotential.uq.compute"

_EXPORTS = {
    "setup_feature_calculator": _FEATURES,
    "extract_features": _FEATURES,
    "extract_features_bulk": _FEATURES,
    "FeatureBuffer": _FEATURES,
    "batch_feature_chunks": _FEATURES,
    "batched_feature_iterator": _FEATURES,
    "tf_dataset_feature_iterator": _FEATURES,
    "GMMUQArtifactBuilder": "tensorpotential.uq.artifact_builder",
    "GMMUQModel": "tensorpotential.uq.gmmuq",
    "ComputeStructureEnergyAndForcesAndVirialAndUncertainty": _COMPUTE,
    "ComputeStructureEnergyAndForcesAndVirialAndGammaOnly": _COMPUTE,
}

__all__ = list(_EXPORTS)

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
