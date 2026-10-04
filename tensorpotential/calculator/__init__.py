from tensorpotential.calculator.foundation_models import grace_fm
from tensorpotential.core.lazy import lazy_exports

__all__ = ["TPCalculator", "grace_fm", "predict_structures"]

# The TensorFlow calculator and the bulk predictor are imported on first access (PEP 562), so that
# ``tensorpotential.calculator.foundation_models`` imports without TensorFlow.
__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "TPCalculator": "tensorpotential.calculator.asecalculator",
        "predict_structures": "tensorpotential.calculator.bulk",
    },
)
