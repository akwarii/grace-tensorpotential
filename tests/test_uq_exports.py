"""The public names of ``tensorpotential.uq`` resolve to the objects of their defining modules."""

from __future__ import annotations

import json

from tests.fresh_python import run_fresh_python

EXPORTS = {
    "setup_feature_calculator": "tensorpotential.uq.feature_extraction",
    "extract_features": "tensorpotential.uq.feature_extraction",
    "extract_features_bulk": "tensorpotential.uq.feature_extraction",
    "FeatureBuffer": "tensorpotential.uq.feature_extraction",
    "batch_feature_chunks": "tensorpotential.uq.feature_extraction",
    "batched_feature_iterator": "tensorpotential.uq.feature_extraction",
    "tf_dataset_feature_iterator": "tensorpotential.uq.feature_extraction",
    "GMMUQArtifactBuilder": "tensorpotential.uq.artifact_builder",
    "GMMUQModel": "tensorpotential.uq.gmmuq",
    "ComputeStructureEnergyAndForcesAndVirialAndUncertainty": "tensorpotential.uq.compute",
    "ComputeStructureEnergyAndForcesAndVirialAndGammaOnly": "tensorpotential.uq.compute",
}


def test_all_lists_exactly_the_public_names(tmp_path):
    code = "import tensorpotential.uq as uq; print(sorted(uq.__all__))"
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == str(sorted(EXPORTS))


def test_every_public_name_is_the_object_of_its_defining_module(tmp_path):
    code = (
        "import importlib, json, tensorpotential.uq as uq\n"
        "bad = []\n"
        "for name, module in json.loads(__EXPORTS__).items():\n"
        "    owner = importlib.import_module(module)\n"
        "    namespace = {}\n"
        "    exec(f'from tensorpotential.uq import {name}', namespace)\n"
        "    if not (getattr(uq, name) is getattr(owner, name) is namespace[name] and name in dir(uq)):\n"
        "        bad.append(name)\n"
        "print(json.dumps(bad))\n"
    ).replace("__EXPORTS__", repr(json.dumps(EXPORTS)))
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "[]"


def test_a_submodule_still_imports_from_the_package(tmp_path):
    code = "from tensorpotential.uq import constants, gmmuq; print(constants.__name__, gmmuq.__name__)"
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-2:] == [
        "tensorpotential.uq.constants",
        "tensorpotential.uq.gmmuq",
    ]


def test_an_unknown_name_raises_attribute_error(tmp_path):
    code = (
        "import tensorpotential.uq as uq\n"
        "try:\n    uq.no_such_name\nexcept AttributeError:\n    print('ok')\n"
    )
    result = run_fresh_python(code, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "ok"
