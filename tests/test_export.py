import numpy as np
import tensorflow as tf
import yaml

from tensorpotential.potentials.presets import FS
from tensorpotential.export import export_to_yaml


def test_export_to_yaml(tmp_path):
    np.random.seed(1)
    tf.random.set_seed(1)

    fs_ins = FS(
        element_map={"Al": 0, "Li": 1, "H": 2},
        lmax=2,
        n_rad_base=4,
        n_rad_max=12,
        embedding_size=8,
        max_order=3,
    ).get_instructions()

    # BUILD
    for _, ins in fs_ins.items():
        ins.build(tf.float64)

    out_file = tmp_path / "test_export_to_yaml.yaml"
    export_to_yaml(fs_ins, str(out_file))

    with open(out_file) as f:
        pot_yaml = yaml.load(f, yaml.SafeLoader)

    print(f"pot_yaml.keys={pot_yaml.keys()}")

    for k in ["elements", "emb_spec", "radial_basis", "functions"]:
        assert k in pot_yaml

    assert pot_yaml["elements"] == ["Al", "Li", "H"]
    assert pot_yaml["emb_spec"] == {
        "params": [1.0, 1.0, 1.0, 0.5, 1.0, 2.0, 1.0, 0.75],
        "type": "FinnisSinclairShiftedScaled",
    }
    print("radial_basis.keys=", pot_yaml["radial_basis"].keys())
    for k in ["Z", "crad", "cutoff", "n_rad_base", "nradmax", "radbasename"]:
        assert k in pot_yaml["radial_basis"]
    assert np.shape(pot_yaml["radial_basis"]["Z"]) == (12 * 3,)

    # shape = [n_rad_max, lmax + 1, n_rad_base]
    assert np.shape(pot_yaml["radial_basis"]["crad"]) == (12 * 3 * 4,)

    assert len(pot_yaml["functions"]) == 3
    assert len(pot_yaml["functions"][0]) == 108

    tdbfunc = pot_yaml["functions"][0][12]
    print(f"tdbfunc={tdbfunc}")
    for k, v in {
        "mu0": 0,
        "ns": 1,
        "ls": [0, 0],
        "ms_combs": [0, 0],
        "gen_cgs": [1.0],
        "ndensity": 4,
        "coeff": [-0.00653062, 0.2612635, -0.33575857, 0.86569158],
        "rank": 2,
    }.items():
        assert k in tdbfunc
        assert tdbfunc[k] == v or np.allclose(tdbfunc[k], v)


def test_export_to_yaml_gives_the_same_file_for_a_list_and_a_dict_and_records_shift_and_scale(tmp_path):
    tf.random.set_seed(1)
    fs_ins = FS(
        element_map={"Al": 0, "Li": 1},
        lmax=1,
        n_rad_base=4,
        n_rad_max=4,
        embedding_size=4,
        max_order=2,
    ).get_instructions()
    for _, ins in fs_ins.items():
        ins.build(tf.float64)

    from_dict = tmp_path / "dict.yaml"
    from_list = tmp_path / "list.yaml"
    export_to_yaml(fs_ins, str(from_dict))
    export_to_yaml(list(fs_ins.values()), str(from_list))
    assert from_dict.read_text() == from_list.read_text()

    exported = yaml.safe_load(from_dict.read_text())
    assert exported["shift"] == 0.0 and exported["scale"] == 1.0
    # one block per element; every function carries n_rad_max radial indices' coefficients
    assert sorted(exported["functions"]) == [0, 1]
    assert len(exported["functions"][0]) == len(exported["functions"][1])
    assert {f["ns"] for f in exported["functions"][0]} == {1, 2, 3, 4}


def test_shift_and_scale_entries_follow_what_the_target_instruction_holds():
    from types import SimpleNamespace

    from tensorpotential.export import _add_shift_and_scale

    target = SimpleNamespace(
        scale=tf.constant(2.0, dtype=tf.float64),
        constant_shift=1.5,
        atomic_shift_map=tf.constant([[1.0], [2.0]], dtype=tf.float64),
    )
    entries: dict = {}
    _add_shift_and_scale(entries, {"ConstantScaleShiftTarget": target})
    assert entries == {"E0": [1.0, 2.0], "shift": 1.5, "scale": 2.0}

    # a plain-number scale has no .numpy(); no atomic shifts: only shift and scale are written
    plain = SimpleNamespace(scale=3.0, constant_shift=0.25, atomic_shift_map=None)
    entries = {}
    _add_shift_and_scale(entries, {"ConstantScaleShiftTarget": plain})
    assert entries == {"shift": 0.25, "scale": 3.0}

    # a scale that is not a number of this model (None): no scale entry
    none_scale = SimpleNamespace(scale=None, constant_shift=0.5, atomic_shift_map=None)
    entries = {}
    _add_shift_and_scale(entries, {"ConstantScaleShiftTarget": none_scale})
    assert entries == {"shift": 0.5}

    entries = {}
    _add_shift_and_scale(entries, {})  # no target instruction: nothing added
    assert entries == {}
