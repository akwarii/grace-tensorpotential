import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import bulk, fcc111

from tensorpotential import constants
from tensorpotential.data.databuilder import construct_batches, GeometricalDataBuilder
from tests.neighbour_oracle import brute_force_pairs, recover_shifts
from tests.tolerances import NEIGHBOUR_VECTOR_F64 as VECTOR_TOL

import pytest

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_construct_batches_list() -> None:
    at1 = bulk("Al", cubic=True)
    at2 = bulk("Al", "sc", a=5, cubic=True)

    at3 = bulk("Cu", cubic=True)
    at4 = bulk("Cu", "sc", a=3, cubic=True)

    data_builders = [GeometricalDataBuilder({"Al": 0, "Cu": 1}, cutoff=6)]
    ase_atoms_list = [at1, at2, at3, at4]
    batches, padding_stats = construct_batches(
        ase_atoms_list,
        data_builders=data_builders,
        batch_size=2,
        max_n_buckets=1,
        return_padding_stats=True,
        verbose=False,
    )

    print("batches", batches)
    print("padding_stats", padding_stats)
    assert len(batches) == 2
    assert padding_stats is not None
    b0 = batches[0]
    b1 = batches[1]
    assert b0["n_struct_total"] == 3
    assert b1["n_struct_total"] == 3

    assert b0["batch_tot_nat"] == 6
    assert b1["batch_tot_nat"] == 6


def test_construct_batches_df() -> None:
    df = pd.read_pickle(os.path.join(dir_path, "data/MoNbTaW_train50.pkl.gz"))
    elements_map = {"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}
    data_builders = [GeometricalDataBuilder(elements_map=elements_map, cutoff=6)]

    batches, padding_stats = construct_batches(
        df,
        data_builders=data_builders,
        batch_size=10,
        max_n_buckets=2,
        return_padding_stats=True,
        verbose=True,
    )

    print("padding_stats", padding_stats)
    padding_stats_ref = {
        "pad_nstruct": 5,
        "pad_nat": 5,
        "pad_nneigh": 56,
        "nreal_struc": 50,
        "nreal_atoms": 844,
        "nreal_neigh": 48904,
    }
    assert len(batches) == 5
    assert padding_stats is not None
    b0 = batches[0]
    b1 = batches[1]
    assert b0["n_struct_total"] == 11
    assert b1["n_struct_total"] == 11

    assert padding_stats_ref == padding_stats_ref


@pytest.mark.xfail
def test_construct_batches_multiple_db() -> None:
    from tensorpotential.constants import CELL_VECTORS, ATOMIC_POS

    try:
        from tensorpotential.extra.gen_tensor.databuilder import (
            PositionsDataBuilder,
            CellDataBuilder,
        )
    except ImportError:
        assert 1 == 0
    df = pd.read_pickle(os.path.join(dir_path, "data/MoNbTaW_train50.pkl.gz"))
    elements_map = {"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}
    data_builders = [
        GeometricalDataBuilder(elements_map=elements_map, cutoff=6),
        PositionsDataBuilder(),
        CellDataBuilder(),
    ]

    batches, padding_stats = construct_batches(
        df,
        data_builders=data_builders,
        batch_size=10,
        max_n_buckets=2,
        return_padding_stats=True,
        verbose=True,
    )

    print("padding_stats", padding_stats)
    padding_stats_ref = {
        "pad_nstruct": 5,
        "pad_nat": 5,
        "pad_nneigh": 56,
        "nreal_struc": 50,
        "nreal_atoms": 844,
        "nreal_neigh": 48904,
    }
    assert len(batches) == 5
    assert padding_stats is not None
    b0 = batches[0]
    b1 = batches[1]
    assert b0["n_struct_total"] == 11
    assert b1["n_struct_total"] == 11
    #
    assert CELL_VECTORS in b0
    assert len(b1[CELL_VECTORS].shape) == 3
    assert ATOMIC_POS in b1
    print("b0", b0[CELL_VECTORS])

    assert padding_stats_ref == padding_stats_ref


def test_bucketing_split_dense_one_width_per_batch():
    from ase.build import bulk
    from tensorpotential.data.databuilder import bucketing_split_dense
    from tensorpotential import constants as C

    db = GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=True)
    # 3 small cells + 1 supercell; a supercell has IDENTICAL per-atom coordination to
    # its primitive cell, so all four may land in the same band. This test checks the
    # per-batch invariant (PAD_MAX_NEIGH present, bond count derived correctly, no
    # structure lost) regardless of how many bands the planner produces.
    structs = [
        bulk("Cu", "fcc", a=3.6, cubic=True),
        bulk("Cu", "fcc", a=3.6, cubic=True),
        bulk("Cu", "fcc", a=3.6, cubic=True),
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2),
    ]
    data = [db.extract_from_ase_atoms(s) for s in structs]

    batches, max_pad = bucketing_split_dense(
        data, batch_size=4, max_n_buckets="auto", slot_budget="auto"
    )
    assert len(batches) == len(max_pad)
    # every batch carries an integer reshape width and the derived bond count
    for mp in max_pad:
        assert C.PAD_MAX_NEIGH in mp
        assert mp[C.PAD_MAX_N_NEIGHBORS] == mp[C.PAD_MAX_N_ATOMS] * mp[C.PAD_MAX_NEIGH]
    # no structure lost
    total = sum(len(b) for b in batches)
    assert total == 4


def test_bucketing_split_dense_drop_cap(caplog):
    from ase.build import bulk
    from tensorpotential.data.databuilder import bucketing_split_dense

    db = GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=True)
    small = bulk("Cu", "fcc", a=3.6, cubic=True)
    data = [db.extract_from_ase_atoms(small) for _ in range(3)]
    # cap=0 drops everything (every atom has >=1 neighbor)
    with caplog.at_level(logging.WARNING):
        batches, max_pad = bucketing_split_dense(
            data, batch_size=4, max_n_buckets="auto", max_neigh_cap=0
        )
    assert batches == [] and max_pad == []
    assert any("dense_max_neigh_cap=0" in r.message for r in caplog.records)


def test_bucketing_split_dense_separates_buckets_by_max_neigh():
    from ase.build import bulk
    from tensorpotential.data.databuilder import bucketing_split_dense, _struct_max_neigh
    from tensorpotential import constants as C

    db = GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=True)
    normal = bulk("Cu", "fcc", a=3.6, cubic=True)      # lower coordination within rcut
    compressed = bulk("Cu", "fcc", a=2.2, cubic=True)  # high coordination within rcut
    data = [db.extract_from_ase_atoms(normal), db.extract_from_ase_atoms(compressed)]
    mns = [_struct_max_neigh(d) for d in data]
    # sanity: the two structures have clearly different max_neigh
    assert abs(mns[0] - mns[1]) >= 16, mns

    batches, max_pad = bucketing_split_dense(
        data, batch_size=4, max_n_buckets="auto", slot_budget="auto"
    )
    # adaptive bucketing -> the two distinct max_neigh fall in separate width buckets,
    # each padded to its own max -> different reshape widths -> cannot share a batch
    widths = {mp[C.PAD_MAX_NEIGH] for mp in max_pad}
    assert len(widths) >= 2, widths
    assert len(batches) >= 2
    for mp in max_pad:
        assert mp[C.PAD_MAX_N_NEIGHBORS] == mp[C.PAD_MAX_N_ATOMS] * mp[C.PAD_MAX_NEIGH]
    # each width hugs its bucket's own max (adaptive, not a rounded-up fixed tier)
    assert sorted(widths) == sorted(mns)


def test_construct_batches_df_parallel() -> None:
    df = pd.read_pickle(os.path.join(dir_path, "data/MoNbTaW_train50.pkl.gz"))
    elements_map = {"Mo": 0, "Nb": 1, "Ta": 2, "W": 3}
    data_builders = [GeometricalDataBuilder(elements_map=elements_map, cutoff=6)]

    batches, padding_stats = construct_batches(
        df,
        data_builders=data_builders,
        batch_size=10,
        max_n_buckets=2,
        return_padding_stats=True,
        verbose=True,
        max_workers=2,
    )

    print("padding_stats", padding_stats)
    padding_stats_ref = {
        "pad_nstruct": 5,
        "pad_nat": 5,
        "pad_nneigh": 56,
        "nreal_struc": 50,
        "nreal_atoms": 844,
        "nreal_neigh": 48904,
    }
    assert len(batches) == 5
    assert padding_stats is not None
    b0 = batches[0]
    b1 = batches[1]
    assert b0["n_struct_total"] == 11
    assert b1["n_struct_total"] == 11
    assert b0["batch_tot_nat"] == 199
    assert b1["batch_tot_nat"] == 199

    assert padding_stats_ref == padding_stats_ref


def test_pad_batch_dense_groups_bonds_into_per_atom_blocks():
    """Dense reshape places atom i's bonds in the contiguous slot block [i*width,(i+1)*width):
    reshaping BOND_IND_I to [max_nat, width] gives rows whose REAL (within-cutoff) entries all
    equal the row index, with exactly counts[i] real entries per row. A vacancy makes
    coordination heterogeneous, so this is FALSE for the flat seg-sum layout."""
    import numpy as np
    from ase.build import bulk
    from tensorpotential import constants as C

    db = GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=True)
    s = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    del s[0]  # vacancy -> heterogeneous coordination
    batch = db.join_to_batch([db.extract_from_ase_atoms(s)])
    real_nat = int(np.asarray(batch[C.N_ATOMS_BATCH_REAL]))
    counts = np.bincount(np.asarray(batch[C.BOND_IND_I]), minlength=real_nat)[:real_nat]
    assert counts.min() < counts.max(), counts  # sanity: heterogeneous, else vacuous
    width = int(counts.max())

    max_nat = real_nat
    db.pad_batch(batch, {
        C.PAD_MAX_N_STRUCTURES: 1, C.PAD_MAX_N_ATOMS: max_nat,
        C.PAD_MAX_NEIGH: width, C.PAD_MAX_N_NEIGHBORS: max_nat * width,
    })

    assert int(np.asarray(batch[C.BOND_IND_I]).shape[0]) == max_nat * width
    ind_i = np.asarray(batch[C.BOND_IND_I]).reshape(max_nat, width)
    bnorm = np.linalg.norm(np.asarray(batch[C.BOND_VECTOR]).reshape(max_nat, width, 3), axis=2)
    for atom in range(real_nat):
        real = bnorm[atom] <= db.cutoff
        assert real.sum() == counts[atom], (atom, int(real.sum()), int(counts[atom]))
        assert np.all(ind_i[atom][real] == atom), atom


def test_pad_batch_dense_fake_atom_block_all_dummy():
    """With atom padding (max_nat = real_nat + 1) the fake atom gets a full width block of
    dummy bonds (> cutoff), real atom blocks stay centered on their index, and the dense bond
    count is max_nat * width."""
    import numpy as np
    from ase.build import bulk
    from tensorpotential import constants as C

    db = GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=True)
    s = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    del s[0]
    batch = db.join_to_batch([db.extract_from_ase_atoms(s)])
    real_nat = int(np.asarray(batch[C.N_ATOMS_BATCH_REAL]))
    counts = np.bincount(np.asarray(batch[C.BOND_IND_I]), minlength=real_nat)[:real_nat]
    width = int(counts.max())
    max_nat = real_nat + 1  # one fake atom
    db.pad_batch(batch, {
        C.PAD_MAX_N_STRUCTURES: 2, C.PAD_MAX_N_ATOMS: max_nat,
        C.PAD_MAX_NEIGH: width, C.PAD_MAX_N_NEIGHBORS: max_nat * width,
    })
    assert int(np.asarray(batch[C.ATOMIC_MU_I]).shape[0]) == max_nat
    assert int(np.asarray(batch[C.BOND_IND_I]).shape[0]) == max_nat * width
    bnorm = np.linalg.norm(np.asarray(batch[C.BOND_VECTOR]).reshape(max_nat, width, 3), axis=2)
    assert np.all(bnorm[max_nat - 1] > db.cutoff)  # fake atom block entirely dummy
    ind_i = np.asarray(batch[C.BOND_IND_I]).reshape(max_nat, width)
    for atom in range(real_nat):
        real = bnorm[atom] <= db.cutoff
        assert np.all(ind_i[atom][real] == atom), atom


def test_construct_batches_dense_layout_and_stats():
    import numpy as np
    from ase.build import bulk
    from tensorpotential import constants as C

    structs = [
        bulk("Cu", "fcc", a=3.6, cubic=True),
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 1, 1),
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 1),
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2),
    ]
    db = GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=True)
    batches, stats = construct_batches(
        structs,
        data_builders=[db],
        batch_size=2,
        max_n_buckets="auto",
        return_padding_stats=True,
        verbose=False,
    )
    # every batch is in the per-atom-uniform layout
    for b in batches:
        nb = int(np.asarray(b[C.BOND_IND_I]).shape[0])
        nat = int(np.asarray(b[C.N_ATOMS_BATCH_TOTAL]))
        assert nb % nat == 0, f"{nb} not a multiple of {nat}"
    assert stats is not None
    assert stats["nreal_atoms"] == sum(len(s) for s in structs)


def test_construct_batches_dense_matches_segment_real_counts():
    from ase.build import bulk

    structs = [bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2) for _ in range(3)]
    seg = construct_batches(
        structs, data_builders=[GeometricalDataBuilder({"Cu": 0}, cutoff=6.0)],
        batch_size=3, max_n_buckets=1, return_padding_stats=True, verbose=False,
    )[1]
    dense = construct_batches(
        structs, data_builders=[GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=True)],
        batch_size=3, max_n_buckets=1, return_padding_stats=True, verbose=False,
    )[1]
    # identical real content, only padding differs
    assert seg["nreal_atoms"] == dense["nreal_atoms"]
    assert seg["nreal_neigh"] == dense["nreal_neigh"]


def test_dense_batch_parity_with_segment_sum(cu_two_layer):
    import numpy as np
    import tensorflow as tf
    from ase.build import bulk
    from tensorpotential import constants as C

    # Three 2x2x1 supercells (16 atoms each) with different rattles.
    # Equal atom counts are required: the dense planner sorts structures by nat within a band
    # (pack_structures_elastic), so mixed-size structures arrive in a different order than the
    # segment path, which preserves insertion order.  Same-size structures are stable under that
    # sort, so both paths see the same ordering and the real-prefix comparison is valid.
    structs = [
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 1),
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 1),
        bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 1),
    ]
    for i, s in enumerate(structs):
        s.rattle(stdev=0.05, seed=i)

    def make_batch(dense):
        db = GeometricalDataBuilder({"Cu": 0}, cutoff=6.0, dense_nbr=dense)
        batches = construct_batches(
            [s.copy() for s in structs], data_builders=[db],
            batch_size=3, max_n_buckets=1, verbose=False,
        )
        assert len(batches) == 1, f"expected 1 batch, got {len(batches)}"
        return batches[0]

    def to_tensors(batch):
        out = {}
        for k, v in batch.items():
            arr = np.asarray(v)
            if np.issubdtype(arr.dtype, np.floating):
                out[k] = tf.constant(arr, dtype=tf.float64)
            else:
                out[k] = tf.constant(arr, dtype=tf.int32)
        return out

    m_seg, m_den = cu_two_layer(False), cu_two_layer(True)
    seg = to_tensors(make_batch(False))
    den = to_tensors(make_batch(True))

    r_seg = m_seg.train_function(m_seg.instructions, seg)
    r_den = m_den.train_function(m_den.instructions, den)

    n_struct = len(structs)
    n_atoms = sum(len(s) for s in structs)
    e_seg = r_seg[C.PREDICT_TOTAL_ENERGY].numpy()[:n_struct]
    e_den = r_den[C.PREDICT_TOTAL_ENERGY].numpy()[:n_struct]
    f_seg = r_seg[C.PREDICT_FORCES].numpy()[:n_atoms]
    f_den = r_den[C.PREDICT_FORCES].numpy()[:n_atoms]

    assert np.allclose(e_seg, e_den, atol=1e-6, rtol=0), (e_seg, e_den)
    assert np.allclose(f_seg, f_den, atol=1e-5, rtol=0)


def test_dense_export_resets_dense_nbr_flag(tmp_path):
    """Regression: exporting a dense-capable model must leave dense_nbr=False on the
    shared instructions afterwards.

    save_model flips dense_nbr=True on the shared instructions to trace the
    `compute_dense` signature. Sibling signatures traced in the same export that read
    the flag but never set it (the aux computes and, via grace_uq, `compute_uq*`) would
    otherwise inherit that True and bake the dense reshape (`n_bonds // n_atoms`) into
    their flat-layout graphs -- a size mismatch at inference. The dual signature is
    emitted for any equivariant model, so this is the standard grace_uq export path.
    """
    import numpy as np
    import tensorflow as tf
    from tensorpotential import TPModel
    from tensorpotential.potentials.presets import GRACE_2LAYER_v2_25

    tf.random.set_seed(7)
    np.random.seed(7)
    ins = GRACE_2LAYER_v2_25(
        element_map={"Cu": 0}, rcut=6.0, dense_nbr=True
    ).get_instructions()
    m = TPModel(ins)
    m.build(tf.float64)

    instr = m.instructions
    dense_instr = [
        it
        for it in (instr.values() if hasattr(instr, "values") else instr)
        if getattr(it, "dense_capable", False)
    ]
    assert dense_instr, "expected a dense-capable instruction (equivariant SPBF)"
    # opt-in dense model: the flag is True at rest before export.
    assert any(it.dense_nbr for it in dense_instr)

    m.save_model(str(tmp_path / "m"), input_signature_float_dtype=tf.float64)

    # The export must restore dense_nbr to False so co-traced signatures stay on
    # segment_sum regardless of TF's trace order.
    assert all(not it.dense_nbr for it in dense_instr), [
        it.dense_nbr for it in dense_instr
    ]

    loaded = tf.saved_model.load(str(tmp_path / "m"))
    assert "compute" in loaded.signatures
    assert "compute_dense" in loaded.signatures


# --------------------------------------------------------------------------------------------
# Neighbour list of GeometricalDataBuilder.extract_from_ase_atoms against a brute-force oracle
# (tests/neighbour_oracle.py, which imports nothing from tensorpotential). Logic layer: which
# edges exist, per structure kind. Value layer: the edges and vectors equal the oracle's, and
# coordination numbers of perfect lattices equal hand-counted shells.
# --------------------------------------------------------------------------------------------

ELEMENTS = {"Cu": 0, "Al": 1}
A_CU = 3.61


def run_builder(atoms, cutoff=5.0, **kwargs):
    """Run the builder on a copy; return the copy as the builder left it and the result."""
    work = atoms.copy()
    builder = GeometricalDataBuilder(ELEMENTS, cutoff=cutoff, **kwargs)
    return work, builder.extract_from_ase_atoms(work)


def oracle_cutoff(cutoff, cutoff_table):
    """Per-pair cutoff function for the oracle: the table entry, else the global cutoff."""
    if cutoff_table is None:
        return cutoff
    return lambda a, b: cutoff_table.get((a, b), cutoff)


def edges_with_vectors(work, data, isolated):
    """Real edges of a builder result as ``{(i, j, shift): vector}``, dummy bonds removed.

    An atom without a neighbour gets one dummy bond ``(i, 0)``; ``isolated`` lists those atoms.
    """
    ind_i = data[constants.BOND_IND_I]
    ind_j = data[constants.BOND_IND_J]
    vectors = data[constants.BOND_VECTOR]
    real = ~np.isin(ind_i, isolated)
    shifts = recover_shifts(
        work.positions, work.cell.array, ind_i[real], ind_j[real], vectors[real]
    )
    keys = list(zip(ind_i[real].tolist(), ind_j[real].tolist(), shifts))
    assert len(set(keys)) == len(keys), "the builder returned a repeated edge"
    return dict(zip(keys, vectors[real]))


def check_against_oracle(atoms, cutoff=5.0, cutoff_table=None, **kwargs):
    """Assert that the builder's edges are the oracle's (exactly) and vectors agree (tolerance)."""
    work, data = run_builder(atoms, cutoff, **kwargs)
    expected = brute_force_pairs(
        work.positions,
        work.cell.array,
        work.get_chemical_symbols(),
        oracle_cutoff(cutoff, cutoff_table),
    )
    isolated = sorted(set(range(len(work))) - {p.i for p in expected})
    got = edges_with_vectors(work, data, isolated)
    assert sorted(got) == [(p.i, p.j, p.shift) for p in expected]
    for p in expected:
        np.testing.assert_allclose(
            got[(p.i, p.j, p.shift)], p.vector, rtol=VECTOR_TOL.rtol, atol=VECTOR_TOL.atol
        )
    assert int(data[constants.N_NEIGHBORS_REAL]) == len(expected) + len(isolated)
    return work, data, expected, isolated


def fcc_cubic():
    return bulk("Cu", "fcc", a=A_CU, cubic=True)


def rattled_eight_atoms():
    """Eight atoms, two elements, a triclinic cell, positions displaced by about 0.1 A."""
    atoms = fcc_cubic().repeat((2, 1, 1))
    atoms.set_chemical_symbols(["Cu", "Al"] * 4)
    atoms.set_cell([[2 * A_CU, 0, 0], [1.0, A_CU, 0], [0.5, 0.4, A_CU]], scale_atoms=True)
    atoms.rattle(0.1, seed=3)
    return atoms


def thin_slab():
    """Cu(111), 2 x 2 x 3, open along z with only 1 A of vacuum on each side."""
    slab = fcc111("Cu", size=(2, 2, 3), a=A_CU, vacuum=1.0)
    assert list(slab.pbc) == [True, True, False]
    return slab


def dimer_in_box():
    return Atoms(
        "Cu2", positions=[[10.0, 10, 10], [13.0, 10, 10]], cell=25.0 * np.eye(3), pbc=True
    )


def isolated_atom_cluster():
    """Three atoms close together and one 12 A away, no cell, no periodicity."""
    positions = [[0.0, 0, 0], [2.4, 0, 0], [1.2, 2.0, 0], [12.0, 0, 0]]
    return Atoms("CuAlCuAl", positions=positions)


STRUCTURES = {
    "dimer_in_box": (dimer_in_box, 4.0),
    "dimer_aperiodic": (lambda: Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]]), 4.0),
    "two_atom_cell_with_self_images": (
        lambda: bulk("Cu", "bcc", a=2.9, cubic=True),
        6.0,
    ),
    "fcc_four_atoms": (fcc_cubic, 6.0),
    "rattled_eight_atoms_two_elements": (rattled_eight_atoms, 5.0),
    "thin_slab_partial_pbc": (thin_slab, 6.0),
    "slab_with_wide_vacuum": (lambda: fcc111("Cu", size=(2, 2, 3), a=A_CU, vacuum=6.0), 4.0),
    "cluster_with_isolated_atom": (isolated_atom_cluster, 4.0),
}


@pytest.mark.parametrize("name", sorted(STRUCTURES))
def test_builder_edges_match_the_brute_force_oracle(name: str) -> None:
    make, cutoff = STRUCTURES[name]
    _, _, expected, _ = check_against_oracle(make(), cutoff)
    assert len(expected) > 0 or name == "dimer_in_box"


def test_fcc_edge_count_is_the_hand_counted_shell_sum() -> None:
    """12 + 6 + 24 + 12 + 24 neighbours inside 6 A for Cu (a = 3.61 A), four atoms: 312 edges."""
    _, data = run_builder(fcc_cubic(), 6.0)
    assert int(data[constants.N_NEIGHBORS_REAL]) == 4 * 78
    assert np.bincount(data[constants.BOND_IND_I]).tolist() == [78] * 4


def test_dimer_in_a_large_box_has_two_edges_of_the_bond_length() -> None:
    _, data = run_builder(dimer_in_box(), 4.0)
    assert data[constants.BOND_IND_I].tolist() == [0, 1]
    assert data[constants.BOND_IND_J].tolist() == [1, 0]
    np.testing.assert_allclose(
        data[constants.BOND_VECTOR], [[3.0, 0, 0], [-3.0, 0, 0]], rtol=VECTOR_TOL.rtol, atol=VECTOR_TOL.atol
    )


def test_two_atom_cell_keeps_the_self_image_edges() -> None:
    """bcc Cu, a = 2.9 A, two atoms: an atom is its own neighbour through the cell vectors."""
    _, _, expected, _ = check_against_oracle(bulk("Cu", "bcc", a=2.9, cubic=True), 6.0)
    assert any(p.i == p.j for p in expected)


def test_builder_edge_list_is_sorted_by_central_atom() -> None:
    _, data = run_builder(rattled_eight_atoms(), 5.0)
    assert (np.diff(data[constants.BOND_IND_I]) >= 0).all()


def test_unequal_neighbour_counts_are_reproduced() -> None:
    slab = fcc111("Cu", size=(2, 2, 3), a=A_CU, vacuum=6.0)
    _, data, expected, _ = check_against_oracle(slab, 4.0)
    counts = np.bincount([p.i for p in expected], minlength=len(slab))
    assert len(set(counts.tolist())) > 1
    assert np.bincount(data[constants.BOND_IND_I]).tolist() == counts.tolist()


def test_isolated_atom_gets_one_dummy_bond_and_no_real_edge() -> None:
    """Pins current behaviour: an atom with no neighbour is not left with zero bonds."""
    work, data, expected, isolated = check_against_oracle(isolated_atom_cluster(), 4.0)
    assert isolated == [3]
    on_isolated = data[constants.BOND_IND_I] == 3
    assert on_isolated.sum() == 1
    assert data[constants.BOND_IND_J][on_isolated].tolist() == [0]
    assert np.linalg.norm(data[constants.BOND_VECTOR][on_isolated]) > 4.0


def test_dummy_bond_vectors_for_several_isolated_atoms_in_an_orthorhombic_cell() -> None:
    """Pins current behaviour, not an endorsement: with k > 1 isolated atoms in a cell whose
    side lengths differ, the dummy vectors are the row sums of the cell plus the cutoff,
    reshaped across atoms, so atom 0 gets (24, 24, 34) and atom 1 (34, 54, 54), not (24, 34, 54)
    each. Both stay far outside the cutoff, so no real interaction is created."""
    atoms = Atoms(
        "Cu2",
        positions=[[0.0, 0, 0], [0, 0, 10.0]],
        cell=np.diag([20.0, 30.0, 50.0]),
        pbc=True,
    )
    _, data = run_builder(atoms, 4.0)
    assert data[constants.BOND_IND_I].tolist() == [0, 1]
    assert data[constants.BOND_IND_J].tolist() == [0, 0]
    np.testing.assert_allclose(
        data[constants.BOND_VECTOR], [[24.0, 24, 34], [34, 54, 54]], rtol=VECTOR_TOL.rtol
    )
    assert (np.linalg.norm(data[constants.BOND_VECTOR], axis=1) > 4.0).all()


def test_pair_exactly_at_the_cutoff_is_not_an_edge() -> None:
    atoms = Atoms("Cu2", positions=[[10.0, 10, 10], [14.0, 10, 10]], cell=25.0 * np.eye(3), pbc=True)
    _, data, expected, isolated = check_against_oracle(atoms, 4.0)
    assert expected == []
    assert isolated == [0, 1]


def test_pair_just_inside_the_cutoff_is_an_edge() -> None:
    atoms = Atoms("Cu2", positions=[[10.0, 10, 10], [13.9995, 10, 10]], cell=25.0 * np.eye(3), pbc=True)
    _, _, expected, _ = check_against_oracle(atoms, 4.0)
    assert len(expected) == 2


def test_thin_slab_edges_include_the_periodic_images_along_the_open_direction() -> None:
    """The input is open along z; the builder makes it periodic, so 1 A of vacuum does not
    separate the slab from its images (936 edges, against 600 for a true slab) at 6 A."""
    slab = thin_slab()
    _, data, expected, _ = check_against_oracle(slab, 6.0)
    open_z = brute_force_pairs(
        slab.positions, slab.cell.array, slab.get_chemical_symbols(), 6.0, periodic=(True, True, False)
    )
    assert len(expected) == 936
    assert len(open_z) == 600
    assert int(data[constants.N_NEIGHBORS_REAL]) == len(expected)


def test_output_dtypes_and_scalars() -> None:
    work, data = run_builder(rattled_eight_atoms(), 5.0)
    int_keys = [
        constants.ATOMIC_MU_I,
        constants.BOND_MU_I,
        constants.BOND_MU_J,
        constants.BOND_IND_I,
        constants.BOND_IND_J,
        constants.N_ATOMS_BATCH_REAL,
        constants.N_STRUCTURES_BATCH_REAL,
        constants.N_NEIGHBORS_REAL,
    ]
    assert all(data[k].dtype == np.int32 for k in int_keys)
    assert data[constants.BOND_VECTOR].dtype == np.float64
    assert int(data[constants.N_ATOMS_BATCH_REAL]) == 8
    assert int(data[constants.N_STRUCTURES_BATCH_REAL]) == 1
    symbols = work.get_chemical_symbols()
    assert data[constants.ATOMIC_MU_I].tolist() == [ELEMENTS[s] for s in symbols]
    assert data[constants.BOND_MU_I].tolist() == [
        ELEMENTS[symbols[i]] for i in data[constants.BOND_IND_I]
    ]
    assert data[constants.BOND_MU_J].tolist() == [
        ELEMENTS[symbols[j]] for j in data[constants.BOND_IND_J]
    ]


# ---- per-pair cutoffs (cutoff_dict -> process_cutoff_dict) ----

CUTOFF_DICT = {"CuCu": 3.2, "CuAl": 4.2, "AlAl": 4.8}
CUTOFF_TABLE = {
    ("Cu", "Cu"): 3.2,
    ("Cu", "Al"): 4.2,
    ("Al", "Cu"): 4.2,
    ("Al", "Al"): 4.8,
}


def test_cutoff_dict_edges_match_the_oracle() -> None:
    _, _, expected, _ = check_against_oracle(
        rattled_eight_atoms(), 3.0, CUTOFF_TABLE, cutoff_dict=CUTOFF_DICT
    )
    pairs = {(p.i % 2, p.j % 2) for p in expected}
    assert pairs == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_cutoff_dict_equals_a_strict_post_filter_of_the_largest_cutoff() -> None:
    """Builder with the largest cutoff everywhere, filtered by ``d < rc(pair)``, equals the
    builder with the cutoff dictionary."""
    atoms = rattled_eight_atoms()
    work, wide = run_builder(atoms, 4.8)
    _, with_dict = run_builder(atoms, 3.0, cutoff_dict=CUTOFF_DICT)
    symbols = work.get_chemical_symbols()
    rc = np.array(
        [
            CUTOFF_TABLE[symbols[i], symbols[j]]
            for i, j in zip(wide[constants.BOND_IND_I], wide[constants.BOND_IND_J])
        ]
    )
    keep = np.linalg.norm(wide[constants.BOND_VECTOR], axis=1) < rc
    assert 0 < keep.sum() < len(keep)
    wide_edges = recover_edges(work, wide, keep)
    dict_edges = recover_edges(work, with_dict, np.ones(len(with_dict[constants.BOND_IND_I]), bool))
    assert sorted(wide_edges) == sorted(dict_edges)


def recover_edges(work, data, mask):
    ind_i = data[constants.BOND_IND_I][mask]
    ind_j = data[constants.BOND_IND_J][mask]
    shifts = recover_shifts(
        work.positions, work.cell.array, ind_i, ind_j, data[constants.BOND_VECTOR][mask]
    )
    return list(zip(ind_i.tolist(), ind_j.tolist(), shifts))


def test_pair_at_a_pair_cutoff_is_excluded_and_just_inside_is_kept() -> None:
    """Cu-Al cutoff 4.25 A (exact in binary, so the separation 14.25 - 10 is exactly 4.25)."""
    cutoff_dict = {"CuCu": 3.25, "CuAl": 4.25, "AlAl": 4.75}
    table = {("Cu", "Cu"): 3.25, ("Cu", "Al"): 4.25, ("Al", "Cu"): 4.25, ("Al", "Al"): 4.75}
    cell = 25.0 * np.eye(3)
    on_boundary = Atoms("CuAl", positions=[[10.0, 10, 10], [14.25, 10, 10]], cell=cell, pbc=True)
    inside = Atoms("CuAl", positions=[[10.0, 10, 10], [14.24, 10, 10]], cell=cell, pbc=True)
    _, _, on_edges, _ = check_against_oracle(on_boundary, 3.0, table, cutoff_dict=cutoff_dict)
    _, _, in_edges, _ = check_against_oracle(inside, 3.0, table, cutoff_dict=cutoff_dict)
    assert on_edges == []
    assert len(in_edges) == 2


def test_pairs_missing_from_the_cutoff_dict_use_the_global_cutoff() -> None:
    """Cu-Cu has 5.0 A, larger than the global 4.0 A, so the largest cutoff differs from the
    fallback: Cu-Al and Al-Al must use 4.0, not 5.0."""
    table = {("Cu", "Cu"): 5.0}
    _, _, expected, _ = check_against_oracle(
        rattled_eight_atoms(), 4.0, table, cutoff_dict={"CuCu": 5.0}
    )
    assert {(p.i % 2, p.j % 2) for p in expected} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    _, wide = run_builder(rattled_eight_atoms(), 5.0)
    assert len(expected) < int(wide[constants.N_NEIGHBORS_REAL])


def test_aperiodic_box_is_sized_by_the_largest_pair_cutoff() -> None:
    """An Al-Al dimer 1 A apart with Al-Al cutoff 6 A and every other cutoff 2 A: the box the
    builder makes must exceed the largest cutoff, or periodic images would be neighbours."""
    atoms = Atoms("Al2", positions=[[0.0, 0, 0], [1.0, 0, 0]])
    table = {("Al", "Al"): 6.0, ("Cu", "Cu"): 2.0, ("Al", "Cu"): 2.0, ("Cu", "Al"): 2.0}
    _, _, expected, _ = check_against_oracle(
        atoms, 2.0, table, cutoff_dict={"AlAl": 6.0, "CuCu": 2.0, "AlCu": 2.0}
    )
    assert [(p.i, p.j, p.shift) for p in expected] == [(0, 1, (0, 0, 0)), (1, 0, (0, 0, 0))]


# ---- rigid motions on the real builder ----


def edge_set(work, data):
    ind_i, ind_j = data[constants.BOND_IND_I], data[constants.BOND_IND_J]
    shifts = recover_shifts(
        work.positions, work.cell.array, ind_i, ind_j, data[constants.BOND_VECTOR]
    )
    return set(zip(ind_i.tolist(), ind_j.tolist(), shifts))


def per_atom_sorted_distances(data, n_atoms):
    dist = np.linalg.norm(data[constants.BOND_VECTOR], axis=1)
    return [np.sort(dist[data[constants.BOND_IND_I] == i]) for i in range(n_atoms)]


def test_translation_leaves_the_edge_set_and_vectors_unchanged() -> None:
    atoms = rattled_eight_atoms()
    work0, data0 = run_builder(atoms, 5.0)
    moved = atoms.copy()
    moved.translate([0.7, -1.3, 2.9])
    work1, data1 = run_builder(moved, 5.0)
    assert edge_set(work0, data0) == edge_set(work1, data1)
    np.testing.assert_allclose(
        np.sort(data1[constants.BOND_VECTOR], axis=0),
        np.sort(data0[constants.BOND_VECTOR], axis=0),
        rtol=VECTOR_TOL.rtol,
        atol=VECTOR_TOL.atol,
    )


def test_translation_with_wrapping_leaves_the_distances_per_atom_unchanged() -> None:
    atoms = rattled_eight_atoms()
    _, data0 = run_builder(atoms, 5.0)
    moved = atoms.copy()
    moved.translate([2.7, -3.3, 5.9])
    moved.wrap()
    assert not np.allclose(moved.positions, atoms.positions)
    _, data1 = run_builder(moved, 5.0)
    for a, b in zip(per_atom_sorted_distances(data0, 8), per_atom_sorted_distances(data1, 8)):
        np.testing.assert_allclose(a, b, rtol=VECTOR_TOL.rtol, atol=VECTOR_TOL.atol)


def test_permuting_the_atoms_relabels_the_edges() -> None:
    atoms = rattled_eight_atoms()
    work0, data0 = run_builder(atoms, 5.0)
    order = np.random.default_rng(2).permutation(8)  # new atom k is old atom order[k]
    new_index = np.argsort(order)  # old atom a is new atom new_index[a]
    permuted = atoms[order.tolist()]
    work1, data1 = run_builder(permuted, 5.0)
    relabelled = {(int(new_index[i]), int(new_index[j]), s) for i, j, s in edge_set(work0, data0)}
    assert relabelled == edge_set(work1, data1)
    assert data1[constants.ATOMIC_MU_I].tolist() == [
        ELEMENTS[atoms.get_chemical_symbols()[k]] for k in order
    ]


def test_rotating_the_cell_leaves_edges_and_distances_unchanged() -> None:
    atoms = rattled_eight_atoms()
    work0, data0 = run_builder(atoms, 5.0)
    rotated = atoms.copy()
    rotated.rotate(37.0, (1.0, 2.0, 3.0), rotate_cell=True)
    assert not np.allclose(rotated.cell.array, atoms.cell.array)
    work1, data1 = run_builder(rotated, 5.0)
    assert edge_set(work0, data0) == edge_set(work1, data1)
    by_key0 = edges_with_vectors(work0, data0, [])
    by_key1 = edges_with_vectors(work1, data1, [])
    for key, vector in by_key0.items():
        assert np.linalg.norm(by_key1[key]) == pytest.approx(np.linalg.norm(vector), rel=1e-12)
    check_against_oracle(rotated, 5.0)


# ---- enforce_pbc as the builder sees it: pinned behaviour (see tests/test_utils.py) ----


def test_builder_edits_the_callers_atoms_in_place() -> None:
    """Pins current behaviour, a hazard for stress and any reuse of the structure: an aperiodic
    input leaves the builder periodic, with a new cell and translated positions."""
    atoms = Atoms("Cu2", positions=[[0.0, 0, 0], [2.5, 0, 0]])
    GeometricalDataBuilder(ELEMENTS, cutoff=4.0).extract_from_ase_atoms(atoms)
    assert atoms.pbc.tolist() == [True, True, True]
    assert atoms.cell.array.any()
    assert atoms.positions[0, 0] != 0.0


def test_builder_makes_a_slab_periodic_along_the_open_direction() -> None:
    """Pins current behaviour, not an endorsement: partial periodic boundaries are overridden,
    so a thin slab sees its own images across 1 A of vacuum (936 edges, not the 600 of the slab)."""
    slab = thin_slab()
    work, data = run_builder(slab, 6.0)
    assert work.pbc.tolist() == [True, True, True]
    assert int(data[constants.N_NEIGHBORS_REAL]) == 936
    assert 600 == len(
        brute_force_pairs(
            slab.positions,
            slab.cell.array,
            slab.get_chemical_symbols(),
            6.0,
            periodic=(True, True, False),
        )
    )


@pytest.mark.parametrize("pbc", [False, [True, True, False], True])
def test_zero_third_cell_vector_stops_the_pair_search(pbc) -> None:
    """Pins current behaviour, not an endorsement: with ``pbc`` not all False, the zero vector is
    kept and made periodic, and the neighbour search raises ``LinAlgError`` (singular cell)."""
    cell = np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 0.0]])
    atoms = Atoms("Cu2", positions=[[0.0, 0, 0], [2.5, 0, 0]], cell=cell, pbc=pbc)
    builder = GeometricalDataBuilder(ELEMENTS, cutoff=4.0)
    if pbc is False:
        data = builder.extract_from_ase_atoms(atoms)
        assert int(data[constants.N_NEIGHBORS_REAL]) == 2
    else:
        with pytest.raises(np.linalg.LinAlgError):
            builder.extract_from_ase_atoms(atoms)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_aperiodic_cluster_equals_the_isolated_cluster(seed: int) -> None:
    """Physical value: the box the builder invents is large enough that edges are exactly those
    of the cluster in open space (the oracle with no periodicity at all)."""
    rng = np.random.default_rng(seed)
    atoms = Atoms("Cu10", positions=rng.normal(scale=2.0, size=(10, 3)))
    work, data = run_builder(atoms, 4.0)
    expected = brute_force_pairs(
        work.positions, work.cell.array, work.get_chemical_symbols(), 4.0, periodic=(False,) * 3
    )
    isolated = sorted(set(range(10)) - {p.i for p in expected})
    got = edges_with_vectors(work, data, isolated)
    assert sorted(got) == [(p.i, p.j, p.shift) for p in expected]
    assert len(expected) > 10
