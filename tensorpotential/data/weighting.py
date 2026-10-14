from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tensorpotential.constants import (
    DATA_ENERGY_WEIGHTS,
    DATA_FORCE_WEIGHTS,
    DATA_VIRIAL_WEIGHTS,
)
from tensorpotential.data.process_df import (
    E_CHULL_DIST_PER_ATOM,
    E_FORMATION_PER_ATOM,
    EFFECTIVE_ENERGY,
    E_CORRECTED_PER_ATOM_COLUMN,
    ENERGY_CORRECTED_COL,
    FORCES_COL,
    NUMBER_OF_ATOMS,
    ASE_ATOMS,
    compute_formation_energy,
    compute_compositions,
)

LOG_FMT = "%(asctime)s %(levelname).1s - %(message)s"
log = logging.getLogger()


def check_df_non_empty(df: pd.DataFrame):
    if len(df) == 0:
        raise RuntimeError(
            "Couldn't operate with empty dataset. Try to reduce filters and constraints"
        )


class EnergyBasedWeightingPolicy:

    def __init__(
        self,
        nfit=None,
        cutoff=None,
        DElow=1.0,
        DEup=10.0,
        DE=1.0,
        DF=1.0,
        wlow=None,  # 0.75,
        DFup=None,
        n_lower=None,
        n_upper=None,
        reftype="all",
        seed=42,
        energy="convex_hull",
    ):
        """

        :param nfit:
        :param cutoff:
        :param DElow:
        :param DEup:
        :param DE:
        :param DF:
        :param wlow:
        :param reftype:
        :param seed:
        :param energy: "convex" or "cohesive"
        """
        # #### Data selection and weighting
        # number of structures to be used in fit
        self.nfit = nfit
        # lower threshold: all structures below lower threshold are used in the fit (if fewer than nfit)
        self.DElow = DElow
        # upper threshold: structures between lower and upper threshold are selected randomly (after DElow structures)
        self.DEup = DEup
        # Delta E: energy offset in energy weights
        self.DE = DE
        # Delta F: force offset in force weights
        self.DF = DF

        # maximal forces projection
        self.DFup = DFup
        # relative fraction of structures below lower threshold in energy weights
        if wlow is not None and isinstance(wlow, str) and wlow.lower() == "none":
            wlow = None
        self.wlow = wlow
        # use all/bulk/cluster reference data
        self.reftype = reftype
        # random seed
        self.seed = seed

        self.cutoff = cutoff

        self.energy = energy  # cohesive or convex_hull
        self.n_lower = n_lower
        self.n_upper = n_upper

    def __str__(self):
        return (
            "EnergyBasedWeightingPolicy(nfit={nfit}, n_lower={n_lower}, n_upper={n_upper}, energy={energy},"
            + " DElow={DElow}, DEup={DEup}, DFup={DFup}, DE={DE}, "
            + "DF={DF}, wlow={wlow}, reftype={reftype},seed={seed})"
        ).format(
            nfit=self.nfit,
            n_lower=self.n_lower,
            n_upper=self.n_upper,
            cutoff=self.cutoff,
            DElow=self.DElow,
            DEup=self.DEup,
            DFup=self.DFup,
            DE=self.DE,
            DF=self.DF,
            wlow=self.wlow,
            energy=self.energy,
            reftype=self.reftype,
            seed=self.seed,
        )

    def generate_weights(self, df):
        """Select structures and attach energy, force and virial weights.

        The selection and the weights follow the tiers of ``DElow``/``DEup``; the
        returned frame holds the selected structures in shuffled order.
        """
        self._resolve_nfit(df)
        df = self._restrict_reftype(df)
        check_df_non_empty(df)

        if self.cutoff is not None:
            log.info(
                "EnergyBasedWeightingPolicy::cutoff is provided but will be ignored"
            )
        else:
            log.info(
                "No cutoff for EnergyBasedWeightingPolicy is provided, no structures outside cutoff that "
                + "will now be removed"
            )

        # #### structure selection
        self._set_effective_energy(df)
        df = self._drop_large_forces(df)
        check_df_non_empty(df)

        # remove high energy structures
        df = df[df[EFFECTIVE_ENERGY] < self.DEup]  # .reset_index(drop=True)
        check_df_non_empty(df)

        df, n_upper_selected = self._select_structures(df)
        self._warn_if_only_low_tier(df, n_upper_selected)
        self._set_energy_weights(df)
        self._set_force_weights(df)
        return df

    def _resolve_nfit(self, df):
        """Set ``nfit`` from the dataset size or from ``n_lower + n_upper``."""
        if self.nfit is None:
            if self.n_upper is None and self.n_lower is None:
                self.nfit = len(df)
                log.info("Set nfit to the dataset size {}".format(self.nfit))
            elif self.n_upper is not None and self.n_lower is not None:
                self.nfit = self.n_upper + self.n_lower
                log.info(
                    "Set nfit ({}) = n_lower ({}) + n_upper ({})".format(
                        self.nfit, self.n_lower, self.n_upper
                    )
                )
            else:  # nfit=None, one of n_upper or n_lower not None
                raise ValueError(
                    "nfit is None. Please provide both n_lower and n_upper"
                )
        elif self.n_upper is not None or self.n_lower is not None:
            raise ValueError("nfit is not None. No n_lower or n_upper is expected")

    def _restrict_reftype(self, df):
        """Keep bulk (periodic) or cluster (non-periodic) structures, or both."""
        if self.reftype == "bulk":
            log.info("Reducing to bulk data")
            df = df[df.pbc]
            log.info("Dataset size after reduction: {}".format(len(df)))
        elif self.reftype == "cluster":
            log.info("Reducing to cluster data")
            df = df[~df.pbc]
            log.info("Dataset size after reduction: {}".format(len(df)))
        else:
            log.info("Keeping bulk and cluster data")
        return df

    def _set_effective_energy(self, df):
        """Write ``EFFECTIVE_ENERGY`` (the energy above the chosen reference) into ``df``."""
        if self.energy == "convex_hull":
            log.info(
                "EnergyBasedWeightingPolicy: energy reference frame - convex hull distance (if possible)"
            )
            df[EFFECTIVE_ENERGY] = df[E_CHULL_DIST_PER_ATOM]  # already computed
        elif self.energy == "cohesive":
            log.info(
                "EnergyBasedWeightingPolicy: energy reference frame - cohesive energy"
            )
            emin = df[E_CORRECTED_PER_ATOM_COLUMN].min()
            df[EFFECTIVE_ENERGY] = df[E_CORRECTED_PER_ATOM_COLUMN] - emin
        elif self.energy == "zero_formation_energy":
            self._set_zero_formation_energy(df)
        else:
            raise ValueError(
                (
                    "Unknown EnergyBasedWeightingPolicy.energy={} settings. Possible values: convex_hull (default) or "
                    + "cohesive"
                ).format(self.energy)
            )

    @staticmethod
    def _set_zero_formation_energy(df):
        """``EFFECTIVE_ENERGY`` = max(formation energy, 0), shifted to a zero minimum."""
        log.info(
            "EnergyBasedWeightingPolicy: energy reference frame - zero_formation_energy (max of formation_energy and 0)"
        )
        elements = compute_compositions(df)
        df["NUMBER_OF_ATOMS"] = df["ase_atoms"].map(len)
        df[E_CORRECTED_PER_ATOM_COLUMN] = df[ENERGY_CORRECTED_COL] / df["NUMBER_OF_ATOMS"]
        compute_formation_energy(
            df,
            elements=elements,
            epa_gs_dict="mp_elemental_ref_entries",
            energy_per_atom_column=E_CORRECTED_PER_ATOM_COLUMN,
            verbose=True,
        )
        df[EFFECTIVE_ENERGY] = df[E_FORMATION_PER_ATOM]
        # set EFFECTIVE_ENERGY to zero if it is negative
        neg_form_energy_mask = df[E_FORMATION_PER_ATOM] < 0
        df.loc[neg_form_energy_mask, EFFECTIVE_ENERGY] = 0.0
        df[EFFECTIVE_ENERGY] -= df[EFFECTIVE_ENERGY].min()
        log.info(
            f"EnergyBasedWeightingPolicy: EFFECTIVE_ENERGY (shifted min -> 0) stats:\n {df[EFFECTIVE_ENERGY].describe()}"
        )

    def _drop_large_forces(self, df):
        """Remove the structures with an on-atom force above ``DFup`` (when set)."""
        if self.DFup is None:
            return df
        log.info(
            "Maximal allowed on-atom force vector length is DFup = {:.3f}".format(
                self.DFup
            )
        )
        fmax_column = df["forces"].map(lambda f: np.max(np.linalg.norm(f, axis=1)))
        size_before = len(df)
        df = df[fmax_column <= self.DFup]
        size_after = len(df)
        log.info(
            "{} structures with higher than dFup forces are removed. Current size: {} structures".format(
                size_before - size_after, size_after
            )
        )
        return df

    def _tier_masks(self, df):
        """Boolean masks of the structures below and above ``DElow``."""
        elow_mask = df[EFFECTIVE_ENERGY] < self.DElow
        eup_mask = df[EFFECTIVE_ENERGY] >= self.DElow
        return elow_mask, eup_mask

    def _select_structures(self, df):
        """Pick the structures of both tiers at random; return them and the upper-tier count."""
        elow_mask, eup_mask = self._tier_masks(df)
        nlow = elow_mask.sum()
        nup = eup_mask.sum()
        log.info("{} structures below DElow={} eV/atom".format(nlow, self.DElow))
        log.info(
            "{} structures between DElow={} eV/atom and DEup={} eV/atom".format(
                nup, self.DElow, self.DEup
            )
        )
        log.info("all other structures were removed")

        low_candidate_list = df.index[elow_mask]
        up_candidate_list = df.index[eup_mask]

        np.random.seed(self.seed)
        low_selected_list = self._select_lower_tier(low_candidate_list, nlow)
        up_selected_list = self._select_upper_tier(
            up_candidate_list, nup, len(low_selected_list)
        )

        takelist = np.hstack([low_selected_list, up_selected_list])
        np.random.shuffle(takelist)

        df = df.loc[takelist]  # .reset_index(drop=True)
        check_df_non_empty(df)
        return df, len(up_selected_list)

    def _select_lower_tier(self, candidates, nlow):
        """Take all candidates, or ``n_lower`` (``nfit`` when unset) of them at random."""
        n_max = self.n_lower if self.n_lower is not None else self.nfit
        if nlow <= n_max:
            return candidates
        return np.random.choice(candidates, n_max, replace=False)

    def _select_upper_tier(self, candidates, nup, n_low_selected):
        """Take all candidates, or ``n_upper`` (what ``nfit`` leaves) of them at random."""
        if self.n_upper is not None:
            if self.n_upper < nup:
                return np.random.choice(candidates, self.n_upper, replace=False)
            return candidates
        nremain = self.nfit - n_low_selected
        if nremain <= nup:
            return np.random.choice(candidates, nremain, replace=False)
        return candidates

    def _warn_if_only_low_tier(self, df, n_upper_selected):
        """Raise ``wlow`` to one when no structure of the upper tier was selected."""
        elow_mask, eup_mask = self._tier_masks(df)

        log.info("{} structures were selected".format(len(df)))

        assert elow_mask.sum() + eup_mask.sum() == len(df)
        if n_upper_selected == 0 and self.wlow is not None and self.wlow != 1.0:
            log.warning(
                (
                    "All structures were taken from low-tier, but relative weight of low-tier (wlow={}) "
                    + "is less than one. It will be adjusted to one"
                ).format(self.wlow)
            )
            self.wlow = 1.0

    def _set_energy_weights(self, df):
        """Write the normalised ``1 / (E + DE)**2`` energy weights, tier-rebalanced to ``wlow``."""
        log.info("Setting up energy weights")
        DE = abs(self.DE)

        df[DATA_ENERGY_WEIGHTS] = 1 / (df[EFFECTIVE_ENERGY] + DE) ** 2
        df[DATA_ENERGY_WEIGHTS] = (
            df[DATA_ENERGY_WEIGHTS] / df[DATA_ENERGY_WEIGHTS].sum()
        )

        e_weights_sum = df[DATA_ENERGY_WEIGHTS].sum()
        assert np.allclose(
            e_weights_sum, 1
        ), "Energy weights doesn't sum up to one: {}".format(e_weights_sum)
        self._rebalance_tiers(df)

    def _rebalance_tiers(self, df):
        """Scale the energy weights so the lower tier carries ``wlow`` of the total."""
        elow_mask, eup_mask = self._tier_masks(df)
        wlowcur = df.loc[elow_mask, DATA_ENERGY_WEIGHTS].sum()
        wupcur = df.loc[eup_mask, DATA_ENERGY_WEIGHTS].sum()

        log.info(
            "Current relative energy weights: {:.3f}/{:.3f}".format(wlowcur, wupcur)
        )
        if self.wlow is None:
            log.warning("wlow=None, no weights adjustment")
            return
        self.wlow = float(self.wlow)
        if not 1.0 > wlowcur > 0.0:
            log.warning("No weights adjustment possible")
            return
        log.info(
            "Will be adjusted to            : {:.3f}/{:.3f}".format(
                self.wlow, 1 - self.wlow
            )
        )
        flow = self.wlow / wlowcur
        fup = (1 - self.wlow) / (1 - wlowcur)

        df.loc[elow_mask, DATA_ENERGY_WEIGHTS] = (
            flow * df.loc[elow_mask, DATA_ENERGY_WEIGHTS]
        )
        df.loc[eup_mask, DATA_ENERGY_WEIGHTS] = (
            fup * df.loc[eup_mask, DATA_ENERGY_WEIGHTS]
        )
        energy_weights_sum = df[DATA_ENERGY_WEIGHTS].sum()
        assert np.allclose(
            energy_weights_sum, 1
        ), "Energy weights sum differs from one and equal to {}".format(
            energy_weights_sum
        )
        wlowcur = df.loc[elow_mask, DATA_ENERGY_WEIGHTS].sum()
        wupcur = df.loc[eup_mask, DATA_ENERGY_WEIGHTS].sum()
        log.info(
            "After adjustment: relative energy weights: {:.3f}/{:.3f}".format(
                wlowcur, wupcur
            )
        )
        assert np.allclose(wlowcur, self.wlow)
        assert np.allclose(wupcur, 1 - self.wlow)

    def _set_force_weights(self, df):
        """Write per-atom ``1 / (|F|**2 + DF)`` force weights scaled by the energy weights."""
        log.info("Setting up force weights")
        DF = abs(self.DF)
        df[FORCES_COL] = df[FORCES_COL].map(np.array)
        assert (df[FORCES_COL].map(len) == df[ASE_ATOMS].map(len)).all(), ValueError(
            "Number of atoms doesn't corresponds to shape of forces"
        )
        df[DATA_FORCE_WEIGHTS] = df[FORCES_COL].map(
            lambda forces: 1 / (np.sum(forces**2, axis=1) + DF)
        )
        assert (df[DATA_FORCE_WEIGHTS].map(len) == df[NUMBER_OF_ATOMS]).all()
        df[DATA_FORCE_WEIGHTS] = df[DATA_FORCE_WEIGHTS] * df[DATA_ENERGY_WEIGHTS]
        w_forces_norm = df[DATA_FORCE_WEIGHTS].map(sum).sum()
        df[DATA_FORCE_WEIGHTS] = df[DATA_FORCE_WEIGHTS] / w_forces_norm
        df[DATA_VIRIAL_WEIGHTS] = df[DATA_ENERGY_WEIGHTS]
        energy_weights_sum = df[DATA_ENERGY_WEIGHTS].sum()
        assert np.allclose(
            energy_weights_sum, 1
        ), "Energy weights sum differs from one and equal to {}".format(
            energy_weights_sum
        )
        forces_weights_sum = df[DATA_FORCE_WEIGHTS].map(sum).sum()
        assert np.allclose(
            forces_weights_sum, 1
        ), "Forces weights sum differs from one and equal to {}".format(
            forces_weights_sum
        )
