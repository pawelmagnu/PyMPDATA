"""
Created at 17.12.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from ..arakawa_c.scalar_field import ScalarField
from ..arakawa_c.vector_field import VectorField

import numpy as np
from ..utils import debug_flag
from .jit_flags import jit_flags

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba

eps = 1e-7


@numba.njit(**jit_flags)
def extremum_3arg(extremum: callable, a1: float, a2: float, a3: float):
    return extremum(extremum(a1, a2), a3)


@numba.njit(**jit_flags)
def extremum_4arg(extremum: callable, a1: float, a2: float, a3: float, a4: float):
    return extremum(extremum(extremum(a1, a2), a3), a4)

@numba.njit(**jit_flags)
def psi_max_1(psi: ScalarField.Impl):
    return np.maximum(psi.at(-1, 0), psi.at(1, 0))

@numba.njit(**jit_flags)
def psi_max_2(psi: ScalarField.Impl, psi_tmp: ScalarField.Impl):
    return np.maximum(psi.at(0, 0), psi_tmp.at(0, 0))

@numba.njit(**jit_flags)
def psi_min_1(psi: ScalarField.Impl):
    return np.minimum(psi.at(-1, 0), psi.at(1, 0))

@numba.njit(**jit_flags)
def psi_min_2(psi: ScalarField.Impl, psi_tmp: ScalarField.Impl):
    return np.minimum(psi.at(0, 0), psi_tmp.at(0, 0))


# set - TODO: function knows the op!!!
@numba.njit(**jit_flags)
def frac(nom: ScalarField.Impl, den: ScalarField.Impl):
    return nom.at(0, 0) / (den.at(0, 0) + eps)

# max - TODO
@numba.njit(**jit_flags)
def beta_up_nom_1(psi: ScalarField.Impl):
    return np.maximum(psi.at(-1, 0), psi.at(1, 0))

# set - TODO
@numba.njit(**jit_flags)
def beta_up_nom_2(psi: ScalarField.Impl, psi_max: ScalarField.Impl, psi_tmp: ScalarField.Impl, G: ScalarField.Impl):
    return G.at(0, 0) * (
        extremum_3arg(np.maximum, psi_max.at(0, 0), psi_tmp.at(0, 0), psi.at(0, 0)) - psi.at(0, 0)
    )

# sum
@numba.njit(**jit_flags)
def beta_up_den(flx: VectorField.Impl):
    return (
        np.maximum(flx.at(-.5, 0), 0)
        - np.minimum(flx.at(+.5, 0), 0)
    )


@numba.njit(**jit_flags)
def beta_dn(
        psi: ScalarField.Impl,
        psi_min: ScalarField.Impl,
        flx: VectorField.Impl,
        G: ScalarField.Impl
):
    # TODO: loops over dimensions
    assert psi.dimension == 1
    return (
       (
            psi.at(0, 0)
            - extremum_4arg(np.minimum, psi_min.at(0, 0), psi.at(-1, 0), psi.at(0, 0), psi.at(1, 0))
       ) * G.at(0, 0)
    ) / (
       np.maximum(flx.at(+.5, 0), 0)
       - np.minimum(flx.at(-.5, 0), 0)
       + eps
    )


def make_GC_mono():
    @numba.njit(**jit_flags)
    def fct_GC_mono(
        GC: VectorField.Impl,
        beta_up: ScalarField.Impl,
        beta_dn: ScalarField.Impl
    ):
        # TODO: this version is for iga or positive sign signal only
        result = GC.at(+.5, 0) * np.where(
            # if
            GC.at(+.5, 0) > 0,
            # then
            extremum_3arg(np.minimum, 1, beta_dn.at(0, 0), beta_up.at(1, 0)),
            # else
            extremum_3arg(np.minimum, 1, beta_up.at(0, 0), beta_dn.at(1, 0))
        )
        return result
    return fct_GC_mono