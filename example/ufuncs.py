from . import ufuncs_example
import numpy as np


def absorbance(x1, x2, x3, x4, *args, **kwargs):
    return ufuncs_example.absorbance(x1, x2, x3, x4, *args, **kwargs)


def iph_with_ufunc(hv, prm_array, phi_N):
    r"""
    Compute the complex photocurrent using custom ufunc.

    Parameters
    ------------
    hv: array-like
        Light energy in eV.
    prm_array: array-like
        Parameter array
    phi_N: array-like
        Photon flux.

    Returns
    ---------
    iph: array-like
        Complex photocurrent.

    """
    nb_SC, cols = prm_array.shape

    iph_calc_complex = np.zeros(shape=hv.shape, dtype=np.complex128)

    for k in range(nb_SC):
        iph_calc_complex[:] += prm_array[k, 0]**prm_array[k, 6] * ufuncs_example.absorbance(hv, prm_array[k, 4], prm_array[k, 6], prm_array[k, 7]) * np.exp(1j * np.deg2rad(prm_array[k, 2]))
    iph_calc_complex *= phi_N

    return iph_calc_complex


absorbance.__doc__ = ufuncs_example.absorbance.__doc__
