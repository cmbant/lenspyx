=================
Quadratic Estimator
=================

This section covers the quadratic estimator functionality in lenspyx, which is used for CMB lensing reconstruction.

Overview
--------

The quadratic estimator (QE) is a statistical technique that exploits the fact that lensing introduces specific correlations between different modes in the CMB. By measuring these correlations using pairs of filtered CMB fields, we can reconstruct the lensing potential.

The QE implementation in lenspyx consists of several key components:

1. **Inverse-variance filtering** (``lenspyx.qest.ivfs``): Filters the observed CMB maps to downweight noisy modes
2. **Quadratic estimator calculation** (``lenspyx.qest.qest``): Combines filtered maps to extract the lensing signal
3. **Response calculation** (``lenspyx.qest.qresp``): Computes the normalization required to convert the raw QE output to an unbiased estimate of the lensing potential
4. **Noise bias calculation** (``lenspyx.qest.nhl``): Computes the noise bias (N0, N1, etc.) in the power spectrum of the reconstructed lensing potential

Basic Usage
----------

Here's a basic example of how to use the quadratic estimator functionality:

.. code-block:: python

    import numpy as np
    from lenspyx import synfast, get_geom
    from lenspyx.utils import get_ffp10_cls
    from lenspyx.utils_hp import gauss_beam, almxfl, alm2cl, alm_copy
    from lenspyx.qest.qest import Qlms, OpFilt

    # Parameters
    lmax_unl = 3000  # Maximum multipole for unlensed fields
    lmax_filt = 2000  # Maximum multipole for filtering
    lmax_qlm = 500   # Maximum multipole for QE output
    geom_info = ('thingauss', {'lmax': 4000, 'smax': 2})  # Geometry specification
    
    # Get CMB power spectra
    cls_unl, cls_len, _ = get_ffp10_cls(lmax=lmax_unl)
    geom = get_geom(geom_info)
    
    # Define beam and noise properties
    beam = gauss_beam(5. / 180 / 60 * np.pi, lmax=lmax_filt)  # 5 arcmin beam
    inoise = {
        'tt': beam ** 2 / (35. / 180 / 60 * np.pi) ** 2,  # 35 μK-arcmin for temperature
        'ee': beam ** 2 / (55. / 180 / 60 * np.pi) ** 2,  # 55 μK-arcmin for polarization
        'bb': beam ** 2 / (55. / 180 / 60 * np.pi) ** 2   # 55 μK-arcmin for polarization
    }
    transfs = {f: np.ones(lmax_filt + 1, dtype=float) for f in 'teb'}  # Transfer functions
    
    # Generate lensed CMB maps
    maps, (unl_alms, unl_lab) = synfast(cls_unl, lmax=lmax_unl, geometry=geom_info, alm=True)
    
    # Convert maps to harmonic space
    tlm = geom.adjoint_synthesis(maps['T'], 0, lmax_filt, lmax_filt, 0).squeeze()
    eblm = geom.adjoint_synthesis(maps['QU'], 2, lmax_filt, lmax_filt, 0)
    
    # Apply transfer functions
    almxfl(tlm, transfs['t'], lmax_filt, True)
    almxfl(eblm[0], transfs['e'], lmax_filt, True)
    almxfl(eblm[1], transfs['b'], lmax_filt, True)
    
    # Prepare alms dictionary
    alms = {'t': tlm, 'e': eblm[0], 'b': eblm[1]}
    
    # Create inverse-variance filtering object
    filtr = OpFilt(cls_len, transfs, inoise)
    
    # Create QE calculator object
    qlms_dd = Qlms(filtr, filtr, cls_len, lmax_qlm)
    
    # Calculate QE (gradient and curl components)
    plm, olm = qlms_dd.get_qlms('ptt', alms, verbose=True)
    
    # Calculate response
    rp, ro = qlms_dd.get_response('ptt', 'p', cls_len)
    
    # Apply normalization
    almxfl(plm, 1.0 / rp, lmax_qlm, True)

Estimator Types
--------------

Lenspyx supports various types of quadratic estimators:

- **Temperature-only**: ``'ptt'``
- **Polarization-only**: ``'p_p'``
- **Minimum-variance**: ``'p'`` (combines T, E, and B)
- **Individual polarization pairs**: ``'pte'``, ``'ptb'``, ``'pee'``, ``'peb'``, ``'pbb'``

The choice of estimator depends on the available data and the specific requirements of the analysis.

Inverse-Variance Filtering
-------------------------

The ``OpFilt`` class in ``lenspyx.qest.ivfs`` implements inverse-variance filtering of CMB maps. This filtering is crucial for optimal lensing reconstruction as it downweights noisy modes.

.. code-block:: python

    from lenspyx.qest.ivfs import OpFilt
    
    # Create inverse-variance filter
    filtr = OpFilt(cls_cmb, transfs, noise)

The filtering operation is defined by:

.. math::

    \\tilde{X} = (S^{-1} + B^\\dagger N^{-1} B)^{-1} B^\\dagger N^{-1} d

where:

- :math:`\\tilde{X}` is the filtered field
- :math:`S` is the CMB signal covariance
- :math:`B` is the transfer function (e.g., beam)
- :math:`N` is the noise covariance
- :math:`d` is the observed data

Response Calculation
------------------

The response function converts the raw QE output to an unbiased estimate of the lensing potential:

.. math::

    \\hat{\\phi}(L) = R(L)^{-1} \\times \\text{raw\\_QE}(L)

The response can be calculated using:

.. code-block:: python

    rp, ro = qlms_dd.get_response('ptt', 'p', cls_cmb)

Noise Bias Calculation
--------------------

The noise bias in the power spectrum of the reconstructed lensing potential can be calculated using:

.. code-block:: python

    from lenspyx.qest.nhl import nhl
    
    # Calculate N0 bias
    n0 = nhl(filtr, filtr, 'ptt', 'ptt', cls_cmb, lmax_qlm)

The N0 bias is the disconnected (Gaussian) contribution to the power spectrum of the reconstructed lensing potential.

Advanced Usage
------------

For more advanced usage, including custom filtering schemes and noise bias subtraction, please refer to the example scripts in the ``lenspyx/tests/qes`` directory.

API Reference
-----------

.. automodule:: lenspyx.qest.qest
    :members: eval_qe, Qlms

.. automodule:: lenspyx.qest.ivfs
    :members: OpFilt

.. automodule:: lenspyx.qest.qresp
    :members:

.. automodule:: lenspyx.qest.nhl
    :members:
