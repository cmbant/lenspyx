"""
Unit tests for the quadratic estimator functionality in lenspyx.

These tests verify that the quadratic estimator can correctly
reconstruct the lensing potential from simulated CMB maps.
"""
import unittest
import numpy as np
from lenspyx import synfast, get_geom
from lenspyx.utils import get_ffp10_cls
from lenspyx.utils_hp import gauss_beam, almxfl, alm2cl, alm_copy, synalm
from lenspyx.qest.qest import Qlms, OpFilt
from lenspyx.qest.ivfs import OpFilt


def copy_cls(cls, include=()):
    """Returns cls for the desired fields only"""
    ret = {}
    for k in cls:
        if k[0] in include and k[1] in include:
            ret[k] = np.copy(cls[k])
    return ret


class TestQuadraticEstimator(unittest.TestCase):
    """Test cases for the quadratic estimator functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across test methods."""
        # Parameters
        cls.lmax_unl = 2000  # Maximum multipole for unlensed fields
        cls.lmax_filt = 1500  # Maximum multipole for filtering
        cls.lmax_qlm = 300   # Maximum multipole for QE output
        cls.geom_info = ('thingauss', {'lmax': 3000, 'smax': 2})  # Geometry specification
        
        # Get Cls from FFP10 (Planck Full Focal Plane simulation 10)
        cls.cls_unl, cls.cls_len, _ = get_ffp10_cls(lmax=cls.lmax_unl)
        cls.geom = get_geom(cls.geom_info)
        
        # Define beam and noise properties (simple Gaussian beam and white noise)
        cls.beam = gauss_beam(5. / 180 / 60 * np.pi, lmax=cls.lmax_filt)  # 5 arcmin beam
        cls.inoise = {
            'tt': cls.beam ** 2 / (35. / 180 / 60 * np.pi) ** 2,  # 35 μK-arcmin for temperature
            'ee': cls.beam ** 2 / (55. / 180 / 60 * np.pi) ** 2,  # 55 μK-arcmin for polarization
            'bb': cls.beam ** 2 / (55. / 180 / 60 * np.pi) ** 2   # 55 μK-arcmin for polarization
        }
        cls.transfs = {f: np.ones(cls.lmax_filt + 1, dtype=float) for f in 'teb'}  # Transfer functions
        
        # Generate lensed CMB maps
        maps, (unl_alms, unl_lab) = synfast(cls.cls_unl, lmax=cls.lmax_unl, 
                                           geometry=cls.geom_info, verbose=False, alm=True)
        
        # Convert maps to harmonic space
        tlm = cls.geom.adjoint_synthesis(maps['T'], 0, cls.lmax_filt, cls.lmax_filt, 0).squeeze()
        eblm = cls.geom.adjoint_synthesis(maps['QU'], 2, cls.lmax_filt, cls.lmax_filt, 0)
        
        # Apply transfer functions
        almxfl(tlm, cls.transfs['t'], cls.lmax_filt, True)
        almxfl(eblm[0], cls.transfs['e'], cls.lmax_filt, True)
        almxfl(eblm[1], cls.transfs['b'], cls.lmax_filt, True)
        
        # Add instrumental noise
        tlm_noisy = tlm.copy() + synalm(1. / cls.inoise['tt'], cls.lmax_filt, cls.lmax_filt)
        elm_noisy = eblm[0].copy() + synalm(1. / cls.inoise['ee'], cls.lmax_filt, cls.lmax_filt)
        blm_noisy = eblm[1].copy() + synalm(1. / cls.inoise['bb'], cls.lmax_filt, cls.lmax_filt)
        cls.alms = {'t': tlm_noisy, 'e': elm_noisy, 'b': blm_noisy}
        
        # Get input lensing potential for comparison
        cls.plm_in = alm_copy(unl_alms[unl_lab.index('p')], cls.lmax_unl, cls.lmax_qlm, cls.lmax_qlm)

    def test_tt_estimator(self):
        """Test the temperature-only quadratic estimator."""
        # Create inverse-variance filtering object
        includes = ['t']
        cls_filt = copy_cls(self.cls_len, include=includes)
        filtr = OpFilt(cls_filt, self.transfs, self.inoise)
        
        # Create QE calculator object
        qlms_dd = Qlms(filtr, filtr, self.cls_len, self.lmax_qlm)
        
        # Calculate (unormalized) lensing potentials (gradient and curl)
        plm, _ = qlms_dd.get_qlms('ptt', self.alms, verbose=False)
        
        # Calculate estimator normalization
        rp, _ = qlms_dd.get_response('ptt', 'p', self.cls_len)
        
        # Apply normalization
        plm_norm = plm.copy()
        # Avoid division by zero for L=0,1
        rp_safe = rp.copy()
        rp_safe[:2] = 1.0
        almxfl(plm_norm, 1.0 / rp_safe, self.lmax_qlm, True)
        
        # Calculate correlation with input
        ls = np.arange(10, 100)  # Focus on a reasonable L range
        corr = alm2cl(plm_norm, self.plm_in, self.lmax_qlm, self.lmax_qlm, self.lmax_qlm)[ls]
        auto = alm2cl(plm_norm, plm_norm, self.lmax_qlm, self.lmax_qlm, self.lmax_qlm)[ls]
        input_cl = self.cls_unl['pp'][ls]
        
        # Calculate correlation coefficient
        corr_coeff = corr / np.sqrt(auto * input_cl)
        mean_corr = np.mean(corr_coeff)
        
        # The correlation should be positive and reasonably high
        self.assertGreater(mean_corr, 0.3, 
                          "TT estimator correlation with input is too low")

    def test_pol_estimator(self):
        """Test the polarization-only quadratic estimator."""
        # Create inverse-variance filtering object
        includes = ['e', 'b']
        cls_filt = copy_cls(self.cls_len, include=includes)
        filtr = OpFilt(cls_filt, self.transfs, self.inoise)
        
        # Create QE calculator object
        qlms_dd = Qlms(filtr, filtr, self.cls_len, self.lmax_qlm)
        
        # Calculate (unormalized) lensing potentials (gradient and curl)
        plm, _ = qlms_dd.get_qlms('p_p', self.alms, verbose=False)
        
        # Calculate estimator normalization
        rp, _ = qlms_dd.get_response('p_p', 'p', self.cls_len)
        
        # Apply normalization
        plm_norm = plm.copy()
        # Avoid division by zero for L=0,1
        rp_safe = rp.copy()
        rp_safe[:2] = 1.0
        almxfl(plm_norm, 1.0 / rp_safe, self.lmax_qlm, True)
        
        # Calculate correlation with input
        ls = np.arange(10, 100)  # Focus on a reasonable L range
        corr = alm2cl(plm_norm, self.plm_in, self.lmax_qlm, self.lmax_qlm, self.lmax_qlm)[ls]
        auto = alm2cl(plm_norm, plm_norm, self.lmax_qlm, self.lmax_qlm, self.lmax_qlm)[ls]
        input_cl = self.cls_unl['pp'][ls]
        
        # Calculate correlation coefficient
        corr_coeff = corr / np.sqrt(auto * input_cl)
        mean_corr = np.mean(corr_coeff)
        
        # The correlation should be positive
        self.assertGreater(mean_corr, 0.1, 
                          "Polarization estimator correlation with input is too low")

    def test_gmv_estimator(self):
        """Test the minimum-variance (GMV) quadratic estimator."""
        # Create inverse-variance filtering object
        includes = ['t', 'e', 'b']
        cls_filt = copy_cls(self.cls_len, include=includes)
        filtr = OpFilt(cls_filt, self.transfs, self.inoise)
        
        # Create QE calculator object
        qlms_dd = Qlms(filtr, filtr, self.cls_len, self.lmax_qlm)
        
        # Calculate (unormalized) lensing potentials (gradient and curl)
        plm, _ = qlms_dd.get_qlms('p', self.alms, verbose=False)
        
        # Calculate estimator normalization
        rp, _ = qlms_dd.get_response('p', 'p', self.cls_len)
        
        # Apply normalization
        plm_norm = plm.copy()
        # Avoid division by zero for L=0,1
        rp_safe = rp.copy()
        rp_safe[:2] = 1.0
        almxfl(plm_norm, 1.0 / rp_safe, self.lmax_qlm, True)
        
        # Calculate correlation with input
        ls = np.arange(10, 100)  # Focus on a reasonable L range
        corr = alm2cl(plm_norm, self.plm_in, self.lmax_qlm, self.lmax_qlm, self.lmax_qlm)[ls]
        auto = alm2cl(plm_norm, plm_norm, self.lmax_qlm, self.lmax_qlm, self.lmax_qlm)[ls]
        input_cl = self.cls_unl['pp'][ls]
        
        # Calculate correlation coefficient
        corr_coeff = corr / np.sqrt(auto * input_cl)
        mean_corr = np.mean(corr_coeff)
        
        # The correlation should be positive and higher than the other estimators
        self.assertGreater(mean_corr, 0.4, 
                          "GMV estimator correlation with input is too low")
        

if __name__ == '__main__':
    unittest.main()
