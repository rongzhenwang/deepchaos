from scipy.stats import norm
from src.theory import fast_integral

tolerance = 1e-5

def test_1d_fast_integral():
    #Test A: Gaussian PDF
    integrand = norm.pdf
    zmin, zmax, dz = -10, 10, 0.05
    calc_integral = fast_integral(integrand, zmin=zmin, zmax=zmax, dz=dz, ndim=1)
    true_integral = 1.0
    assert abs(true_integral - calc_integral) < tolerance
    abs(true_integral - calc_integral) < tolerance, f"1D Gaussian PDF integral test fails"

    # Test B: polynomial function x^2
    integrand = lambda x: x**2
    zmin, zmax, dz = -2, 2, 0.001
    calc_integral = fast_integral(integrand, zmin=zmin, zmax=zmax, dz=dz, ndim=1)
    true_integral = 16.0 / 3.0
    assert abs(true_integral - calc_integral) < tolerance, f"1D Poly integral test fails"
    
    print('Passed test_1d_fast_integral()')

def run_all():   
    test_1d_fast_integral()