import numpy as np

def fast_integral(integrand, zmin, zmax, dz, ndim=1):
    zs = np.arange(zmin, zmax, dz)  # ndarray, samples for a single dim
    axes = (zs,) * ndim  # tuple, containing ndim zs, e.g., ndim=2 -> axes = (zs, zs)
    zgrid = np.meshgrid(*axes)  # tuple, containing ndim len(zs)*len(zs) matrix, e.g., ndim=2 -> zgrid = (Z1, Z2), Z1 varies across axis0, Z2 varies across axis2
    out = integrand(*zgrid)  # ndarray, of size len(zs)*len(zs), integrand values on each position    

    axes_to_sum = tuple(np.arange(ndim))  # (0, 1, ... , ndim)
    integral_sum = out.sum(axis=axes_to_sum)
    volume_element = dz**ndim  # volume of a cube with interval dz
    
    return integral_sum * volume_element