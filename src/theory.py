import numpy as np
from scipy.stats import norm

def fast_integral(integrand, zmin, zmax, dz, ndim=1):
    """
    Estimates a multi-dimensional integral via a simple grid-based Riemann sum.

    This function calculates:
    Integral ≈ Σ [ integrand(z_grid_points) * (dz**ndim) ] over the N-dimensional space [zmin, zmax]^ndim.

    Parameters:
    ----------
    integrand : Callable[..., np.ndarray]
        A vectorized function that accepts `ndim` numpy arrays (the grid coordinates) as separate arguments.
        - Input: `ndim` arrays, each of shape (N_1, ..., N_ndim), in this case N_1 = ... = N_ndim = floor( (zmax - zmin) / dz). 
        - Output: Must return a numpy array whose first `ndim` dimensions are (N_1, ..., N_ndim). 
                  (Allows for extra batch dimensions, e.g., (N_1, ..., N_ndim, M_1, ...))
        
    zmin, zmax : float
        Integration bounds.
    
    dz : float
        Step size (grid spacing).
    
    ndim : int, optional
        Number of dimensions to integrate over, by default 1.

    Returns:
    -------
    np.ndarray
        The estimated result of the integral.
        If the `integrand` output contained extra batch dimensions (M_1, ...), those dimensions are preserved.
    """
    # For 1D,
    #    e.g., zmin, zmax, dz, ndim = -3, 3, 1.0, 1
    #          zs = [-3. -2. -1.  0.  1.  2.], type:<class 'numpy.ndarray'>
    #          axes = ((-3., -2., -1.,  0.,  1.,  2.),), type:<class 'tuple'>
    #          zgrid = [array([-3., -2., -1.,  0.,  1.,  2.])], type:<class 'list'>
    #          axes_to_sum = (0,)

    zs = np.arange(zmin, zmax, dz)  # 1D ndarray, shape: (N,)
    axes = (zs,) * ndim  # tuple, replicate grid for each dimension
    zgrid = np.meshgrid(*axes)  # list of ndim (N, N, ...) arrays
    out = integrand(*zgrid)  # ndarray of shape (N, ..., N, M_1, ...), integrand values on each position    

    axes_to_sum = tuple(np.arange(ndim))  # tuple, integration axes, e.g., ndim=2 -> (0, 1)
    integral_sum = out.sum(axis=axes_to_sum)  # ndarray of shape (M_1, ...), sum over integration axes
    volume_element = dz**ndim
    
    return integral_sum * volume_element  # Return estimated integral


def qmap(qin, weight_sigma=1.0, bias_sigma=0.0, nonlinearity=np.tanh, zmin=-10, zmax=10, dz=0.05):

    """ 
    Calculates the mean-field variance propagation "q-map" (q_out = V(q_in)).
    
    This function implements Eq. (3) from Poole (2016):
    V(q) = \sigma_w^2 * ∫ [ \phi(sqrt(q) * z)^2 * p(z) ] dz + \sigma_b^2
    
    Parameters:
    ----------
    qin : float or numpy.ndarray
        One or a batch of input variances (q_in).
    weight_sigma : float, optional
        Standard deviation of the weights (\sigma_w).
    bias_sigma : float, optional
        Standard deviation of the biases (\sigma_b).
    nonlinearity : Callable, optional
        The activation function, (\phi).
    zmin, zmax, dz : float, optional
        Integration parameters passed to fast_integral.

    Returns:
    -------
    numpy.ndarray
        The output V(q_in), same shape as qin.
    """

    # Ensure qin is an array to allow for broadcasting
    qin = np.atleast_1d(qin)  # Shape: (M,)
    def integrand(z):
        """
        Calculates the integrand: p(z) * \phi(sqrt(q_in) * z)^2
        
        This function is vectorized, using broadcasting to compute all (z, qin) combinations at once.

        Parameters:
        z : np.ndarray, Shape (N,)
            The 1D integration grid points from fast_integral.

        Returns:
        np.ndarray, Shape (N, M)
            The integrand evaluated at each (z, qin) combination.
            
        Broadcasting Details:
        ---------------------
        z[:, None]   -> Shape (N, 1)
        qin[None, :] -> Shape (1, M)
        
        (N, 1) * (1, M) -> broadcasts to (N, M)
        """
        #    e.g., z: [-3. -2. -1.  0.  1.  2.], type:<class 'numpy.ndarray'>, shape:(6,)
        #          qin: [0.   1.25 2.5  3.75 5.  ], type:<class 'numpy.ndarray'>, shape:(5,)
        #          z[:, None]: [[-3.] [-2.] [-1.] [ 0.] [ 1.] [ 2.]], type:<class 'numpy.ndarray'>, shape:(6, 1)
        #          qin[None, :]: [[0.   1.25 2.5  3.75 5.  ]], type:<class 'numpy.ndarray'>, shape:(1, 5)
        #          return, shape:(6, 5)
        return norm.pdf(z[:, None]) * (nonlinearity(np.sqrt(qin[None, :]) * z[:, None]))**2
    
    integral = fast_integral(integrand, zmin, zmax, dz, ndim=1)  # Calculate integration
    qout = weight_sigma**2 * integral + bias_sigma**2  # Calculate V(q_in)

    return qout