import jax.numpy as jnp
from jax import jit

# Define functions outside of the class
@jit
def mina_u( a, H, epsilon ):
    """

    Parameters:
    -----------
        a : ndarray, shape (n,)
            The probability histogram of the sample of size n.
        H : ndarray, shape (n,m)
            It is the matrix obtained from C - f.
        epsilon :  float
                   The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
                   
    Returns:
    --------
        ndarray, shape (m,)
        The value of potential g obtained from the Schrodinger-bridge equation between the potentials f and g.
    """
    return - epsilon * jnp.log( jnp.sum( a[:, None] * jnp.exp( -H / epsilon ), axis = 0 ) )
@jit
def minb_u( b, H, epsilon ):
    """

    Parameters:
    -----------
        b : ndarray, shape (m,)
            The probability histogram of the sample of size m.
        H : ndarray, shape (n,m)
            It is the matrix obtained from C - g.
        epsilon :  float
                   The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
 
    Returns:
    --------
        ndarray, shape (n,)
        The value of potential f obtained from the Schrodinger-bridge equation between the potentials f and g.
    """
    return - epsilon * jnp.log( jnp.sum( b[None, :] * jnp.exp( -H / epsilon ), axis = 1 ) )
@jit
def mina( a, H, epsilon ):
    """

    Parameters:
    -----------
        a : ndarray, shape (n,)
            The probability histogram of the sample of size n.
        H : ndarray, shape (n,m)
            It is the matrix obtained from C - f.
        epsilon :  float
                   The regularization factor in the entropy regularized optimization setup of the optimal transport problem.

    Returns:
    --------
        ndarray, shape (m,)
        The value of potential g obtained from the Schrodinger-bridge equation between the potentials f and g along with log-exp regularization.

    """
    return mina_u( a, H - jnp.min( H, axis = 0 ), epsilon ) + jnp.min( H, axis = 0 )
@jit
def minb( b, H, epsilon ):
    """

    Parameters:
    -----------
        b : ndarray, shape (m,)
            The probability histogram of the sample of size m.
        H : ndarray, shape (n,m)
            It is the matrix obtained from C - g.
        epsilon :  float
                   The regularization factor in the entropy regularized optimization setup of the optimal transport problem.

    Returns:
    --------
        ndarray, shape (n,)
        The value of potential g obtained from the Schrodinger-bridge equation between the potentials f and g along with log-exp regularization.
    """
    return minb_u( b, H - jnp.min( H, axis = 1 )[:, None], epsilon ) + jnp.min( H, axis = 1 )

def update( a, b, C, epsilon, tol = 1e-12, niter = 500 ):
    """

    Parameters:
    -----------
        C : ndarray, shape (n,m), 
            n and m are the sizes of the samples from the two point clouds.
            It is the cost matrix between the sample points of the two point clouds.
        a : ndarray, shape (n,)
            The probability histogram of the sample of size n.
        b : ndarray, shape (m,)
            The probability histogram of the sample of size m.
        epsilon : float
                  The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
        tol :  float
               The tolerance limit for the error. Defaults to 1e-12.
        maxiter :  int 
                   The maximum iteration for the optimization algorithm. Defaults to 100.

    Returns:
    --------
        Returns a dictionary where the keys are strings and the values are ndarrays.
        The following are the keys of the dictionary and the descriptions of their values:
            potential_f : ndarray, shape (n,)
                          The optimal Kantorovich potential f.
            potential_g : ndarray, shape (m,)
                          The optimal Kantorovich potential g.
            error : ndarray, shape (k,), where k is the number of iterations
                    Errors observed over the iteration of the algorithm.
    """
    error = []
    f, g = a, b
    for i in range(niter):
        g = mina( a, C - f[:, None], epsilon )
        f = minb( b, C - g[None, :], epsilon ) 
        # generate the coupling
        P = a[:, None] * jnp.exp( ( f[:, None] + g[None, :] - C ) / epsilon ) * b[None, :]
        # check conservation of mass
        error.append( jnp.linalg.norm( jnp.sum( P, axis = 0 ) - b, 1 ) )
        if error[i] <= tol:
            print( "Terminating after iteration: ", i )
            break
    # end for
    if i + 1 >= niter:
        print( "Terminating after maximal number of iterations: ", niter )
    return {
        'error'       : error,
        'potential_f' : f + epsilon * jnp.log(a).reshape(a.shape[0],),
        'potential_g' : g + epsilon * jnp.log(b).reshape(b.shape[0],)  # Change of convention because of line (*)
    }