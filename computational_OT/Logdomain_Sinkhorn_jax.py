import jax.numpy as jnp
from jax import jit
@jit
def mina_u( a, H, epsilon ):
    """

    Parameters:
    -----------
        H : ndarray, shape (n,m)
            It is the matrix obtained from C - f.                   
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
        H : ndarray, shape (n,m)
            It is the matrix obtained from C - g.
 
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
        H : ndarray, shape (n,m)
            It is the matrix obtained from C - f.
    Returns:
    --------
        ndarray, shape (m,)
        The exp-log regularized g.
    """
    return mina_u( a, H - jnp.min( H, axis = 0 ), epsilon ) + jnp.min( H, axis = 0 )
@jit
def minb( b, H, epsilon ):
    """

    Parameters:
    -----------
        H : ndarray, shape (n,m)
            It is the matrix obtained from C - g.
    Returns:
    --------
        ndarray, shape (n,)
        The exp-log regularized f.
    """
    return minb_u( b, H - jnp.min( H, axis = 1 )[:, None], epsilon ) + jnp.min( H, axis = 1 )

def update( a, b, C, epsilon, tol = 1e-12, max_iterations = 500 ):
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
        tol : float
              The tolerance limit for the error. Defaults to 1e-12.
        max_iterations : int 
                         The maximum iteration for the optimization algorithm. Defaults to 100.

    Returns:
    --------
        Returns a dictionary where the keys are strings and the values are ndarrays.
        The following are the keys of the dictionary and the descriptions of their values:
            potential_f : ndarray, shape (n,)
                          The optimal Kantorovich potential f.
            potential_g : ndarray, shape (m,)
                          The optimal Kantorovich potential g.
            error : list
                    The list of errors observed when checking conservation of mass .
    """
    error = []
    f, g = a, b
    for i in range( max_iterations ):
        g = mina( a, C - f[:, None], epsilon )
        f = minb( b, C - g[None, :], epsilon ) 
        # Generate the coupling
        P = a[:, None] * jnp.exp( ( f[:, None] + g[None, :] - C ) / epsilon ) * b[None, :]
        # Check conservation of mass
        error.append(   jnp.linalg.norm( jnp.sum( P, axis = 1 ) - a, 1 ) 
                        +
                        jnp.linalg.norm( jnp.sum( P, axis = 0 ) - b, 1 ) )
        if error[i] <= tol:
            print( "Terminating after iteration: ", i )
            break
    # end for
    if i + 1 >= max_iterations:
        print( "Terminating after maximal number of iterations: ", max_iterations )
    return {
        'error'       : error,
        'potential_f' : f + epsilon * jnp.log(a).reshape(a.shape[0],),
        'potential_g' : g + epsilon * jnp.log(b).reshape(b.shape[0],) 
    }