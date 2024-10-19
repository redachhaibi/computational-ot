import jax.numpy as jnp

def _log_regularise( min, a, K, u, epsilon ):
    """

    Parameters:
    -----------
        min : ndarray
              Row min of the cost function, that is, - epsilon * log( k ) minus the potential f.
        a : ndarray, shape (n,)
                The probability histogram of the sample of size n.
        K : ndarray, shape (n,m)
              The Gibb's kernel.
        u : ndarray, shape (n,1)
            The initial left marginal of the coupling.
        epsilon : float
                  The regularization parameter.

    Returns:
    --------
        _type_: _description_
    """
    return jnp.log( jnp.sum( a[:,None] * u[:,None] * K * jnp.exp( min/epsilon )[None,:], 0 ) )

def _objectivefunction( a, b, epsilon, x, K ):
    """

    Parameters:
    -----------
        a : _description_
        b : _description_
        epsilon : _description_
        x : _description_
        K : _description_

    Returns:
    --------
        _type_: _description_
    """
    a_ = a.reshape( a.shape[0],)
    u = jnp.exp( x/epsilon ).reshape( x.shape[0], )
    min_x = jnp.min( - epsilon * jnp.log(K) - x, 0 )
    y = - epsilon * _log_regularise( min_x, a_, K, u, epsilon ) + min_x[None,:]
    return jnp.dot( x.T, a ) + jnp.dot( y, b )   
    
def _computegradientf( a, b, u, K, v, min_f, epsilon ):
    """

    Parameters:
    -----------
        a : _description_
        b : _description_
        u : _description_
        K : _description_
        v : _description_
        min_f : _description_
        epsilon : _description_

    Returns:
    --------
        _type_: _description_
    """
    a_ = a.reshape( a.shape[0], )
    b_ = b.reshape( b.shape[0], )
    gradient = a - jnp.sum( a_[:,None] * u[:,None] * K * v[None,:] * jnp.exp( min_f/epsilon )[None,:] * b_[None,:], 1 ).reshape( a.shape[0], -1 )
    return gradient


def _wolfe1( f, c, a, b, K, epsilon, alpha, p, slope, rho ):#Armijo Condition
    """

    Parameters:
    -----------
        f : _description_
        c : _description_
        a : _description_
        b : _description_
        K : _description_
        epsilon : _description_
        alpha : _description_
        p : _description_
        slope : _description_
        rho : _description_

    Returns:
    --------
        _type_: _description_
    """
    reduction_count = 0       
    while True:   
        condition = _objectivefunction( a, b, epsilon, f + alpha * p, K ) < _objectivefunction( a, b, epsilon, f, K ) + c * alpha * slope
        if condition or jnp.isnan( _objectivefunction( a, b, epsilon, f + alpha * p, K ) ) :
            alpha = rho * alpha                                                     
            reduction_count += 1
        else:
            break
    return alpha

def _update( K, a, b, f, epsilon, rho, c, tol = 1e-12, maxiter = 50 ):
    """

    Parameters:
    -----------
        K : _description_
        a : _description_
        b : _description_
        f : _description_
        epsilon : _description_
        rho : _description_
        c : _description_
        tol : _description_. Defaults to 1e-12.
        maxiter : _description_. Defaults to 50.
        debug : _description_. Defaults to False.

    Returns:
    --------    
        _type_: _description_
    """
    a_ = a.reshape( a.shape[0], )
    b_ = b.reshape( b.shape[0], )
    min_f = jnp.min( - epsilon * jnp.log( K ) - f, 0 )
    u = jnp.exp( f/epsilon ).reshape( f.shape[0], )
    g = - epsilon * _log_regularise( min_f, a_, K, u, epsilon )
    v = jnp.exp( g/epsilon ).reshape( g.shape[0], )
    alpha_list = []
    err = []
    objvalues = [] 
    null_vector = jnp.hstack( jnp.ones( a.shape[0] ) )/jnp.sqrt( a.shape[0] )
    null_vector = jnp.reshape( null_vector, ( a.shape[0], 1 ) )
    reg_matrix = jnp.dot( null_vector, null_vector.T )
    i = 0
    while True: 
        # Compute gradient w.r.t f:
        grad_f = _computegradientf( a, b, u, K, v, min_f, epsilon )
        # Compute the Hessian:
        M = a_[:,None] * u[:,None] * K * v[None,:] * jnp.exp( min_f/epsilon )[None,:] * jnp.sqrt(b_)[None,:]
        Hessian = jnp.sum( M * jnp.sqrt(b_)[None,:], 1 )[:,None] * jnp.identity( a.shape[0] ) - jnp.dot( M, M.T ) 
        mean_eig = - jnp.mean( jnp.linalg.eigh( Hessian )[0] )
        Hessian = - Hessian/epsilon
        Hessian =  Hessian + mean_eig * ( reg_matrix )/epsilon
        # Compute solution of Ax = b:
        try:    
            p_k = - jnp.linalg.solve( Hessian, grad_f )
        except jnp.linalg.LinAlgError as e:
            print(f"An error occurred: {e}")
            return jnp.zeros(6)
        p_k = p_k - null_vector * jnp.dot( null_vector.flatten(), p_k.flatten() )
        # Wolfe condition 1: Armijo Condition:  
        slope = jnp.dot( p_k.T, grad_f )[0][0]
        alpha = 1
        alpha = _wolfe1( f, c, a, b, K, epsilon, alpha, p_k, slope, rho )
        alpha_list.append( alpha )
        # Update f and g:
        f = f + alpha * p_k
        min_f = jnp.min( - epsilon * jnp.log( K ) - f, 0 )
        u  = jnp.exp( f/epsilon ).reshape( f.shape[0], )
        g = - epsilon * _log_regularise( min_f, a_, K, u, epsilon )
        v  = jnp.exp( g/epsilon ).reshape( g.shape[0], )
        # Error computation:
        P  =  a_[:,None] * u[:,None] * K * v[None,:] * jnp.exp( min_f/epsilon )[None,:] * b_[None,:]
        err.append( jnp.linalg.norm( jnp.sum( P, 1 ) - a_, 1 ) )
        # Calculating objective function:
        value = _objectivefunction( a, b, epsilon, f, K )
        objvalues.append( value[0] )
        # Check error:
        if i < maxiter and ( err[-1] > tol ):
            i += 1
        else:   
            print( "Terminating after iteration: ",i )
            break
    return {
        "potential_f"       : f.reshape(a.shape[0],),
        "potential_g"       : g.reshape(b.shape[0],) + min_f,
        "error"             : err,
        "objectives"        : objvalues,
        "linesearch_steps"  : alpha_list,
        "Hessian"           : Hessian
    }

