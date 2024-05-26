import jax
import jax.numpy as jnp
import jaxlinop
import time
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
    
def _log_regularise(min,a,K,u,epsilon):
    return jnp.log(jnp.sum(a[:,None]*u[:,None]*K*jnp.exp(min/epsilon)[None,:],0))

def _objectivefunction(a,b,epsilon,x,K):
    a_ = a.reshape(a.shape[0],)
    u = jnp.exp(x/epsilon).reshape(x.shape[0],)
    min_x = jnp.min(-epsilon*jnp.log(K)-x,0)
    y = -epsilon*_log_regularise(min_x,a_,K,u,epsilon)+min_x[None,:]
    return jnp.dot(x.T, a) + jnp.dot(y, b)   
    
def _computegradientf(a,b,u,K,v,min_f,epsilon):
    a_ = a.reshape(a.shape[0],)
    b_ = b.reshape(b.shape[0],)
    gradient = a-jnp.sum(a_[:,None]*u[:,None]*K*v[None,:]*jnp.exp(min_f/epsilon)[None,:]*b_[None,:], 1).reshape(a.shape[0],-1)
    return gradient


def _wolfe1(f, c, a, b, K, epsilon, alpha, p, slope, rho):#Armijo Condition
    """Backtracking""" 
    reduction_count = 0       
    while True:   
        condition = _objectivefunction( a, b, epsilon, f+alpha*p, K ) < _objectivefunction( a, b, epsilon, f, K )+c*alpha*slope
        if condition or jnp.isnan( _objectivefunction(a,b,epsilon, f+alpha*p, K )):
            alpha = rho*alpha                                                     
            reduction_count += 1
        else:
            break
    return alpha

def _precond_inversion_v0( unnormalized_Hessian, gradient, null_vector, precond_vectors, epsilon, iterative_inversion=-1, debug=False ):
    timings = []  
    start = time.time()
    # Record list of unwinding transformations on final result
    unwinding_transformations = []

    # Construct modified Hessian
    diag = 1/jnp.sqrt( jnp.diag(unnormalized_Hessian).flatten() )
    modified_Hessian = diag[:,None]*unnormalized_Hessian*diag[None,:]
    
    # Dummy variable to work on
    matrix = modified_Hessian
    # Preconditioning along null vector
    vector = null_vector
    vector = vector/diag
    vector = vector/jnp.linalg.norm( vector )
    vector = vector.reshape( (len(vector), 1) )
    matrix = matrix + jnp.dot( vector, vector.T )
    # Transformations
    gradient = diag[:,None]*gradient
    unwinding_transformations.append( [diag, lambda d,x : d[:,None]*x] )

   # Conditioning with other vectors
    k = len( precond_vectors )
    n = null_vector.shape[0]
    start1=time.time()
    for i in range( k ):
        vector = precond_vectors[i]
        value  = jnp.dot( jnp.dot( matrix, vector ), vector)
        vector = vector.reshape( (vector.shape[0], 1) ) 
        P_matrix = jnp.identity( n ) + ( 1/jnp.sqrt(value)-1 )*jnp.dot( vector, vector.T )
        # Transforms
        matrix = jnp.dot( P_matrix, jnp.dot( matrix, P_matrix ) )
        gradient = jnp.dot( P_matrix, gradient)
        unwinding_transformations.append( [P_matrix, lambda P,x : jnp.dot( P, x )] )
    # end for
    end = time.time()
    interval = 1e3*( end-start1 )
    timings.append( interval )
    print( "\n|--- Time required for preconditioning matrix formation: ", jnp.round( interval,5 ) ,"ms---|" )
    # end for

    start2 = time.time()
    # Debug
    if debug:
      eig, v = jnp.linalg.eigh( matrix )
      sorting_indices = jnp.argsort( eig )
      eig = eig[sorting_indices]
      v   = v[:, sorting_indices]
      #
      print( "List of smallest eigenvalues: ", eig[:5])
      print( "List of largest  eigenvalues: ", eig[-5:])
    end = time.time()
    interval = 1e3*( end-start2 )
    timings.append( interval )

    print( "|--- Time taken for the debug step: ", jnp.round( interval,5 ),"ms---|" )

    start3 = time.time()
    # Solve
    Hessian_stabilized = -matrix/epsilon
    if iterative_inversion >= 0:
      accumulator = gradient
      inverse = gradient
      delta   = -( matrix-jnp.identity( n ) ) # Unipotent part
      for i in range( iterative_inversion ):
        accumulator = jnp.dot( delta, accumulator )
        inverse = inverse + accumulator
      p_k = epsilon*inverse
    else:
      p_k = -jnp.linalg.solve( Hessian_stabilized, gradient )
    end = time.time()
    interval = 1e3*( end-start3 )
    timings.append( interval )
    print( "|--- Time taken to invert the linear system for p_k: ",jnp.round( interval,5 ),"ms---|" )

    start4 = time.time()
    # Unwind
    for transform in unwinding_transformations:
      P,f = transform
      p_k = f(P,p_k)
    end = time.time()
    interval = 1e3*( end-start4 )
    timings.append( interval )
    print( "|--- Time taken for unwinding: ",jnp.round( interval,5),"ms---|" )
    interval = 1e3*( end-start )
    timings.append( interval )    

    print( "|--- Time taken for the complete code block: ",jnp.round( interval,2),"ms---|\n" )
    return p_k,timings

def _precond_inversion_v1( unnormalized_Hessian, gradient, null_vector, precond_vectors, epsilon,  iterative_inversion=-1, debug=False ):
    timings = []
    start = time.time()
    # Record list of unwinding transformations on final result
    unwinding_transformations = []
    # Construct modified Hessian
    diag = 1/jnp.sqrt( jnp.diag( unnormalized_Hessian ).flatten() )
    modified_Hessian = diag[:,None]*unnormalized_Hessian*diag[None,:]
    # Dummy variable to work on
    matrix = modified_Hessian
    # Preconditioning along null vector
    vector = null_vector
    vector = vector/diag
    vector = vector/jnp.linalg.norm( vector )
    vector = vector.reshape( ( len( vector ), 1) )
    matrix = matrix + jnp.dot( vector, vector.T )
    # Transformations
    gradient = diag[:,None]*gradient
    unwinding_transformations.append( [diag, lambda d,x : d[:,None]*x] )
    end = time.time()
    interval = 1e3*( end-start )
    timings.append( interval )
    print( "\n|--- Time required for initial preconditioning: ", jnp.round( interval,5 ) ,"ms---|" )
    # Conditioning with other vectors
    k = len( precond_vectors )
    n = null_vector.shape[0]
    start1 = time.time()
    P_matrix = jnp.identity(n)
    for i in range(k):
      vector = precond_vectors[i]
      value  = jnp.dot( jnp.dot( matrix, vector ), vector )
      vector = vector.reshape( (vector.shape[0], 1) )
      P_matrix = P_matrix + ( 1/jnp.sqrt( value )-1 )*jnp.dot( vector, vector.T )
    # end for
    unwinding_transformations.append( [P_matrix, lambda P,x : jnp.dot(P, x)] )
    matrix = jnp.dot( P_matrix, jnp.dot(matrix, P_matrix) )
    gradient = jnp.dot( P_matrix, gradient)
    end = time.time()
    interval = 1e3*( end-start1 )
    timings.append( interval ) 
    print( "|--- Time required for preconditioning matrix formation: ", jnp.round( interval,5 ) ,"ms---|" )
    start2 = time.time()
    # Debug
    if debug:
      eig, v = jnp.linalg.eigh( matrix )
      sorting_indices = jnp.argsort(eig)
      eig = eig[sorting_indices]
      v   = v[:, sorting_indices]
      print( "List of smallest eigenvalues: ", eig[:5])
      print( "List of largest  eigenvalues: ", eig[-5:])
    end = time.time()
    interval = 1e3*(end-start2)
    timings.append( interval )
    print( "|--- Time taken for the debug step: ", jnp.round( interval,5 ),"ms---|" )
    start3 = time.time()
    # Solve
    Hessian_stabilized = -matrix/epsilon
    if iterative_inversion >= 0:
      accumulator = gradient
      inverse = gradient
      delta   = -(matrix-jnp.identity(n)) # Unipotent part
      
      for i in range(iterative_inversion):
        accumulator = jnp.dot(delta, accumulator)
        inverse = inverse + accumulator
      p_k = epsilon*inverse
    else:
      p_k = -jnp.linalg.solve( Hessian_stabilized, gradient)
    end = time.time()
    interval = 1e3*(end-start3)
    timings.append(interval)
    print( "|--- Time taken to invert the linear system for p_k: ",jnp.round( interval,5 ),"ms---|")
    start4 = time.time()
      # Unwind
    for transform in unwinding_transformations:
      P,f = transform
      p_k = f(P,p_k)
    end = time.time()
    interval = 1e3*(end-start4)
    timings.append(interval)
    print( "|--- Time taken for unwinding: ",jnp.round( interval,5 ),"ms---|" )
    interval = 1e3*(end-start)
    timings.append( interval )
    print( "|--- Time taken for the complete code block: ",jnp.round( interval,2 ),"ms---|\n" )
    return p_k,timings

def _precond_inversion_v2( unnormalized_Hessian, gradient, null_vector, precond_vectors, epsilon, iterative_inversion=-1, debug=False ):
    timings = []
    start = time.time()
    # Record list of unwinding transformations on final result
    unwinding_transformations = []
    # Construct modified Hessian
    diag = 1/jnp.sqrt( jnp.diag( unnormalized_Hessian ).flatten() )
    modified_Hessian = diag[:,None]*unnormalized_Hessian*diag[None,:] 
    # Dummy variable to work on
    matrix = modified_Hessian
    # Preconditioning along null vector
    vector = null_vector
    vector = vector/diag
    vector = vector/jnp.linalg.norm( vector )
    vector = vector.reshape( (len( vector ), 1) )
    matrix = matrix + jnp.dot( vector, vector.T )
    # Transformations
    gradient = diag[:,None]*gradient
    unwinding_transformations.append( [diag, lambda d,x : d[:,None]*x] )
    end = time.time()
    interval = 1e3*( end-start )
    timings.append( interval )
    print( "\n|--- Time required for initial preconditioning: ", jnp.round( interval,5 ) ,"ms---|" )
    # Conditioning with other vectors
    #  Naming conventions:
    #  y = Preconditioning vectors as a numpy matrix n by k
    #  matrix = our matrix A to precondition
    n = null_vector.shape[0]
    start1=time.time()
    y = jnp.array( precond_vectors ).T # Matrix of size n by k
    # Compute eigenvalues
    Ay = jnp.dot( matrix, y )
    eigenvalues = jnp.sum( y * Ay, axis=0 )
    # Compute P_matrix = id + y*diag(values)*y.T
    values = ( (1/jnp.sqrt(eigenvalues))-1 )    # Vector of size k
    z = y*values[None,:]
    P_matrix = jnp.identity( n ) + jnp.dot( z, y.T )
    # Old version
    # P_matrix = np.identity(n)
    # for i in range(k):
    #   vector = self.precond_vectors[i]
    #   #value  = np.dot( np.dot( matrix, vector ), vector)
    #   vector = vector.reshape( (vector.shape[0], 1) )
    #   P_matrix = P_matrix+ (1/np.sqrt(eigenvalues[i])-1)*np.dot( vector, vector.T)
    # # end for
    end = time.time()
    unwinding_transformations.append( [P_matrix, lambda P,x : jnp.dot(P, x)] )
    interval = 1e3*( end-start1 )
    timings.append( interval )
    print( "|--- Time required for preconditioning matrix formation: ", jnp.round( interval,5 ) ,"ms---|" )
    # Changing A=matrix to PAP
    start2 = time.time()
    # Old version -- O(n^3)
    # matrix = np.dot( P_matrix, np.dot(matrix, P_matrix) )
    # New version -- O(n^2)
    # PAP = (id + y*diag(values)*y.T)*A*(id + y*diag(values)*y.T)
    #     = A + A*y*diag(values)*y.T + y*diag(values)*(A*y).T + y*diag(values)*y.T*A*y*diag(values)*y.T
    B = jnp.dot( Ay, z.T )
    C = z @ jnp.dot( y.T, Ay ) @ z.T
    matrix = matrix + B + B.T + C
    gradient = jnp.dot( P_matrix, gradient )
    end = time.time()
    interval = 1e3*( end-start2 )
    timings.append( interval )
    print( "|--- Time required for changing A to PAP: ", jnp.round( interval,5 ) ,"ms---|" )
    start3 = time.time()
    # Debug
    if debug:
      eig, v = jnp.linalg.eigh( matrix )
      sorting_indices = jnp.argsort( eig )
      eig = eig[sorting_indices]
      v   = v[:, sorting_indices]
      #
      print( "List of smallest eigenvalues: ", eig[:5] )
      print( "List of largest  eigenvalues: ", eig[-5:] )
    end = time.time()
    interval = 1e3*( end-start3 )
    timings.append( interval )
    print( "|--- Time taken for the debug step: ", jnp.round( interval,5 ),"ms---|" )
    start4 = time.time()      
    # Solve
    Hessian_stabilized = -matrix/epsilon
    if iterative_inversion >= 0:
      accumulator = gradient
      inverse = gradient
      delta   = -(matrix-jnp.identity(n)) # Unipotent part
      for i in range(iterative_inversion):
        accumulator = jnp.dot(delta, accumulator)
        inverse = inverse + accumulator
      p_k = epsilon*inverse
    else:
      p_k = -jnp.linalg.solve( Hessian_stabilized, gradient )
    end = time.time()
    interval = 1e3*( end-start4 )
    timings.append( interval )
    print( "|--- Time taken to invert the linear system for p_k: ",jnp.round( interval,5 ),"ms---|" )
    start5 = time.time()
    # Unwind
    for transform in unwinding_transformations:
      P,f = transform
      p_k = f(P,p_k)
    end = time.time()
    interval = 1e3*( end-start5 )
    timings.append( interval )
    print( "|--- Time taken for unwinding: ",jnp.round( interval,5 ),"ms---|" )
    interval = 1e3*( end - start )
    timings.append( interval )
    print( "|--- Time taken for the complete code block: ",jnp.round( interval ,2),"ms---|\n" ) 
    return p_k,timings

def _precond_inversion_v3( unnormalized_Hessian, gradient, null_vector, precond_vectors, epsilon, iterative_inversion=-1, debug=False,optType = None ):
    timings = []
    start = time.time()
    # Record list of unwinding transformations on final result
    unwinding_transformations = []
    # Construct modified Hessian
    diag = 1/jnp.sqrt( jnp.diag(unnormalized_Hessian).flatten() )
    modified_Hessian = diag[:,None]*unnormalized_Hessian*diag[None,:]   
    # Dummy variable to work on
    matrix = modified_Hessian
    # Preconditioning along null vector
    vector = null_vector
    vector = vector/diag
    vector = vector/jnp.linalg.norm(vector)
    vector = vector.reshape( (len(vector), 1) )
    matrix = matrix + jnp.dot( vector, vector.T)
    # Transformations
    gradient = diag[:,None]*gradient
    unwinding_transformations.append( [diag, lambda d,x : d[:,None]*x])
    # Record timing
    end = time.time()
    interval = 1e3*(end-start)
    timings.append( interval )
    print("\n|--- Time required for initial preconditioning: ", jnp.round(interval,5) ,"ms---|")
    # Conditioning with other vectors
    #  Naming conventions:
    #  y = Preconditioning vectors as a numpy matrix n by k
    #  matrix = our matrix A to precondition
    n = null_vector.shape[0]
    start0 = time.time()
    y = jnp.array( precond_vectors ).T # Matrix of size n by k
    # Compute eigenvalues
    Ay = jnp.dot( matrix, y)
    eigenvalues = jnp.sum( y * Ay, axis=0 )
    # Compute P_matrix = id + y*diag(values)*y.T
    values = (1/jnp.sqrt(eigenvalues)-1)    # Vector of size k
    z = y*values[None,:]
    P_matrix = jnp.identity(n) + jnp.dot( z, y.T)
    # Done
    unwinding_transformations.append( [P_matrix, lambda P,x : jnp.dot(P, x)] )
    # Record timings
    end = time.time()
    interval = 1e3*(end-start0)
    timings.append( interval )
    print( "|--- Time required for preconditioning matrix formation: ", jnp.round(interval,5) ,"ms---|" )
    # Changing A=matrix to PAP
    start2 = time.time()
    # PAP = (id + y*diag(values)*y.T)*A*(id + y*diag(values)*y.T)
    #     = A + A*y*diag(values)*y.T + y*diag(values)*(A*y).T + y*diag(values)*y.T*A*y*diag(values)*y.T
    B = jnp.dot( Ay, z.T)
    C = z @ jnp.dot(y.T, Ay) @ z.T
    matrix = matrix + B + B.T + C
    gradient = jnp.dot( P_matrix, gradient)
    end=time.time()
    interval = 1e3*(end-start2)
    timings.append( interval )
    print( "|--- Time required for changing A to PAP: ", jnp.round( interval,5 ) ,"ms---|" )
    # Solve either iteratively using CG or exactly
    def mv(vector):
        return jnp.dot(m, vector)
    start3 = time.time()
    Hessian_stabilized = -matrix/epsilon
    if iterative_inversion >= 0:
      m = matrix
      if optType == 'cg':
        inverse, exit_code = jax.scipy.sparse.linalg.cg( mv, gradient, x0=gradient, maxiter=iterative_inversion, tol=1e-10 )
        print("...")
        print( "  --- CG exit code: ", exit_code)
      else:
        inverse, exit_code = jax.scipy.sparse.linalg.gmres( mv, gradient, x0=gradient, maxiter=iterative_inversion, tol=1e-10 )
        print( "  --- GMRES exit code: ", exit_code)
      p_k = epsilon*inverse
      p_k = p_k.reshape( (p_k.shape[0], 1) ) # For some reason, this outputs (n,) and the next line outputs (n,1)
    else:
      start = time.time()
      p_k = -jnp.linalg.solve( Hessian_stabilized, gradient)
      end = time.time()
      print("p_k inverse time stamp: ", 1e-3*(end-start))        
    end = time.time()
    interval = 1e3*(end-start3)
    timings.append(interval)
    print( "|--- Time taken to invert the linear system for p_k: ",jnp.round( interval,5),"ms---|" )
    start4 = time.time()
    # Unwind
    for transform in unwinding_transformations:
      P,f = transform
      p_k = f(P,p_k)
    end = time.time()
    interval = 1e3*(end-start4)
    timings.append( interval )
    print("|--- Time taken for unwinding: ",jnp.round( interval ,5 ) ,"ms---|")
    interval = 1e3*(end-start)
    timings.append( interval )
    print("|--- Time taken for the complete code block: ",jnp.round( interval,2),"ms---|\n")
    return p_k, timings
  
def _precond_inversion_v4( unnormalized_Hessian, gradient, null_vector, precond_vectors, epsilon, iterative_inversion=-1, debug = False,optType = None ):
    timings = []
    start = time.time()
    # Record list of unwinding transformations on final result
    unwinding_transformations = []
    # Construct modified Hessian
    diag = 1/jnp.sqrt( jnp.diag(unnormalized_Hessian).flatten() )
    modified_Hessian = diag[:,None]*unnormalized_Hessian*diag[None,:]
    # Dummy variable to work on
    matrix = modified_Hessian
    # Preconditioning along null vector
    vector = null_vector
    vector = vector/diag
    vector = vector/jnp.linalg.norm(vector)
    vector_E = vector
    # Transformations (Initial on gradient and final on result)
    gradient = diag[:,None]*gradient
    unwinding_transformations.append( lambda x : diag[:,None]*x )
    # Record timing
    end = time.time()
    interval = 1e3*(end-start)
    timings.append( interval )
    print("\n|--- Time required for initial preconditioning: ", jnp.round(interval,5) ,"ms---|")
    # Conditioning with other vectors
    #  Naming conventions:
    #  y = Preconditioning vectors as a numpy matrix n by k
    #  matrix = our matrix A to precondition
    #  We only form the data y and z such that
    #  P = id + z*y.T
    start0 = time.time()
    y = jnp.array( precond_vectors ).T # Matrix of size n by k
    # Compute eigenvalues
    Ay = jnp.dot( matrix, y)
    eigenvalues = jnp.sum( y * Ay, axis=0 )
    # Compute data for P = id + y*diag(values)*y.T
    values = (1/jnp.sqrt(eigenvalues)-1)    # Vector of size k
    z = y*values[None,:]
    # Record timings
    end = time.time()
    interval = 1e3*(end-start0)
    timings.append( interval )
    print( "|--- Time required for preconditioning matrix formation: ", jnp.round(interval,5) ,"ms---|" )
    # Changing A=matrix to PAP
    # Function mapping v to Pv
    # P = Id + z*y.T
    start2 = time.time()
    def _apply_P(vector):
      return vector + z @ ( y.T @ vector)
    # Function mapping v to P(A+E)Pv
    # A is matrix
    # E is vector_E*vector_E.T
    def preconditioned_map(vector):
      vector =  _apply_P(vector)
      vector =  jnp.dot(matrix, vector) + (vector_E*jnp.dot(vector_E, vector))[:,None] 
      vector = _apply_P(vector)
      return vector
    # Apply P
    # At beginning on gradient
    # At the end 
    gradient = _apply_P(gradient) 
    unwinding_transformations.append( lambda x : _apply_P(x) )
    end=time.time()
    interval = 1e3*(end-start2)
    timings.append( interval )    
    print( "|--- Time required for changing A to PAP: ", jnp.round( interval,5 ) ,"ms---|" )
    # Solve either iteratively using CG or exactly
    start3 = time.time()
    Hessian_stabilized = -matrix/epsilon
    if iterative_inversion >= 0:
      m = matrix
      A = jaxlinop.DenseLinearOperator(m)
      A.__matmul__ = preconditioned_map
      if optType == 'cg':
        inverse, exit_code = jax.scipy.sparse.linalg.cg(  A.__matmul__, gradient, x0 = gradient, maxiter = iterative_inversion, tol = 1e-10 )
        print( "  --- CG exit code: ", exit_code)
      else:
        inverse, exit_code = jax.scipy.sparse.linalg.gmres( A.__matmul__, gradient, x0 = gradient, maxiter = iterative_inversion, tol = 1e-10 )
        print( "  --- GMRES exit code: ", exit_code)
      p_k = epsilon*inverse
      p_k = p_k.reshape( (p_k.shape[0], 1) ) # For some reason, this outputs (n,) and the next line outputs (n,1)
    else:
      start = time.time()
      p_k = -jnp.linalg.solve( Hessian_stabilized, gradient)
      end = time.time()
      print("p_k inverse time stamp: ", 1e-3*(end-start))
    end = time.time()
    interval = 1e3*(end-start3)
    timings.append(interval)
    print( "|--- Time taken to invert the linear system for p_k: ",jnp.round( interval,5),"ms---|" )
    start4 = time.time()
    # Unwind
    for transform in unwinding_transformations:
      p_k = transform(p_k)
    end = time.time()
    interval = 1e3*(end-start4)
    timings.append( interval )
    print("|--- Time taken for unwinding: ",jnp.round( interval ,5 ) ,"ms---|")
    interval = 1e3*(end-start)
    timings.append( interval )

    print("|--- Time taken for the complete code block: ",jnp.round( interval,2),"ms---|\n")
    return p_k, timings
  
def _precond_inversion( unnormalized_Hessian, gradient, null_vector, precond_vectors, epsilon,   iterative_inversion = -1, debug = False, optType = None ):
    

    timings = []

    start = time.time()
    # Record list of unwinding transformations on final result
    unwinding_transformations = []

  
    # Construct modified Hessian
    diag = 1/jnp.sqrt( jnp.diag(unnormalized_Hessian).flatten() )
    modified_Hessian = diag[:,None]*unnormalized_Hessian*diag[None,:]
    
    # Dummy variable to work on
    matrix = modified_Hessian 
    # Preconditioning along null vector
    vector = null_vector
    vector = vector/diag
    vector = vector/jnp.linalg.norm(vector)
    vector_E = vector
    if iterative_inversion < 0:
      vector = vector.reshape( (len(vector), 1) )
      matrix = matrix + jnp.dot( vector, vector.T)
    # Transformations (Initial on gradient and final on result)
    gradient = diag[:,None]*gradient
    unwinding_transformations.append(lambda x : diag[:,None]*x)
    # Record timing
    end = time.time()
    interval = 1e3*(end-start)
    timings.append( interval )
    # Conditioning with other vectors
    #  Naming conventions:
    #  y = Preconditioning vectors as a numpy matrix n by k
    #  matrix = our matrix A to precondition
    #  We only form the data y and z such that
    #  P = id + z*y.T
    start0 = time.time()
    y = jnp.array( precond_vectors ).T # Matrix of size n by k
    # Compute eigenvalues
    Ay = jnp.dot( matrix, y)
    eigenvalues = jnp.sum( y * Ay, axis=0 )
    # Compute data for P = id + y*diag(values)*y.T
    values = (1/jnp.sqrt(eigenvalues)-1)    # Vector of size k
    z = y*values[None,:]
    # Record timings
    end = time.time()
    interval = 1e3*(end-start0)
    timings.append( interval )

    # Changing the A matrix to PAP
    start2 = time.time()

    # Function mapping v to Pv
    # P = Id + z*y.T
    def _apply_P(vector):
      return vector + z @ ( y.T @ vector)
    # Function mapping v to P(A+E)Pv
    # A is matrix
    # E is vector_E*vector_E.T
    def preconditioned_map(vector):
      vector =  _apply_P(vector)
      vector =  jnp.dot(matrix, vector) + (vector_E*jnp.dot(vector_E, vector))[:,None] 
      vector =  _apply_P(vector)
      return vector
    # Apply P
    # At beginning on gradient
    # At the end 
    gradient = _apply_P(gradient)
    unwinding_transformations.append( lambda x : _apply_P(x) )
    end      = time.time()
    interval = 1e3*(end-start2)
    timings.append( interval )  
    #
    # Solve either iteratively using CG or exactly
    start3 = time.time()
    if iterative_inversion >= 0:
      m  = matrix
      A = jaxlinop.DenseLinearOperator(m)
      A.__matmul__ = preconditioned_map
      if optType == 'cg':
        inverse, exit_code = jax.scipy.sparse.linalg.cg( A.__matmul__ , gradient, x0 = gradient, maxiter = iterative_inversion, tol = 1e-10 )
        # print( "  --- CG exit code: ", exit_code)
      else:
        inverse, exit_code = jax.scipy.sparse.linalg.gmres( A.__matmul__ , gradient, x0 = gradient, maxiter = iterative_inversion, tol = 1e-10 )
        # print( "  --- GMRES exit code: ", exit_code)
      p_k = epsilon*inverse
      p_k = p_k.reshape( (p_k.shape[0], 1) ) # For some reason, this outputs (n,) and the next line outputs (n,1)
    else:
      B = jnp.dot( Ay, z.T)
      C = z @ jnp.dot(y.T, Ay) @ z.T
      matrix = matrix + B + B.T + C
      Hessian_stabilized = -matrix/epsilon
      p_k = -jnp.linalg.solve( Hessian_stabilized, gradient)
    end = time.time() 
    interval = 1e3*(end-start3)
    timings.append(interval)
    start4 = time.time()
    # Unwind
    for transform in unwinding_transformations:
      p_k = transform(p_k)
    end = time.time()
    interval = 1e3*(end-start4)
    timings.append( interval )
    interval = 1e3*(end-start)
    timings.append( interval )
    # print("|--- Time taken for the complete code block: ",np.round( interval,2),"ms---|\n")
    return p_k, timings
  

def _update(K,a,b,f,g,epsilon,rho,c,null_vector,precond_vectors,tol = 1e-12, maxiter = 100, iterative_inversion = -1, version = 1, debug=False, optType='cg'):
    a_ = a.reshape(a.shape[0],)
    b_ = b.reshape(b.shape[0],)
    min_f = jnp.min(-epsilon*jnp.log(K)-f,0)
    u = jnp.exp(f/epsilon).reshape(f.shape[0],)
    g = -epsilon*_log_regularise(min_f,a_,K,u,epsilon)
    v = jnp.exp(g/epsilon).reshape(g.shape[0],)
    alpha_list = []
    err = []
    objvalues = [] 
    timing = []
    i = 0 
    while True:
        # Compute gradient w.r.t f:
        grad_f = _computegradientf(a,b,u,K,v,min_f,epsilon)
        # Compute the Hessian:
        M = a_[:,None]*u[:,None]*K*v[None,:]*jnp.exp(min_f/epsilon)[None,:]*jnp.sqrt(b_)[None,:]
        Hessian = jnp.sum(M*jnp.sqrt(b_)[None,:],1)[:,None]*jnp.identity(a.shape[0])-jnp.dot( M , M.T )
        # Compute solution of Ax = b:
        if version == 4:
          print("\n At iteration: ",i)
          p_k, temp = _precond_inversion_v4(  Hessian, 
                                              grad_f, 
                                              null_vector,
                                              precond_vectors,
                                              epsilon,
                                              iterative_inversion=iterative_inversion, 
                                              debug = debug , 
                                              optType = optType)
          timing.append(temp)
        elif version == 3:
          print("\n At iteration: ",i)
          p_k, temp = _precond_inversion_v3(  Hessian, 
                                              grad_f, 
                                              null_vector,
                                              precond_vectors,
                                              epsilon,   
                                              iterative_inversion=iterative_inversion, 
                                              debug = debug, 
                                              optType = optType )
          timing.append(temp)
        elif version == 2:
          print("\n At iteration: ",i)
          p_k,temp  = _precond_inversion_v2(  Hessian, 
                                              grad_f, 
                                              null_vector,
                                              precond_vectors,
                                              epsilon,   
                                              iterative_inversion=iterative_inversion, 
                                              debug = debug)
          timing.append(temp)
        elif version == 1:
          print("\n At iteration: ",i)
          p_k,temp  = _precond_inversion_v1(  Hessian, 
                                              grad_f, 
                                              null_vector,
                                              precond_vectors,
                                              epsilon,   
                                              iterative_inversion=iterative_inversion, 
                                              debug = debug
                                               )
          timing.append(temp)
        elif version == 0:
          print("\n At iteration: ",i)
          p_k,temp  = _precond_inversion_v0(  Hessian, 
                                              grad_f, 
                                              null_vector,
                                              precond_vectors,
                                              epsilon,   
                                              iterative_inversion=iterative_inversion, 
                                              debug = debug)
          timing.append(temp)
        else:
          p_k, temp = _precond_inversion(     Hessian, 
                                              grad_f, 
                                              null_vector,
                                              precond_vectors,
                                              epsilon,   
                                              iterative_inversion=iterative_inversion, 
                                              debug = debug, 
                                              optType = optType )
          timing.append(temp)

        start = time.time()
        # Wolfe condition 1( Armijo condition ):
        slope = jnp.dot(p_k.T, grad_f)[0][0]
        alpha = 1
        alpha = _wolfe1(f, c, a, b, K, epsilon, alpha, p_k, slope, rho)
        alpha_list.append(alpha)
        # Update f and g:
        f = f + alpha*p_k
        min_f = jnp.min(-epsilon*jnp.log(K)-f,0)
        u  = jnp.exp(f/epsilon).reshape(f.shape[0],)
        g = -epsilon*_log_regularise(min_f,a_,K,u,epsilon)
        v  = jnp.exp(g/epsilon).reshape(g.shape[0],)
        # Error computation:
        P  =  a_[:,None]*u[:,None]*K*v[None,:]*jnp.exp(min_f/epsilon)[None,:]*b_[None,:]
        err.append(jnp.linalg.norm(jnp.sum(P,1)-a_,1))
        # Calculating objective function:
        value = _objectivefunction(a, b,epsilon, f,K)
        objvalues.append(value[0])
        # Check error:
        if i< maxiter and (err[-1]>tol ):
            i+=1
        else:   
            print("Terminating after iteration: ",i)
            break            
        
    return {
        "potential_f"       : f.reshape(a.shape[0],),
        "potential_g"       : g.reshape(b.shape[0],)+min_f,
        "error"             : err,
        "objectives"        : objvalues,
        "linesearch_steps"  : alpha_list,
        "timings"           : timing,
        "Hessian"           : Hessian
    }


