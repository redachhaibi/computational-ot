import numpy as np
import scipy
import time

class semi_dual_dampedNewton_with_precodonditioning_np:
    def __init__( self,  C, a, b, f, epsilon, rho, c, null_vector, precond_vectors, exp_log = "True" ):
        """
        
        Parameters:
        -----------
            C : ndarray, shape (n,m), 
                It is the cost matrix between the points of the sample point clouds.
            a : ndarray, shape (n,)
                The probability histogram of the sample of size n.
            b : ndarray, shape (m,)
                The probability histogram of the sample of size m.
            f : ndarray, shape (n,) 
                The initial Kantorovich potential f.
            rho : float
                  Damping factor for the line search ascent step size.
            c : float
                Damping factor for the slope in the Armijo's condition.
            epsilon : float
                      The regularization parameter.
            null_vector : ndarray, shape (n,)
                          The null vector of the Hessian to be used for null vector preconditioning .
            precond_vectors : list of ndarrays, shape (n,)
                              The stack of preconditioning vectors of shape obtained from the Hessian at optimum obtained from the algorithm without any preconditioning,
                              that is, semi-dual damped Newton with only null vector preconditioning and exact inversion.
            exp_log : bool
                      Indicating to use exp-log regularization or not.
        """
        self.C = C
        self.a = a
        self.b = b
        self.f = f
        self.epsilon = epsilon
        self.rho = rho           
        self.c = c
        self.null_vector = null_vector
        self.precond_vectors = precond_vectors
        self.exp_log = exp_log
        self.alpha_list = []
        self.err = []
        self.objvalues = [] 
        self.timing = []
        self.out = []
        self.g = self._get_g( self.C - self.f[:,None] )# Shape: (m,)
        self.z = self.C - self.f[:,None] - self.g[None,:]# Shape: (n,m)

    def _objectivefunction( self, f ) :
        """ 
     
        Parameters:
        -----------
            f : ndarray, shape (n,)
                The input Kantorovich potential f.
                
        Returns: 
        --------
            Q_semi(f) : float
                        The value of semi-dual objective function obtained by evaluating the formula Q_semi(f) = < f, a > + < g( f, C, epsilon ), b >,
                        where g( f, C, epsilon ) denotes the value of Kantorovich potential g evaluated using the Schrodinger-bridge equations between f and g.
        """
        g = self._get_g( self.C - f[:,None] )
        Q_semi = np.dot( f, self.a ) + np.dot( g, self.b )
        return Q_semi

    def _computegradientf( self ):
        """ 
            Returns:
            --------
            ndarray, shape: (n,)
            The gradient of the objective function.
        """
        gradient = self.a * ( np.ones( self.a.shape[0] ) - np.sum( np.exp( - self.z/self.epsilon ) * self.b[None,:], axis = 1 ) )# Shape: (n,)
        return gradient
    
    def _get_g( self, H ):
      """ 
        Here we computing g.
        Parameters:
        -----------
            H : ndarray, shape (n,m)
                It is the matrix obtained from the difference C - f.
        Returns:
        --------
            ndarray, shape (m,)
            The value of potential g.
      """
      if self.exp_log :
        # With exp-log regularization
        g = self._explog_g( H )# Shape: (m,)
      else:
        # Without exp-log regularization
        g = self._g( H )# Shape: (m,)
      return g
    
    def _g( self, H ):
        """
        Here we compute the value of the potential g by using its Schrodinger-bridge relation with the potential f.
        Parameters:
        -----------
            H : ndarray, shape (n,m)
                It is the matrix obtained from the difference C - f.
        Returns:
        --------
            ndarray, shape (m,)
            The value of potential g.
        """
        return -self.epsilon * np.log( np.sum( self.a[:,None] * np.exp( -H/self.epsilon ), 0 ) )# Shape: (m,)
    
    def _explog_g( self, H ):
        """
        Here we incorporate the exp-log regularization method in the computation of the potential g.
        Parameters:
        -----------
            H : ndarray, shape (n,m)
                It is the matrix obtained from C - f.

        Returns:
        --------
            ndarray, shape (m,)
            The exp-log regularized value of potential g.

        """
        return self._g( H - np.min( H, 0 ) ) + np.min( H, 0 )# Shape: (m,)
   
    def _wolfe1( self, alpha, p, slope ):
        #Armijo Condition
        """
        Here we use the Armijo condition to decide the ascent step length for updating the potentials towards the ascent direction. 
        Parameters:
        -----------
            alpha : float  
                    The ascent step size.
            p : ndarray, shape (n,)
                The ascent direction.
            slope : float
                    It is the inner product of the gradient and p.

        Returns:
        --------
            alpha : float
                    The updated ascent step size.
        """
        reduction_count = 0     
        while True:   
            condition = self._objectivefunction( self.f + alpha * p ) < self._objectivefunction( self.f ) + self.c * alpha * slope
            if condition or np.isnan( self._objectivefunction( self.f + alpha * p  ) ):
                alpha = self.rho * alpha                                                     
                reduction_count += 1
            else:
                break
        return alpha
        
          
    def _precond_inversion_v0( self,  unnormalized_Hessian, gradient, iterative_inversion, debug ):
        """

        Parameters:
        -----------
            unnormalized_Hessian :  ndarray, shape (n,n)
                                    The unnormalized Hessian.
            gradient :  ndarray, shape (n,) 
                        The gradient of the objective function with respect to potential f.
            iterative_inversion : int
                                  The number of iterative inversions to be performed to obtain the inverse of the Hessian followed by obtaining the ascent direction.
                                  Defaults to -1, which indicates exact inversion.
            debug : bool
                    To observe the behaviour of the smallest and the largest eigenvalues of the Hessian. Defaults to False.

        Returns:
        --------
        Returns a tuple containing the optimal ascent direction vector p and the recorded timings of various steps of the algorithm. 
        The following are their descriptions:
            p : ndarray, shape (n,)
                The optimal ascent direction vector.
            timings : list
                      The list of timestamps recorded.
        """
        timings = []  
        start = time.time()
        # Record list of unwinding transformations on final result
        unwinding_transformations = []
        # Construct modified Hessian
        diag = 1/np.sqrt( np.diag(unnormalized_Hessian).flatten() )
        self.modified_Hessian = diag[:,None] * unnormalized_Hessian * diag[None,:]
        # Dummy variable to work on
        matrix = self.modified_Hessian
        # Preconditioning along null vector
        vector = self.null_vector# Shape: (n,)
        vector = vector/diag
        vector = vector/np.linalg.norm( vector )
        vector = vector.reshape( ( len(vector), 1 ) )# Reshaping for the next step, shape: (n,1)
        matrix = matrix + np.dot( vector, vector.T )# Null vecctor preconditioning of the Hessian
        # Transformations
        gradient = diag[:,None] * gradient[:,None]
        unwinding_transformations.append( [ diag, lambda d, x : d[:,None] * x ] )

        # Conditioning with other vectors
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        start1 = time.time()
        for i in range( k ):
            vector = self.precond_vectors[i]
            value  = np.dot( np.dot( matrix, vector ), vector )
            vector = vector.reshape( ( vector.shape[0], 1 ) ) 
            P_matrix = np.identity( n ) + ( 1/np.sqrt( value ) - 1 ) * np.dot( vector, vector.T )
            # Transforms
            matrix = np.dot( P_matrix, np.dot( matrix, P_matrix ) )
            gradient = np.dot( P_matrix, gradient )
            unwinding_transformations.append( [ P_matrix, lambda P, x : np.dot( P, x ) ] )
        # end for
        end = time.time()
        interval = 1e3 * ( end - start1 )
        timings.append( interval )
        print( "\n|--- Time required for preconditioning matrix formation: ", np.round( interval, 5 ), "ms---|" )
        # end for

        start2 = time.time()
        # Debug
        if debug:
          eig, v = np.linalg.eigh( matrix )
          sorting_indices = np.argsort( eig )
          eig = eig[ sorting_indices ]
          v   = v[ : , sorting_indices ]
          #
          print( "List of smallest eigenvalues: ", eig[ : 5 ] )
          print( "List of largest  eigenvalues: ", eig[ - 5 : ] )
        end = time.time()
        interval = 1e3 * ( end - start2 )
        timings.append( interval )

        print( "|--- Time taken for the debug step: ", np.round( interval, 5 ), "ms---|" )

        start3 = time.time()
        # Solve
        self.Hessian_stabilized = - matrix/self.epsilon
        if iterative_inversion >= 0:
          accumulator = gradient
          inverse = gradient
          delta   = - ( matrix - np.identity( n ) ) # Unipotent part
          for i in range( iterative_inversion ):
            accumulator = np.dot( delta, accumulator )
            inverse = inverse + accumulator
          p_k = self.epsilon * inverse
        else:
          p_k = - np.linalg.solve( self.Hessian_stabilized, gradient )
        end = time.time()
        interval = 1e3 * ( end - start3 )
        timings.append( interval )
        print( "|--- Time taken to invert the linear system for p_k: ", np.round( interval, 5 ), "ms---|" )

        start4 = time.time()
        # Unwind
        for transform in unwinding_transformations:
          P, f = transform
          p_k = f(P, p_k)    
        end = time.time()
        interval = 1e3 * ( end - start4 )
        timings.append( interval )
        print( "|--- Time taken for unwinding: ", np.round( interval, 5 ), "ms---|" )
        interval = 1e3 * ( end - start )
        timings.append( interval )    

        print( "|--- Time taken for the complete code block: ", np.round( interval, 2 ), "ms---|\n" )
        return p_k.flatten(), timings

    def _precond_inversion_v1( self, unnormalized_Hessian, gradient, iterative_inversion, debug ):
        """

        Parameters:
        -----------
            unnormalized_Hessian :  ndarray, shape (n,n)
                                    The unnormalized Hessian.
            gradient :  ndarray, shape (n,)
                        The gradient of the objective function with respect to potential f.
            iterative_inversion : int
                                  The number of iterative inversions to be performed to obtain the inverse of the Hessian followed by obtaining the ascent direction.
                                  Defaults to -1, which indicates exact inversion.
            debug : bool
                    To observe the behaviour of the smallest and the largest eigenvalues of the Hessian. Defaults to False.

        Returns:
        --------
        Returns a tuple containing the optimal ascent direction vector p and the recorded timings of various steps of the algorithm. 
        The following are their descriptions:
            p : ndarray, shape (n,)
                The optimal ascent direction vector.
            timings : list                                 
                      The list of timestamps recorded.
        """
        timings = []
        start = time.time()
        # Record list of unwinding transformations on final result
        unwinding_transformations = []

        # Construct modified Hessian
        diag = 1/np.sqrt( np.diag( unnormalized_Hessian ).flatten() )
        self.modified_Hessian = diag[:,None] * unnormalized_Hessian * diag[None,:]
        
        # Dummy variable to work on
        matrix = self.modified_Hessian

        # Preconditioning along null vector
        vector = self.null_vector# Shape: (n,)
        vector = vector/diag
        vector = vector/np.linalg.norm( vector )
        vector = vector.reshape( ( len(vector), 1 ) )# Reshaping for the next step, shape: (n,1)
        matrix = matrix + np.dot( vector, vector.T )# Null vecctor preconditioning
        # Transformations
        gradient = diag[:,None] * gradient[:,None]
        unwinding_transformations.append( [ diag, lambda d, x : d[:,None] * x ] )
        end = time.time()
        interval = 1e3 * ( end - start )
        timings.append( interval )
        print( "\n|--- Time required for initial preconditioning: ", np.round( interval, 5 ), "ms---|" )


        # Conditioning with other vectors
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        start1 = time.time()
        P_matrix = np.identity( n )
        for i in range(k):
          vector = self.precond_vectors[i]
          value  = np.dot( np.dot( matrix, vector ), vector )
          vector = vector.reshape( ( vector.shape[0], 1 ) )
          P_matrix = P_matrix + ( 1/np.sqrt( value ) - 1 ) * np.dot( vector, vector.T )
        # end for
        unwinding_transformations.append( [ P_matrix, lambda P, x : np.dot( P, x ) ] )
        
        matrix = np.dot( P_matrix, np.dot( matrix, P_matrix ) )
        gradient = np.dot( P_matrix, gradient )
        end = time.time()
        interval = 1e3 * ( end - start1 )
        timings.append( interval ) 
        print( "|--- Time required for preconditioning matrix formation: ", np.round( interval, 5 ), "ms---|" )

        start2 = time.time()
        # Debug
        if debug:
          eig, v = np.linalg.eigh( matrix )
          sorting_indices = np.argsort( eig )
          eig = eig[ sorting_indices ]
          v   = v[ : , sorting_indices ]
          print( "List of smallest eigenvalues: ", eig[ : 5 ] )
          print( "List of largest  eigenvalues: ", eig[ - 5 : ] )
        end = time.time()
        interval = 1e3 * ( end - start2 )
        timings.append( interval )


        print( "|--- Time taken for the debug step: ", np.round( interval, 5 ), "ms---|" )

        start3 = time.time()
        # Solve
        self.Hessian_stabilized = - matrix/self.epsilon
        if iterative_inversion >= 0:
          accumulator = gradient
          inverse = gradient
          delta   = - ( matrix - np.identity( n ) ) # Unipotent part
          for i in range(iterative_inversion):
            accumulator = np.dot( delta, accumulator )
            inverse = inverse + accumulator
          p_k = self.epsilon * inverse
        else:
          p_k = - np.linalg.solve( self.Hessian_stabilized, gradient )
        end = time.time()
        interval = 1e3 * ( end - start3 )
        timings.append( interval )
        print( "|--- Time taken to invert the linear system for p_k: ", np.round( interval, 5 ), "ms---|")

        start4 = time.time()
         # Unwind
        for transform in unwinding_transformations:
          P, f = transform
          p_k = f(P, p_k)
        end = time.time()
        interval = 1e3 * ( end - start4 )
        timings.append( interval )
        print( "|--- Time taken for unwinding: ", np.round( interval, 5 ), "ms---|" )
        
        interval = 1e3 * ( end - start )
        timings.append( interval )
        print( "|--- Time taken for the complete code block: ", np.round( interval, 2 ), "ms---|\n" )
        return p_k.flatten(), timings

    def _precond_inversion_v2( self, unnormalized_Hessian, gradient, iterative_inversion, debug ):
        """

        Parameters:
        -----------
            unnormalized_Hessian :  ndarray, shape (n,n)
                                    The unnormalized Hessian.
            gradient :  ndarray, shape (n,)
                        The gradient of the objective function with respect to potential f.
            iterative_inversion : int
                                  The number of iterative inversions to be performed to obtain the inverse of the Hessian followed by obtaining the ascent direction.
                                  Defaults to -1, which indicates exact inversion.
            debug : bool
                    To observe the behaviour of the smallest and the largest eigenvalues of the Hessian. Defaults to False.

        Returns:
        --------
        Returns a tuple containing the optimal ascent direction vector p and the recorded timings of various steps of the algorithm. 
        The following are their descriptions:
            p : ndarray, shape (n,)
                The optimal ascent direction vector.
            timings : list
                      The list of timestamps recorded.
        """
        timings = []
        start = time.time()
        # Record list of unwinding transformations on final result
        unwinding_transformations = []

        # Construct modified Hessian
        diag = 1/np.sqrt( np.diag( unnormalized_Hessian ).flatten() )
        self.modified_Hessian = diag[:,None] * unnormalized_Hessian * diag[None,:]
        self.modified_Hessian = unnormalized_Hessian
        
        # Dummy variable to work on
        matrix = self.modified_Hessian

        # Preconditioning along null vector
        vector = self.null_vector# Shape: (n,)
        vector = vector
        vector = vector/np.linalg.norm( vector )
        vector = vector.reshape( ( len(vector), 1 ) )# Reshaping for the next step, shape: (n,1)
        matrix = matrix + np.dot( vector, vector.T )# Null vecctor preconditioning
        # Transformations
        gradient = diag[:,None] * gradient[:,None]
        # unwinding_transformations.append( [ diag, lambda d, x : d[:,None] * x ] )
        end = time.time()
        interval = 1e3 * ( end - start )
        timings.append( interval )
        print( "\n|--- Time required for initial preconditioning: ", np.round( interval, 5 ), "ms---|" )


        # Conditioning with other vectors
        #  Naming conventions:
        #  y = Preconditioning vectors as a numpy matrix n by k
        #  matrix = our matrix A to precondition
        n = self.null_vector.shape[0]
        start1 = time.time()
        y = np.array( self.precond_vectors ).T # Matrix of size n by k
        # Compute eigenvalues
        Ay = np.dot( matrix, y )
        eigenvalues = np.sum( y * Ay, axis = 0 )
        # Compute P_matrix = id + y*diag(values)*y.T
        values = ( ( 1/(np.sqrt(eigenvalues) ) ) - 1 )    # Vector of size k
        z = y * values[None,:]
        P_matrix = ( 1/np.sqrt(2) ) * ( np.identity( n ) + np.dot( z, y.T ) )
        # Old version
        # P_matrix = np.identity(n)
        # for i in range(k):
        #   vector = self.precond_vectors[i]
        #   #value  = np.dot( np.dot( matrix, vector ), vector)
        #   vector = vector.reshape( (vector.shape[0], 1) )
        #   P_matrix = P_matrix+ (1/np.sqrt(eigenvalues[i])-1)*np.dot( vector, vector.T)
        # # end for
        end = time.time()
        unwinding_transformations.append( [ P_matrix, lambda P, x : np.dot( P, x ) ] )
        interval = 1e3 * ( end - start1 )
        timings.append( interval )
        print( "|--- Time required for preconditioning matrix formation: ", np.round( interval, 5 ), "ms---|" )

        # Changing A=matrix to PAP
        start2 = time.time()
        # Old version -- O(n^3)
        # matrix = np.dot( P_matrix, np.dot(matrix, P_matrix) )
        # New version -- O(n^2)
        # PAP = (id + y*diag(values)*y.T)*A*(id + y*diag(values)*y.T)
        #     = A + A*y*diag(values)*y.T + y*diag(values)*(A*y).T + y*diag(values)*y.T*A*y*diag(values)*y.T
        B = np.dot( Ay, z.T )
        C = z @ np.dot( y.T, Ay ) @ z.T
        matrix = matrix + B + B.T + C
        gradient = np.dot( P_matrix, gradient )
        end = time.time()
        interval = 1e3 * ( end - start2 )
        timings.append( interval )
        print( "|--- Time required for changing A to PAP: ", np.round( interval, 5 ), "ms---|" )

        start3 = time.time()
        # Debug
        if debug:
          eig, v = np.linalg.eigh( matrix )
          sorting_indices = np.argsort( eig )
          eig = eig[ sorting_indices ]
          v   = v[ : , sorting_indices ]
          #
          print( "List of smallest eigenvalues: ", eig[ : 5 ] )
          print( "List of largest  eigenvalues: ", eig[ - 5 : ] )
        end = time.time()
        interval = 1e3 * ( end - start3 )
        timings.append( interval )

        print( "|--- Time taken for the debug step: ", np.round( interval, 5 ), "ms---|" )

        start4 = time.time()      

        # Solve
        self.Hessian_stabilized = - matrix/self.epsilon
        if iterative_inversion >= 0:
          accumulator = gradient
          inverse = gradient
          delta   = - ( matrix - np.identity( n ) ) # Unipotent part
          for i in range(iterative_inversion):
            accumulator = np.dot( delta, accumulator )
            inverse = inverse + accumulator
          p_k = self.epsilon * inverse
        else:
          p_k = - np.linalg.solve( self.Hessian_stabilized, gradient )
        end = time.time()
        interval = 1e3 * ( end - start4 )
        timings.append( interval )
        print( "|--- Time taken to invert the linear system for p_k: ", np.round( interval, 5 ), "ms---|" )

        start5 = time.time()
        # Unwind
        for transform in unwinding_transformations:
          P, f = transform
          p_k = f(P, p_k)
        end = time.time()
        interval = 1e3 * ( end - start5 )
        timings.append( interval )

        print( "|--- Time taken for unwinding: ", np.round( interval, 5 ), "ms---|" )
        interval = 1e3 * ( end - start )
        timings.append( interval )

        print( "|--- Time taken for the complete code block: ", np.round( interval, 2 ), "ms---|\n" ) 
        return p_k.flatten(), timings

    def _precond_inversion_v3( self, unnormalized_Hessian, gradient, rtol, atol, iterative_inversion = - 1,  optType = None ):
        """

        Parameters:
        -----------
            unnormalized_Hessian :  ndarray, shape (n,n)
                                    The unnormalized Hessian.
            gradient :  ndarray, shape (n,)
                        The gradient of the objective function with respect to potential f.
            iterative_inversion : int
                                  The number of iterative inversions to be performed to obtain the inverse of the Hessian followed by obtaining the ascent direction.
                                  Defaults to -1, which indicates exact inversion.
            rtol :  float
                    The value of relative tolerance which is a hyperparameter to the iterative inversion algorithm, here it is conjugate gradient or GMRES.
            atol :  float
                    The value of absolute tolerance which is a hyperparameter to the iterative inversion algorithm, here it is conjugate gradient or GMRES.
            optType : str
                      Input for the choice of iterative inversion algorithm. The following are the options:
                      - Conjugate Gradient-- 'cg' : scipy.sparse.linalg.cg
                      - GMRES -- 'gmres' : scipy.sparse.linalg.gmres
        Returns:
        --------
        Returns a tuple containing the optimal ascent direction vector p and the recorded timings of various steps of the algorithm. 
        The following are their descriptions:
            p : ndarray, shape (n,)
                The optimal ascent direction vector.
            timings : list
                      The list of timestamps recorded.
        """
        timings = []
        start = time.time()
        # Record list of unwinding transformations on final result
        unwinding_transformations = []
        # Construct modified Hessian
        diag = 1/np.sqrt( np.diag(unnormalized_Hessian).flatten() )
        self.modified_Hessian = diag[:,None] * unnormalized_Hessian * diag[None,:]        
        # Dummy variable to work on
        matrix = self.modified_Hessian  
        # Preconditioning along null vector
        vector = self.null_vector# Shape: (n,)
        vector = vector/diag
        vector = vector/np.linalg.norm( vector )
        vector = vector.reshape( ( len(vector), 1 ) )# Reshaping for the next step, shape: (n,1)
        matrix = matrix + np.dot( vector, vector.T )# Null vecctor preconditioning
        # Transformations
        gradient = diag[:,None] * gradient[:,None]
        unwinding_transformations.append( [ diag, lambda d, x : d[:,None] * x ] )
        # Record timing 
        end = time.time()
        interval = 1e3 * ( end - start )
        timings.append( interval )
        print("\n|--- Time required for initial preconditioning: ", np.round( interval, 5 ), "ms---|")
        # Conditioning with other vectors
        #  Naming conventions:
        #  y = Preconditioning vectors as a numpy matrix n by k
        #  matrix = our matrix A to precondition
        n = self.null_vector.shape[0]
        start0 = time.time()                                     
        y = np.array( self.precond_vectors ).T # Matrix of size n by k
        # Compute eigenvalues
        Ay = np.dot( matrix, y )
        eigenvalues = np.sum( y * Ay, axis = 0 )
        # Compute P_matrix = id + y*diag(values)*y.T
        values = ( 1/np.sqrt(eigenvalues) - 1 )    # Vector of size k
        z = y * values[None,:]
        P_matrix = ( np.identity( n ) + np.dot( z, y.T ) )
        # Done
        unwinding_transformations.append( [ P_matrix, lambda P, x : np.dot( P, x ) ] )
        # Record timings
        end = time.time()
        interval = 1e3 * ( end - start0 )
        timings.append( interval )
        print( "|--- Time required for preconditioning matrix formation: ", np.round( interval, 5 ), "ms---|" )
        # Changing A=matrix to PAP
        start2 = time.time()
        # PAP = (id + y*diag(values)*y.T)*A*(id + y*diag(values)*y.T)
        #     = A + A*y*diag(values)*y.T + y*diag(values)*(A*y).T + y*diag(values)*y.T*A*y*diag(values)*y.T
        B = np.dot( Ay, z.T )
        C = z @ np.dot(y.T, Ay) @ z.T
        matrix = matrix + B + B.T + C
        gradient = np.dot( P_matrix, gradient )
        end = time.time()
        interval = 1e3 * ( end - start2 )
        timings.append( interval )
        print( "|--- Time required for changing A to PAP: ", np.round( interval, 5 ), "ms---|" )
      

        # Solve either iteratively using CG or exactly
        def mv( vector ):
            return np.dot( matrix, vector ) 
        start3 = time.time()
        if iterative_inversion >= 0:
          self.m = matrix
          A = scipy.sparse.linalg.LinearOperator( ( self.m.shape[0], self.m.shape[1] ), matvec = mv ) 
          if optType == 'cg':
            inverse, exit_code = scipy.sparse.linalg.cg(  A,
                                                          gradient, 
                                                          x0 = gradient, 
                                                          maxiter = iterative_inversion, 
                                                          rtol = rtol, 
                                                          atol = atol )
            print( "  --- CG exit code: ", exit_code)
          else:
            inverse, exit_code = scipy.sparse.linalg.gmres( A, 
                                                            gradient, 
                                                            x0 = gradient, 
                                                            maxiter =   iterative_inversion, 
                                                            rtol = rtol, 
                                                            atol = atol  )
            print( "  --- GMRES exit code: ", exit_code)
          p_k = self.epsilon * inverse
          p_k = p_k.reshape( ( p_k.shape[0], 1 ) ) # For some reason, this outputs (n,) and the next line outputs (n,1)
        else:
          self.Hessian_stabilized = - matrix/self.epsilon
          p_k = - np.linalg.solve( self.Hessian_stabilized, gradient )
        end = time.time()
        interval = 1e3 * ( end - start3 )
        timings.append( interval )
        print( "|--- Time taken to invert the linear system for p_k: ", np.round( interval, 5 ), "ms---|" )
        start4 = time.time()
        # Unwind
        for transform in unwinding_transformations:
          P, f = transform
          p_k = f(P, p_k)
        end = time.time()
        interval = 1e3 * ( end - start4 )
        timings.append( interval )
        print("|--- Time taken for unwinding: ", np.round( interval , 5 ), "ms---|")
        interval = 1e3 * ( end - start )
        timings.append( interval )
        print("|--- Time taken for the complete code block: ", np.round( interval, 2 ), "ms---|\n")
        return p_k.flatten(), timings
      
    def _precond_inversion_v4( self, unnormalized_Hessian, gradient, rtol, atol, iterative_inversion = - 1, optType = None ):
        """

        Parameters:
        -----------
            unnormalized_Hessian :  ndarray, shape (n,n)
                                    The unnormalized Hessian.
            gradient :  ndarray, shape (n,)
                        The gradient of the objective function with respect to potential f.
            iterative_inversion : int
                                  The number of iterative inversions to be performed to obtain the inverse of the Hessian followed by obtaining the ascent direction.
                                  Defaults to -1, which indicates exact inversion.
            rtol :  float
                    The value of relative tolerance which is a hyperparameter to the iterative inversion algorithm, here it is conjugate gradient or GMRES.
            atol :  float
                    The value of absolute tolerance which is a hyperparameter to the iterative inversion algorithm, here it is conjugate gradient or GMRES.
            optType : str
                      Input for the choice of iterative inversion algorithm. The following are the options:
                      - Conjugate Gradient-- 'cg' : scipy.sparse.linalg.cg
                      - GMRES -- 'gmres' : scipy.sparse.linalg.gmres

        Returns:
        --------
        Returns a tuple containing the optimal ascent direction vector p and the recorded timings of various steps of the algorithm. 
        The following are their descriptions:
            p : ndarray, shape (n,)
                The optimal ascent direction vector.
            timings : list
                      The list of timestamps recorded.
        """
        timings = []
        start = time.time()
        # Record list of unwinding transformations on final result
        unwinding_transformations = []
        # Construct modified Hessian
        diag = 1/np.sqrt( np.diag(unnormalized_Hessian).flatten() )
        self.modified_Hessian = diag[:,None] * unnormalized_Hessian * diag[None,:]
        # Dummy variable to work on
        matrix = self.modified_Hessian
        # Preconditioning along null vector
        vector = self.null_vector# Shape: (n,)
        vector = vector/diag
        vector = vector/np.linalg.norm( vector )
        vector_E = vector
        if iterative_inversion < 0:
          vector = vector.reshape( (len(vector), 1) )
          matrix = matrix + np.dot( vector, vector.T)
        # Transformations (Initial on gradient and final on result)
        gradient = diag[:,None] * gradient[:,None]
        unwinding_transformations.append( lambda x : diag[:,None] * x )
        # Record timing
        end = time.time()
        interval = 1e3 * ( end - start )
        timings.append( interval )
        print("\n|--- Time required for initial preconditioning: ", np.round( interval, 5 ), "ms---|")
        # Conditioning with other vectors
        #  Naming conventions:
        #  y = Preconditioning vectors as a numpy matrix n by k
        #  matrix = our matrix A to precondition
        #  We only form the data y and z such that
        #  P = id + z*y.T
        start0 = time.time()
        y = np.array( self.precond_vectors ).T # Matrix of size n by k
        # Compute eigenvalues
        Ay = np.dot( matrix, y )
        eigenvalues = np.sum( y * Ay, axis = 0 )
        # Compute data for P = id + y*diag(values)*y.T
        values = ( 1/np.sqrt(eigenvalues) - 1 )# Vector of size k
        z = y * values[None,:]
        # Record timings
        end = time.time()
        interval = 1e3 * ( end - start0 )
        timings.append( interval )
        print( "|--- Time required for preconditioning matrix formation: ", np.round( interval, 5 ), "ms---|" )

        # Changing A=matrix to PAP
        start2 = time.time()

        # Function mapping v to Pv
        # P = Id + z*y.T
        def _apply_P( vector ):
          return  vector + z @ ( y.T @ vector ) 
        
        # Function mapping v to P(A+E)Pv
        # A is matrix
        # E is vector_E*vector_E.T
        def _preconditioned_map( vector ):
          vector = _apply_P( vector ) 
          vector = np.dot( matrix, vector )  + vector_E * np.dot( vector_E, vector )
          vector = _apply_P( vector ) 
          return vector
        # Apply P
        # At beginning on gradient
        # At the end 
        gradient = _apply_P( gradient ) 
        unwinding_transformations.append( lambda x : _apply_P(x) )
        end = time.time()
        interval = 1e3 * ( end - start2 )
        timings.append( interval )
        print( "|--- Time required for changing A to PAP: ", np.round( interval, 5 ), "ms---|" )
        #/
        # Solve either iteratively using CG or exactly
        start3 = time.time()
        if iterative_inversion >= 0:
          self.m = matrix
          A = scipy.sparse.linalg.LinearOperator( ( self.m.shape[0], self.m.shape[1] ), matvec = _preconditioned_map ) 
          if optType == 'cg':
            inverse, exit_code = scipy.sparse.linalg.cg(  A,
                                                          gradient, 
                                                          x0 = gradient, 
                                                          maxiter = iterative_inversion, 
                                                          rtol = rtol, 
                                                          atol = atol  )
            print( "  --- CG exit code: ", exit_code)
          else: 
            inverse, exit_code = scipy.sparse.linalg.gmres( A,
                                                            gradient, 
                                                            x0 = gradient, 
                                                            maxiter = iterative_inversion, 
                                                            rtol = rtol, 
                                                            atol = atol  )
            print( "  --- GMRES exit code: ", exit_code )
          p_k = self.epsilon * inverse
          p_k = p_k.reshape( ( p_k.shape[0], 1 ) ) # For some reason, this outputs (n,) and the next line outputs (n,1)
        else:
          # Preconditioning for exact inversion
          B = np.dot( Ay, z.T )
          C = z @ np.dot( y.T, Ay ) @ z.T
          matrix = matrix + B + B.T + C
          self.Hessian_stabilized = - matrix/self.epsilon
          p_k = - np.linalg.solve( self.Hessian_stabilized, gradient )
        end = time.time()
        interval = 1e3 * (  end - start3 )
        timings.append( interval )
        print( "|--- Time taken to invert the linear system for p_k: ", np.round( interval, 5 ), "ms---|" )
        start4 = time.time()
        # Unwind
        for transform in unwinding_transformations:
          p_k = transform(p_k)
        end = time.time()
        interval = 1e3 * ( end - start4 )
        timings.append( interval )
        print("|--- Time taken for unwinding: ", np.round( interval, 5 ), "ms---|")
        interval = 1e3 * ( end - start )
        timings.append( interval )
        print("|--- Time taken for the complete code block: ", np.round( interval, 2 ), "ms---|\n")
        return p_k.flatten(), timings
      
    def _precond_inversion( self, unnormalized_Hessian, gradient, rtol, atol, iterative_inversion = - 1,  optType = None ):
        """

        Parameters:
        -----------
            unnormalized_Hessian :  ndarray, shape (n,n)
                                    The unnormalized Hessian.
            gradient :  ndarray, shape (n,)
                        The gradient of the objective function with respect to potential f.
            iterative_inversion : int
                                  The number of iterative inversions to be performed to obtain the inverse of the Hessian followed by obtaining the ascent direction.
                                  Defaults to -1, which indicates exact inversion.
            rtol :  float
                    The value of relative tolerance which is a hyperparameter to the iterative inversion algorithm, here it is conjugate gradient or GMRES.
            atol :  float
                    The value of absolute tolerance which is a hyperparameter to the iterative inversion algorithm, here it is conjugate gradient or GMRES.
            optType : str
                      Input for the choice of iterative inversion algorithm. The following are the options:
                      - Conjugate Gradient-- 'cg' : scipy.sparse.linalg.cg
                      - GMRES -- 'gmres' : scipy.sparse.linalg.gmres

        Returns:
        --------
        Returns a tuple containing the optimal ascent direction vector p and the recorded timings of various steps of the algorithm. 
        The following are their descriptions:
            p : ndarray, shape (n,)
                The optimal ascent direction vector.
            timings : list
                      The list of timestamps recorded.
        """
        timings = []
        start = time.time()
        # Record list of unwinding transformations on final result
        unwinding_transformations = []  
        # Construct modified Hessian
        diag = 1/np.sqrt( np.diag(unnormalized_Hessian).flatten() )
        self.modified_Hessian = diag[:,None] * unnormalized_Hessian * diag[None,:]        
        # Dummy variable to work on
        matrix = self.modified_Hessian
        # Preconditioning along null vector
        vector = self.null_vector# Shape: (n,)
        vector = vector/diag
        vector = vector/np.linalg.norm( vector )
        vector_E = vector
        if iterative_inversion < 0:
          vector = vector.reshape( (len(vector), 1) )
          matrix = matrix + np.dot( vector, vector.T )                                                    
        # Transformations (Initial on gradient and final on result)
        gradient = diag[:,None] * gradient[:,None]
        unwinding_transformations.append( lambda x : diag[:,None] * x )
        # Record timing
        end = time.time()
        interval = 1e3 * ( end - start )
        timings.append( interval )
        # Conditioning with other vectors
        #  Naming conventions:
        #  y = Preconditioning vectors as a numpy matrix n by k
        #  matrix = our matrix A to precondition
        #  We only form the data y and z such that
        #  P = id + z*y.T
        start0 = time.time()
        y = np.array( self.precond_vectors ).T # Matrix of size n by k
        # Compute eigenvalues
        Ay = np.dot( matrix, y )
        eigenvalues = np.sum( y * Ay, axis = 0 )
        # Compute data for P = id + y*diag(values)*y.T
        values = ( 1/np.sqrt(eigenvalues) - 1 )# Vector of size k
        z = y * values[None,:]
        # Record timings
        end = time.time()
        interval = 1e3 * ( end - start0 )
        timings.append( interval )

        # Changing the A matrix to PAP
        start2 = time.time()

        # Function mapping v to Pv
        # P = Id + z*y.T
        def _apply_P( vector):
          return vector + z @ ( y.T @ vector )
        # Function mapping v to P(A+E)Pv
        # A is matrix
        # E is vector_E*vector_E.T
        def _preconditioned_map( vector ):
          vector   = _apply_P( vector )
          vector   = np.dot( matrix, vector )  + vector_E * np.dot( vector_E, vector )
          vector   = _apply_P( vector ) 
          return vector
        # Apply P
        # At beginning on gradient
        # At the end 
        gradient = _apply_P( gradient )
        unwinding_transformations.append( lambda x : _apply_P(x) )
        end      = time.time()
        interval = 1e3 * ( end - start2 )
        timings.append( interval )  
        #
        # Solve either iteratively using CG or exactly
        start3 = time.time()
       
        if iterative_inversion >= 0:
          self.m  = matrix
          A = scipy.sparse.linalg.LinearOperator( ( self.m.shape[0], self.m.shape[1] ), matvec = _preconditioned_map ) 
          if optType == 'cg':
            inverse, exit_code = scipy.sparse.linalg.cg(  A,
                                                          gradient, 
                                                          x0 = gradient, 
                                                          maxiter = iterative_inversion, 
                                                          rtol = rtol, 
                                                          atol = atol )
            # print( "  --- CG exit code: ", exit_code)
          else:
            inverse, exit_code = scipy.sparse.linalg.gmres( A,
                                                            gradient, 
                                                            x0 = gradient, 
                                                            maxiter = iterative_inversion, 
                                                            rtol = rtol, 
                                                            atol = atol )
            # print( "  --- GMRES exit code: ", exit_code)
          p_k = self.epsilon * inverse
          p_k = p_k.reshape( ( p_k.shape[0], 1 ) ) # For some reason, this outputs (n,) and the next line outputs (n,1)
        else:
          # Preconditioning for exact inverse 
          B = np.dot( Ay, z.T )
          C = z @ np.dot( y.T, Ay ) @ z.T
          matrix = matrix + B + B.T + C
          self.Hessian_stabilized = - matrix/self.epsilon
          p_k = -np.linalg.solve( self.Hessian_stabilized, gradient )
        end = time.time() 
        interval = 1e3 * ( end - start3 )
        timings.append( interval )
        start4 = time.time()
        # Unwind
        for transform in unwinding_transformations:
          p_k = transform( p_k )
        end = time.time()
        interval = 1e3 * ( end - start4 )
        timings.append( interval )
        interval = 1e3 * ( end - start )
        timings.append( interval )
        # print("|--- Time taken for the complete code block: ",np.round( interval,2),"ms---|\n")
        return p_k.flatten(), timings
    
    def _update( self, tol = 1e-12, max_iterations = 100, iterative_inversion = - 1, version = 1, relative_tol = 1e-5, absolute_tol = 1e-10,  debug = False, optType = 'cg' ):
        """

        Parameters:
        -----------
            tol : float
                  The tolerance limit for the error. Defaults to 1e-12.
            maxiter : int
                      The maximum iteration for the optimization algorithm. Defaults to 100.
            iterative_inversion : int
                                  The number of iterative inversions to be performed to obtain the inverse of the Hessian followed by obtaining the ascent diorection.
                                  Defaults to -1, which indicates exact inversion.
            relative_tol :  float
                            The value of the hyperparameter relative tolerance of the iterative inversion algorithm, which here is conjugate gradient or GMRES.
            absolute_tol : float
                          The value of the hyperparameter absolute tolerance of the iterative inversion algorithm, which here is conjugate gradient or GMRES.
            version : int
                      The version of the precondioned iterative inversion to be used. Defaults to 1.
            debug : bool
                    Implemented for versions 0, 1 and 2 for observing the eigenvalues and eigenvectors of the Hessian. Defaults to False.
            optType : str
                      Input to choose the iterative inversion algorithm. The following are the options:
                      -- Conjugate Gradient- 'cg' : scipy.sparse.linalg.cg
                      -- GMRES- 'gmres' : scipy.sparse.linalg.gmres
                      

        Returns:
        --------
        Returns a dictionary where the keys are strings and corresponding list of values obtained over the iteration of the algorithm.
        The following are the keys of the dictionary and the descriptions of their values:
            potential_f : ndarray, shape: (n,)
                          The optimal potential f.
            potential_g : ndarray, shape: (m,)
                          The optimal potential g.
            error : list
                    The list of numerical error observed over the iteration of the algorithm.
            objectives : list
                         The list of numerical error observed over the iteration of the algorithm.
            linesearch_steps :  list
                                The list of ascent step size toward the ascecnt direction evaluated using the Armijo condition over the iteration of the the algorithm.
            timings : list
                      The list of timestamps recorded at various steps of the versions.
        """

        i = 1
        while True: 
            # Compute gradient w.r.t f:
            grad_f = self._computegradientf()# Shape : (n,)
            # Compute the Hessian:
            M = self.a[:,None] * np.exp( -self.z/self.epsilon ) * np.sqrt( self.b )[None,:]# Shape : (n,m)
            RowSum_M = np.sum( M * np.sqrt( self.b )[None,:], axis = 1 )# Shape : (n,)
            self.Hessian = np.diag( RowSum_M ) - np.dot( M, M.T )# Shape : (n,n)
            # Compute solution of Ax = b:
            if version == 4:
              print("\n At iteration: ",i)
              p_k, temp = self._precond_inversion_v4( self.Hessian, 
                                                      grad_f, 
                                                      iterative_inversion = iterative_inversion, 
                                                      rtol = relative_tol,
                                                      atol = absolute_tol,
                                                      optType = optType )
              self.timing.append( temp )
            elif version == 3:
              print("\n At iteration: ",i)
              p_k, temp = self._precond_inversion_v3( self.Hessian, 
                                                      grad_f, 
                                                      iterative_inversion = iterative_inversion, 
                                                      rtol = relative_tol,
                                                      atol = absolute_tol,
                                                      optType = optType )
              self.timing.append( temp )
            elif version == 2:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v2( self.Hessian, 
                                                      grad_f, 
                                                      iterative_inversion = iterative_inversion, 
                                                      debug = debug )
              self.timing.append( temp )
            elif version == 1:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v1( self.Hessian, 
                                                      grad_f, 
                                                      iterative_inversion = iterative_inversion, 
                                                      debug = debug )
              self.timing.append( temp )
            elif version == 0:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v0( self.Hessian,
                                                      grad_f,
                                                      iterative_inversion = iterative_inversion,
                                                      debug = debug)
              self.timing.append( temp )
            else:
              #print("At iteration: ",i)
              p_k, temp = self._precond_inversion(  self.Hessian, 
                                                    grad_f, 
                                                    iterative_inversion = iterative_inversion, 
                                                    rtol = relative_tol,
                                                    atol = absolute_tol,
                                                    optType = optType )
              self.timing.append( temp )
            # Wolfe condition 1: Armijo Condition:  
            slope = np.dot( p_k, grad_f )
            alpha = 1
            start = time.time()
            alpha = self._wolfe1( alpha, p_k, slope )
            end = time.time()
            self.alpha_list.append( alpha )
            # Update f and g:
            self.f = self.f + alpha * p_k
            self.g = self._get_g( self.C - self.f[:,None] )# Shape: (m,)
            self.z = ( self.C - self.f[:,None] - self.g[None,:] )# Shape : (n,m)
            P = self.a[:,None] * ( np.exp( - self.z/self.epsilon ) ) * self.b[None,:]# Shape : (n,m)
            # Check conservation of mass
            self.err.append(  np.linalg.norm( np.sum( P, axis = 1 ) - self.a, ord = 1 )
                              +
                              np.linalg.norm( np.sum( P, axis = 0 ) - self.b, ord = 1 ))
            # Evaluating objective function after the ascent update
            value = self._objectivefunction( self.f )
            self.objvalues.append(value)
            # Check error:
            if i< max_iterations and ( self.err[-1] > tol ):
                i += 1
            else:   
                print( "Terminating after iteration: ", i )
                break  
        # end for                                                                                                            
        return {
            "potential_f"       : self.f + self.epsilon * np.log( self.a ),
            "potential_g"       : self.g + self.epsilon * np.log( self.b ),
            "error"             : self.err,
            "objectives"        : self.objvalues,
            "linesearch_steps"  : self.alpha_list,
            "timings"           : self.timing,

        }