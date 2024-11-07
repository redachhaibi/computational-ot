import numpy as np
import scipy
import logging
import time

class damped_Newton_with_preconditioning:
      def __init__( self, K, a, b, f, g, epsilon, rho, c, null_vector, precond_vectors ):
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
            f : ndarray, shape (n,) 
                The initial Kantorovich potential f.
            g : ndarray, shape (n,)
                The initial Kantorovich potential g.
            rho : float
                  Damping factor for the line search update step.
            epsilon : float
                      The regularization parameter.
            c : float
                Damping factor for the slope in the Armijo's condition.
            null_vector : ndarray, shape (n,)
                          The null vector of the Hessian as in the semi-dual damped Newton without preconditioning.
            precond_vectors : list of ndarrays, shape (n,)
                              The list of preconditioning vectors obtained from the Hessian at optimum obtained from the algorithm without any preconditioning,
                              that is, damped Newton with only null vector preconditioning and exact inversion.
                              
        """
        self.K = K
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.x = np.concatenate( ( f, g ), axis = None )
        self.rho = rho           
        self.c = c
        self.null_vector = null_vector
        self.precond_vectors = precond_vectors
        self.alpha = []
        self.err_a = []
        self.err_b = [] 
        self.objvalues = [] 
        self.timing = []
        self.out = []
       
      def _computegradientf( self, f ):
        """
        Computes Gradient with respect to f
        
        Parameters:
        -----------
          f : ndarray, shape (n,)
              The Kantorovich potential f.
        """
        u = np.exp( f/self.epsilon )
        v = np.exp( self.x[self.a.shape[0]:]/self.epsilon )
        return self.a - ( self.a * u * np.dot( self.K,  v  * self.b ) )# Shape : (n,)

 
      def _computegradientg( self, g ):
        """
        Computes Gradient with respect to g
        
        Parameters:
        -----------
          g : ndarray, shape (m,)
              The Kantorovich potential g.

        """
        u = np.exp( self.x[:self.a.shape[0]]/self.epsilon )
        v = np.exp( g/self.epsilon )
        return self.b - ( ( self.b * v ) * np.dot( self.K.T, u * self.a ) )# Shape : (m,)

      def _objectivefunction( self, x ):
        """
        
        Parameters:
        -----------
          x : ndarray, shape (n+m,)
              The stacked vector containing the potentials f and g.
            
        Returns:
        --------
          Q(f,g) :  float
                    The value of objective function obtained by evaluating the formula Q(f,g) = < f, a > + < g, b > - epsilon * ( < u, Kv > - 1 ),
                    where u = exp( f/epsilon ), v = exp( g/epsilon ). 
          
        """
        f = x[:self.a.shape[0]]
        g = x[self.a.shape[0]:]
        regularizer = - self.epsilon * ( np.dot( self.a * np.exp( f/self.epsilon ).T, np.dot( self.K, np.exp( g/self.epsilon ) * self.b ) ) - 1 )
        Q = np.dot( f, self.a ) + np.dot( g, self.b ) + regularizer
        return Q
      def _wolfe1( self, alpha, p, slope ):
        #Armijo Condition
        """

        Parameters:
        -----------
            alpha : float
                    The ascent step size, initialized to 1.
            p : ndarray, shape (n,)
                The optimal direction.
            slope : float
                    It is the inner product of the gradient and p.
        Returns:
        --------
            alpha : float
                    The updated ascent step size.
        """
        reduction_count = 0           
        while True:   
            condition = self._objectivefunction( self.x + alpha * p ) < self._objectivefunction( self.x ) + self.c * alpha * slope
            if condition or np.isnan( self._objectivefunction( self.x + alpha * p ) ):
                alpha = self.rho * alpha                                                     
                reduction_count += 1
            else:
                break
        return alpha

        
      # First implementation
      # To be made faster using
      # - vector operations only
      # - C code following 
      #   |- https://medium.com/spikelab/calling-c-functions-from-python-104e609f2804
      #   |- https://scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html
      #
      # TODO: Complete refactor.
      def _precond_inversion_v0( self, unnormalized_Hessian, gradient, iterative_inversion = - 1, debug = False ):
        """

        Parameters:
        -----------
            unnormalized_Hessian :  ndarray, shape (n,m)
                                    The unnormalized Hessian.
            gradient :  ndarray, shape (n+m,)
                        The gradient vector.
            iterative_inversion : int
                                  The number of iterative inversions. Defaults to -1.
            debug : bool
                    To observe eigenvalues and eigenvectors. Defaults to False.

        Returns:
        --------
        Returns a tuple containing the optimal update direction p and the timings recorded at various steps of the method. 
        The following are their descriptions:
            p : ndarray, shape (n+m,1)
                The optimal direction vector.
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
        vector = self.null_vector
        vector = vector/diag
        vector = vector/np.linalg.norm( vector )
        vector = vector.reshape( ( len(vector), 1 ) )
        matrix = matrix + np.dot( vector, vector.T )
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
          v   = v[ :, sorting_indices ]
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
        print( "|--- Time taken for unwinding: ", np.round( interval, 5 ),"ms---|" )
        interval = 1e3 * ( end - start )
        timings.append( interval )    

        print( "|--- Time taken for the complete code block: ", np.round( interval, 2 ), "ms---|\n" )
        return p_k, timings

      def _precond_inversion_v1( self, unnormalized_Hessian, gradient, iterative_inversion = - 1, debug = False ):
        """

        Parameters:
        -----------
            unnormalized_Hessian :  ndarray, shape (n,m)
                                    The unnormalized Hessian.
            gradient :  ndarray, shape (n+m,)
                        The gradient vector.
            iterative_inversion : int
                                  The number of iterative inversions. Defaults to -1.
            debug : bool
                    To observe eigenvalues and eigenvectors. Defaults to False.

        Returns:
        --------
        Returns a tuple containing the optimal update direction p and the timings recorded at various steps of the method. 
        The following are their descriptions:
            p : ndarray, shape (n+m,1)
                The optimal direction vector.
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
        vector = self.null_vector
        vector = vector/diag
        vector = vector/np.linalg.norm( vector )
        vector = vector.reshape( ( len( vector ), 1 ) )
        matrix = matrix + np.dot( vector, vector.T )
        # Transformations
        gradient = diag[:,None] * gradient[:,None]
        unwinding_transformations.append( [ diag, lambda d, x : d[:,None] * x ] )
        end = time.time()
        interval = 1e3 * ( end-start )
        timings.append( interval )
        print( "\n|--- Time required for initial preconditioning: ", np.round( interval, 5 ), "ms---|" )


        # Conditioning with other vectors
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        start1 = time.time()
        P_matrix = np.identity(n)
        for i in range(k):
          vector = self.precond_vectors[i]
          value  = np.dot( np.dot( matrix, vector ), vector )
          vector = vector.reshape( ( vector.shape[0], 1 ) )
          P_matrix = P_matrix + ( 1/np.sqrt( value )-1 ) * np.dot( vector, vector.T )
        # end for
        unwinding_transformations.append( [P_matrix, lambda P, x : np.dot( P, x )] )
        
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
          v   = v[ :, sorting_indices ]
          #
          print( "List of smallest eigenvalues: ", eig[ : 5 ])
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
          delta   = - ( matrix - np.identity(n) ) # Unipotent part
          
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
        return p_k, timings

      def _precond_inversion_v2( self, unnormalized_Hessian, gradient, iterative_inversion = - 1, debug = False ):
        """

        Parameters:
        -----------
            unnormalized_Hessian :  ndarray, shape (n,m)
                                    The unnormalized Hessian.
            gradient :  ndarray, shape (n+m,)
                        The gradient vector.
            iterative_inversion : int
                                  The number of iterative inversions. Defaults to -1.
            optType : str
                      Input for the choice of iterative inversion algorithm, which here are Conjugate Gradient-'cg' and GMRES-'gmres. Defaults to 'cg'.

        Returns:
        --------
        Returns a tuple containing the optimal update direction p and the timings recorded at various steps of the method. 
        The following are their descriptions:
            p : ndarray, shape (n+m,1)
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
        vector = self.null_vector
        vector = vector/diag
        vector = vector/np.linalg.norm( vector )
        vector = vector.reshape( ( len( vector ), 1 ) )
        matrix = matrix + np.dot( vector, vector.T )
        # Transformations
        gradient = diag[:,None] * gradient[:,None]
        unwinding_transformations.append( [ diag, lambda d,x : d[:,None] * x ] )
        end = time.time()
        interval = 1e3 * ( end - start )
        timings.append( interval )
        print( "\n|--- Time required for initial preconditioning: ", np.round( interval, 5 ), "ms---|" )


        # Conditioning with other vectors
        #  Naming conventions:
        #  y = Preconditioning vectors as a numpy matrix n by k
        #  matrix = our matrix A to precondition
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        start1 = time.time()
        y = np.array( self.precond_vectors ).T # Matrix of size n by k
        # Compute eigenvalues
        Ay = np.dot( matrix, y )
        eigenvalues = np.sum( y * Ay, axis = 0 )
        # Compute P_matrix = id + y*diag(values)*y.T
        values = ( ( 1/np.sqrt(eigenvalues) ) - 1 )    # Vector of size k
        z = y * values[None,:]
        P_matrix = np.identity( n ) + np.dot( z, y.T )
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
          v   = v[ :, sorting_indices ]
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
          delta   = - ( matrix - np.identity(n) ) # Unipotent part
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

        print( "|--- Time taken for the complete code block: ", np.round( interval ,2 ), "ms---|\n" ) 
        return p_k, timings

      def _precond_inversion_v3( self, unnormalized_Hessian, gradient, rtol, atol, iterative_inversion = - 1,  optType = None ):
        """

        Parameters:
        -----------
            unnormalized_Hessian :  ndarray, shape (n,n)
                                    The unnormalized Hessian.
            gradient :  ndarray, shape (n,)
                        The gradient vector.
            iterative_inversion : int
                                  The number of iterative inversions. Defaults to -1.
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
        Returns a tuple containing the optimal update direction p and the timings recorded at various steps of the method. 
        The following are their descriptions:
            p : ndarray, shape (n,)
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
        vector = self.null_vector
        vector = vector/diag
        vector = vector/np.linalg.norm( vector )
        vector = vector.reshape( ( len(vector), 1 ) )
        matrix = matrix + np.dot( vector, vector.T)
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
          A = scipy.sparse.linalg.LinearOperator( ( self.m.shape[0],self.m.shape[1] ), matvec = mv ) 
          if optType == 'cg':
            inverse, exit_code = scipy.sparse.linalg.cg(   A,
                                                          gradient,
                                                          x0 = gradient,
                                                          maxiter = iterative_inversion,
                                                          rtol = rtol,
                                                          atol = atol
                                                          )
            print( "  --- CG exit code: ", exit_code)
          else:
            inverse, exit_code = scipy.sparse.linalg.gmres( A,
                                                            gradient,
                                                            x0 = gradient,
                                                            maxiter = iterative_inversion,
                                                            rtol = rtol,
                                                            atol = atol 
                                                            )
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
        return p_k, timings
      
      def _precond_inversion_v4( self, unnormalized_Hessian, gradient, rtol, atol, iterative_inversion = - 1, optType = None ):
        """

        Parameters:
        -----------
            unnormalized_Hessian :  ndarray, shape (n,n)
                                    The unnormalized Hessian.
            gradient :  ndarray, shape (n,)
                        The gradient vector.
            iterative_inversion : int
                                  The number of iterative inversions. Defaults to -1.
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
        Returns a tuple containing the optimal update direction p and the timings recorded at various steps of the method. 
        The following are their descriptions:
            p : ndarray, shape (n,)
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
        matrix = self.modified_Hessian
        vector = self.null_vector# Shape: (n,)
        vector = vector/diag
        vector = vector/np.linalg.norm( vector )
        vector_E = vector
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
          return ( 1/np.sqrt( 2 ) ) * ( vector + z @ ( y.T @ vector ) )
        
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
        #
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
                                                          atol = atol
                                                          )
            print( "  --- CG exit code: ", exit_code)
          else:
            inverse, exit_code = scipy.sparse.linalg.gmres( A,
                                                            gradient,
                                                            x0 = gradient,
                                                            maxiter = iterative_inversion,
                                                            rtol = rtol,
                                                            atol = atol 
                                                            )
            print( "  --- GMRES exit code: ", exit_code)
          p_k = self.epsilon * inverse
          p_k = p_k.reshape( ( p_k.shape[0], 1 ) ) # For some reason, this outputs (n,) and the next line outputs (n,1)
        else:
          # Preconditioning along null vector                                     
          vector = vector.reshape( (len(vector), 1) )
          matrix = matrix + np.dot( vector, vector.T )  
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
        return p_k, timings
      
      def _precond_inversion( self, unnormalized_Hessian, gradient, rtol, atol, iterative_inversion = - 1 ,  optType = None ):
        """

        Parameters:
        -----------
            unnormalized_Hessian :  ndarray, shape (n,n)
                                    The unnormalized Hessian.
            gradient :  ndarray, shape (n,)
                        The gradient vector.
            iterative_inversion : int
                                  The number of iterative inversions. Defaults to -1.
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
        Returns a tuple containing the optimal update direction p and the timings recorded at various steps of the method. 
        The following are their descriptions:
            p : ndarray, shape (n,)
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
        # Dummy variable to work on
        matrix = self.modified_Hessian
        vector = self.null_vector# Shape: (n,)
        vector = vector/diag
        vector = vector/np.linalg.norm( vector )
        vector_E = vector
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
        values = ( 1/np.sqrt(eigenvalues) - 1 )    # Vector of size k
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
                                                          atol = atol
                                                          )
            # print( "  --- CG exit code: ", exit_code)
          else:
            inverse, exit_code = scipy.sparse.linalg.gmres( A,
                                                            gradient,
                                                            x0 = gradient,
                                                            maxiter = iterative_inversion,
                                                            rtol = rtol,
                                                            atol = atol
                                                            )
            # print( "  --- GMRES exit code: ", exit_code)
          p_k = self.epsilon * inverse
          p_k = p_k.reshape( ( p_k.shape[0], 1 ) ) # For some reason, this outputs (n,) and the next line outputs (n,1)
        else:
          # Preconditioning along null vector                                     
          vector = vector.reshape( (len(vector), 1) )
          matrix = matrix + np.dot( vector, vector.T )           
          # Preconditioning for exact inversion
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
        return p_k, timings
     
      

      def _update( self, tol = 1e-11, max_iterations = 50, iterative_inversion = - 1, version = 1, relative_tol = 1e-5, absolute_tol = 1e-10, debug = False, optType = 'cg') :
        """
        
        Parameters:
        -----------
            tol : float
                  The tolerance limit for the error. Defaults to 1e-12.
            maxiter : int
                      The maximum iteration for the optimization algorithm. Defaults to 100.
            iterative_inversion : int
                                  The number of iterative inversions to be used. Defaults to -1.
            version:  int
                      The version of the precondioned iterative inversion to be used. Defaults to 1.
            debug : bool
                    Implemented for versions 0, 1 and 2 for observing the eigenvalues and eigenvectors of the Hessian. Defaults to False.
            optType : str
                      Input for the choice of iterative inversion algorithm. The following are the options:
                      - Conjugate Gradient-- 'cg' : scipy.sparse.linalg.cg
                      - GMRES -- 'gmres' : scipy.sparse.linalg.gmres

        Returns:
        --------
        Returns a dictionary where the keys are strings and the values are ndarrays.
        The following are the keys of the dictionary and the descriptions of their values:
            potential_f : ndarray, shape (n,)
                          The optimal Kantorovich potential f.
            potential_g : ndarray, shape (m,)
                          The optimal Kantorovich potential g.
            error_a : list
                      The list of numerical errors in estimating the prabability histogram a over the iteration of the algorithm.
            error_b : list
                      The list of numerical errors observed in estimating the probability histogram b over the iteration of the algorithm.
            objectives  : list
                          The list of values of the objective function observed during its optimization iterations.
                        
            linesearch_steps  : list
                                The list of different step sizes obtained by using the Armijo's rule.
            timings : list
                      The list of timestamps recorded at various steps of the versions.
        """
        i = 1
        while True :
            # Compute gradient    
            grad_f = self._computegradientf(self.x[:self.a.shape[0]])
            grad_g = self._computegradientg(self.x[self.a.shape[0]:])
            gradient = np.concatenate( ( grad_f, grad_g ), axis = None )
            
            # Compute Hessian
            u = np.exp( self.x[:self.a.shape[0]]/self.epsilon )
            v = np.exp( self.x[self.a.shape[0]:]/self.epsilon )
                        
            r1 = ( self.a * u * np.dot( self.K,  v  * self.b ) )
            r2 = ( ( self.b * v ) * np.dot( self.K.T, u * self.a ) )
            # r1 = u * np.dot( self.K,v )
            # r2 = v * np.dot( self.K.T, u )
            # P  = u*self.K*(v.T) # WRONG AGAIN: DANGEROUS CODE!!
            u = u.reshape( u.shape[0], )
            v = v.reshape( v.shape[0], )
            P = self.a[:,None] * u[:,None] * self.K * v[None,:] * self.b[None,:]

            A = np.diag( np.array( r1.reshape( r1.shape[0], ) ) )
            B = P
            C = P.T
            D = np.diag( np.array( r2.reshape( r2.shape[0], ) ) )
            result = np.vstack( ( np.hstack( ( A, B ) ), np.hstack( ( C ,D ) ) ) )
                          
                    
            # Inverting Hessian against gradient with preconditioning
            if version == 4:
              print("\n At iteration: ",i)
              p_k, temp = self._precond_inversion_v4( result, 
                                                      gradient, 
                                                      iterative_inversion = iterative_inversion, 
                                                      rtol = relative_tol,
                                                      atol = absolute_tol,
                                                      optType = optType )
              self.timing.append( temp )
            elif version == 3:
              print("\n At iteration: ",i)
              p_k, temp = self._precond_inversion_v3( result, 
                                                      gradient, 
                                                      iterative_inversion = iterative_inversion, 
                                                      rtol = relative_tol,
                                                      atol = absolute_tol,
                                                      optType = optType )
              self.timing.append( temp )
            elif version == 2:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v2(   result, 
                                                        gradient, 
                                                        iterative_inversion = iterative_inversion, 
                                                        debug = debug )
              self.timing.append( temp )
            elif version == 1:
              print("\n At iteration: ",i)
              p_k, temp  = self._precond_inversion_v1(  result, 
                                                        gradient, 
                                                        iterative_inversion = iterative_inversion, 
                                                        debug = debug )
              self.timing.append( temp )
            elif version == 0:
              print("\n At iteration: ",i)
              p_k, temp  = self._precond_inversion_v0(  result,
                                                        gradient,
                                                        iterative_inversion = iterative_inversion,
                                                        debug = debug )
              self.timing.append( temp )
            else:
              #print("At iteration: ",i)
              p_k, temp = self._precond_inversion(  result, 
                                                    gradient, 
                                                    iterative_inversion = iterative_inversion, 
                                                    rtol = relative_tol,
                                                    atol = absolute_tol,
                                                    optType = optType )
              self.timing.append( temp )

           # Stacked
            p_k_stacked = np.concatenate( ( p_k[:self.a.shape[0]], p_k[self.a.shape[0]:] ), axis = None )
            # Wolfe Condition 1: Armijo Condition  
            slope = np.dot( p_k.T, gradient )
            alpha = 1
            start = time.time()
            alpha = self._wolfe1( alpha, p_k_stacked, slope )
            end = time.time()
            self.alpha.append( alpha )
            # Updating the vector stack containing f and g towards the ascent direction
            self.x = self.x + alpha * p_k_stacked
            # print("Norm of (f,g) after update: ",np.linalg.norm(self.x))
            # Error computation 1
            s = self.a * np.exp( self.x[:self.a.shape[0]]/self.epsilon ) * np.dot( self.K, np.exp( self.x[self.a.shape[0]:]/self.epsilon ) * self.b  )
            self.err_a.append( np.linalg.norm( s - self.a ) )
            # Error computation 2
            r = self.b * np.exp( self.x[self.a.shape[0]:]/self.epsilon ) * np.dot( self.K .T, np.exp(self.x[:self.a.shape[0]]/self.epsilon ) * self.a )
            self.err_b.append( np.linalg.norm( r - self.b ) )
            # Evaluating objective function after the ascent update
            value = self._objectivefunction( self.x )
            self.objvalues.append( value )
            
            if i < max_iterations and ( self.err_a[-1] > tol or self.err_b[-1] > tol ) :
                 i += 1
            else:
                print( "Terminating after iteration: ",i )
                break
        # end for
        return {
          "potential_f"      : self.x[:self.a.shape[0]] + self.epsilon * np.log( self.a ),
          "potential_g"      : self.x[self.a.shape[0]:] + self.epsilon * np.log( self.b ),
          "error_a"          : self.err_a,
          "error_b"          : self.err_b,
          "objective_values" : self.objvalues,
          "linesearch_steps" : self.alpha,
          "timings"          : self.timing,
          }

