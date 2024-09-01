import numpy as np

class DampedNewton_SemiDual_np:
    def __init__( self, C, a, b, f, epsilon, rho, c ):
        """
        
        Parameters:
        -----------
            C : ndarray, shape (n,m), 
                It is the cost matrix between the sample points of the two point clouds.
            a : ndarray, shape (n,)
                The probability histogram of the sample of size n.
            b : ndarray, shape (m,)
                The probability histogram of the sample of size m.
            f : ndarray, shape (n,) 
                The initial Kantorovich potential f.
            rho : float
                  Damping factor for the line search update step.
            epsilon : float
                      The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
            c : float
                Damping factor for the slope in the Armijo's condition.
        """
        self.C = C
        self.a = a
        self.b = b
        self.f = f
        self.rho = rho
        self.epsilon = epsilon  
        self.c = c
        self.alpha_list = []
        self.err = []
        self.objvalues = [] 
        null_vector = np.ones( self.a.shape[0] )/np.sqrt( self.a.shape[0] )
        self.null_vector = np.reshape( null_vector, ( self.a.shape[0], 1 ) )# Shape : (n,1)
        self.reg_matrix = np.dot( self.null_vector, self.null_vector.T )# Shape : (n,n)
        self.g = self._logexp_g( self.C - self.f[:,None] )# Shape : (m,)
        self.z = self.C - self.f[:,None] -  self.g[None,:] # Shape : (n,m)

    def _objectivefunction( self, f ) :
        """ 
        
        Parameters:
        -----------
            f : ndarray, shape (n,)
                The Kantorovich potential f.
                
        Returns : 
        ---------
            Q_semi(f) : float
                        The value of semi-dual objective function obtained by evaluating the formula Q_semi(f) = <f,a> + <g(f,C,epsilon),b>,
                        where g(f,C,epsilon) is the value of Kantorovich potential g obtained by using the Schrodinger-bridge equations between f and g.
        """
        g = self._logexp_g( self.C - f[:,None] )
        Q_semi = np.dot( f, self.a ) + np.dot( g, self.b ) 
        return Q_semi
      
    def _computegradientf( self ):
        """ 
            Compute gradient with respect to f of the objective function Q_semi(.).
        """
        gradient = self.a * ( np.ones( self.a.shape[0] ) - np.sum( np.exp( - self.z/self.epsilon ) * self.b[None,:], axis = 1 ) )# Shape : (n,)
        return gradient
    
    def _g( self, H ):
        """

        Parameters:
        -----------
            a : ndarray, shape (n,)
                The probability histogram of the sample of size n.
            H : ndarray, shape (n,m)
                It is the matrix obtained from C - f.
            epsilon :   float
                        The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
                    
        Returns:
        --------
            ndarray, shape (m,)
            The value of potential g obtained from the Schrodinger-bridge equation between the potentials f and g.
        """
        return -self.epsilon * np.log( np.sum( self.a[:,None] * np.exp( -H/self.epsilon ), 0 ) )
    
    def _logexp_g( self, H ):
        """

        Parameters:
        -----------
            a : ndarray, shape (n,)
                The probability histogram of the sample of size n.
            H : ndarray, shape (n,m)
                It   is the matrix obtained from C - f.
            epsilon :  float
                       The regularization factor in the entropy regularized optimization setup of the optimal transport problem.

        Returns:
        --------
            ndarray, shape (m,)
            The value of potential g obtained from the Schrodinger-bridge equation between the potentials f and g after log-exp regularization.

        """
        return self._g( H - np.min( H, 0 ) ) + np.min( H, 0 )
            
   
    def _wolfe1( self, alpha, p, slope ):
        #Armijo Condition
        """

        Parameters:
        -----------
            alpha : float  
                    The update step size.
            p : ndarray, shape (n,)
                The optimal direction.
            slope : float
                    It is the inner product of the gradient and p.

        Returns:
        --------
            alpha : float
                    The updated step size.
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
        
    def _update( self, tol = 1e-12, maxiter = 100 ):
        """

        Parameters:
        ----------
            tol : float
                  The tolerance limit for the error. Defaults to 1e-12.
            maxiter : int 
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
            objectives : ndarray, shape (k,), where k is the number of iterations
                         Objective function values obtained over the iterations of the algorithm.
            linesearch_steps :  ndarray, shape (k,), where k is the number of iterations
                                Different step sizes obtained by using the Armijo's rule along the iterations of the algorithm.
        """
        i = 1
        while True: 
            # Compute gradient w.r.t f:
            grad_f = self._computegradientf()# Shape : (n,)
            # Compute the Hessian:
            M = self.a[:,None] * np.exp( - self.z/self.epsilon ) * np.sqrt( self.b )[None,:]
            Sum_M = np.sum( M * np.sqrt( self.b )[None,:], 1 )
            self.Hessian = - ( np.diag( Sum_M ) - np.dot( M, M.T ) )/self.epsilon
            mean_eig =  -( np.mean( np.diag( Sum_M )  ) )/self.epsilon
            # Regularizing the Hessian using the regularization vector with the factor being the mean of eigenvalues of the Hessian 
            self.Hessian_stabilized = self.Hessian +  mean_eig * self.reg_matrix
            try:    
                p_k = - np.linalg.solve( self.Hessian_stabilized , grad_f )  
            except np.linalg.LinAlgError as e:
                print(f"An error occurred: {e}")
                return np.zeros(5)
            p_k = p_k - self.null_vector.flatten() * np.dot( self.null_vector.flatten(), p_k )# Shape : (n,)
            # Wolfe condition 1: Armijo Condition:  
            slope = np.dot( p_k, grad_f )
            alpha = 1
            alpha = self._wolfe1( alpha, p_k, slope )
            self.alpha_list.append( alpha )
            # Update f and g:
            self.f = self.f + alpha * p_k
            self.g = self._logexp_g( self.C - self.f[:,None] )# Shape : (m,)
            self.z = ( self.C - self.f[:,None] - self.g[None,:] )# Shape : (n,m)
            P = self.a[:,None] * ( np.exp( - self.z/self.epsilon ) ) * self.b[None,:]# Shape : (n,m)
            # Error computation:
            self.err.append( np.linalg.norm( np.sum( P, axis = 1 ) - self.a, ord = 1 ) )
            # Calculating objective function:
            value = self._objectivefunction( self.f )
            self.objvalues.append( value )
            # Check error:
            if i < maxiter and ( self.err[-1] > tol ):
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
            "linesearch_steps"  : self.alpha_list
        }