import numpy as np
class semi_dual_dampedNewton_np:
    def __init__( self, C, a, b, f, epsilon, rho, c, exp_log = 'True' ):
        """
        
        Parameters:
        -----------
            C : ndarray, shape (n,m), 
                It is the cost matrix between the points sampled from the point clouds.
            a : ndarray, shape (n,)
                The probability histogram of the sample of size n.
            b : ndarray, shape (m,)
                The probability histogram of the sample of size m.
            f : ndarray, shape (n,) 
                The initial Kantorovich potential f.
            rho : float
                  Damping factor for the line search ascecnt step size.
            c : float
                Damping factor for the slope in the Armijo's condition.
            epsilon : float
                      The regularization parameter.
            exp_log : bool
                      Indicating to use exp-log regularization or not.
        """
        self.C = C
        self.a = a
        self.b = b
        self.f = f
        self.rho = rho
        self.epsilon = epsilon  
        self.c = c
        self.exp_log = exp_log
        self.alpha_list = []
        self.err = []
        self.objvalues = [] 
        null_vector = np.hstack( np.ones( self.a.shape[0] ) )/np.sqrt( self.a.shape[0] )
        self.null_vector = np.reshape( null_vector, ( self.a.shape[0], 1 ) )# Shape: (n,1)
        self.reg_matrix = np.dot( self.null_vector, self.null_vector.T )# Shape: (n,n)
        self.g = self._get_g( self.C - self.f[:,None] )# Shape: (m,)
        self.z = self.C - self.f[:,None] -  self.g[None,:]# Shape: (n,m)

    def _objectivefunction( self, f ) :
        """ 
     
        Parameters:
        -----------
            f : ndarray, shape (n,)
                The input Kantorovich potential f.
                
        Returns: 
        --------
            Q_semi(f) : float
                        The value of semi-dual objective function obtained by evaluating Q_semi(f) = < f, a > + < g( f, C, epsilon ), b >,
                        where g( f, C, epsilon ) denotes the value of Kantorovich potential g evaluated using the Schrodinger-bridge equations between f and g.
        """
        g = self._get_g( self.C - f[:,None] )# Shape: (m,)
        Q_semi = np.dot( f, self.a ) + np.dot( g, self.b )
        return Q_semi

    def _computegradientf( self ):
        """ 
            Returns:
            --------
            ndarray, shape (n,)
            The gradient of the objective function.
        """
        gradient = self.a * ( np.ones( self.a.shape[0] ) - np.sum( np.exp( - self.z/self.epsilon ) * self.b[None,:], axis = 1 ) )# Shape: (n,)
        return gradient
    
    def _get_g( self, H ):
      """
        Here we compute g.
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
        return -self.epsilon * np.log( np.sum( self.a[:,None] * np.exp( -H/self.epsilon ), axis = 0 ) )# Shape: (m,)
    
    def _explog_g( self, H ):
        """
        Here we incorporate the exp-log regularization method in the computation of the potential g.
        Parameters:
        -----------
            H : ndarray, shape (n,m)
                It is the matrix obtained from the difference C - f.

        Returns:
        --------
            ndarray, shape (m,)
            The exp-log regularized value of potential g.

        """
        return self._g( H - np.min( H, axis = 0 ) ) + np.min( H, axis = 0 )# Shape: (m,)
            
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
        
    def _update( self, tol = 1e-12, max_iterations = 100 ):
        """
        Here perform the iterations of the semi-dual damped Newton algorithm.
        Parameters:
        ----------
            tol : float
                  The tolerance for the numerical error. Defaults to 1e-12.
            max_iterations : int 
                             The maximum number of iteration for the algorithm. Defaults to 100.
        Returns:
        --------
        Returns a dictionary where the keys are strings and corresponding values obtained over the iteration of the algorithm.
        The following are the keys of the dictionary and the descriptions of their values:
            potential_f : ndarray, shape: (n,)
                          The optimal potential f.
            potential_g : ndarray, shape: (m,)
                          The optimal potential g.
            error : list
                    The list of numerical error observed over the iteration of the algorithm.
            objective_values : list
                               The list of objective values observed after each ascent update.
            linesearch_steps : list
                               The list of ascent step size toward the ascecnt direction evaluated using the Armijo condition over the iteration of the the algorithm.
        """
        i = 1
        while True: 
            # Compute gradient w.r.t f:
            grad_f = self._computegradientf()# Shape: (n,)
            # Compute the Hessian:
            M = self.a[:,None] * np.exp( - self.z/self.epsilon ) * np.sqrt( self.b )[None,:]# Shape: (n,m)
            RowSum_M = np.sum( M * np.sqrt( self.b )[None,:], 1 )# Row sum, shape: (n,)
            self.Hessian = -( np.diag( RowSum_M ) - np.dot( M, M.T ) )/self.epsilon# Shape: (n,m)
            mean_eig = -( np.mean( np.diag( RowSum_M ) ) )/self.epsilon
            # Preconditioning the Hessian using its null vector with the factor being the mean of eigenvalues of the Hessian 
            self.Hessian_stabilized = self.Hessian + mean_eig * self.reg_matrix# Shape: (n,m)
            try:    
                p_k = - np.linalg.solve( self.Hessian_stabilized, grad_f )  
            except np.linalg.LinAlgError as e:
                print(f"An error occurred: {e}")
                return -1
            p_k = p_k - self.null_vector.flatten() * np.dot( self.null_vector.flatten(), p_k )# Shape: (n,)
            # Wolfe condition 1: Armijo Condition:      
            slope = np.dot( p_k, grad_f )
            alpha = 1
            alpha = self._wolfe1( alpha, p_k, slope )
            self.alpha_list.append( alpha )
            # Update f and g:
            self.f = self.f + alpha * p_k# Shape: (n,)
            self.g = self._get_g( self.C - self.f[:,None] )# Shape: (m,)
            # Computing the coupling:
            self.z = ( self.C - self.f[:,None] - self.g[None,:] )# Shape: (n,m)
            P = self.a[:,None] * ( np.exp( - self.z/self.epsilon ) ) * self.b[None,:]# Shape: (n,m)
            # Check conservation of mass
            self.err.append(    np.linalg.norm( np.sum( P, axis = 1 ) - self.a, ord = 1 )
                                +
                                np.linalg.norm( np.sum( P, axis = 0 ) - self.b, ord = 1 ))
            # Evaluating objective function after the ascent update
            value = self._objectivefunction( self.f )
            self.objvalues.append( value )
            # Check error:
            if i < max_iterations and ( self.err[-1] > tol ):
                i += 1
            else:   
                print( "Terminating after iteration: ", i )
                break 
        # end for                                                                                                            
        return {
            "potential_f"       : self.f + self.epsilon * np.log( self.a ),
            "potential_g"       : self.g + self.epsilon * np.log( self.b ),
            "error"             : self.err,
            "objective_values"  : self.objvalues,
            "linesearch_steps"  : self.alpha_list
        }