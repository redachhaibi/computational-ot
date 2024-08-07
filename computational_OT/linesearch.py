import numpy as np

class LineSearch:

      def __init__( self, K, a, b, f, g, epsilon, rho, rho_inc, c, z ):
        """
        
        Parameters:
        -----------
            K : ndarray, shape (n,m)
                The Gibb's kernel.
            a : ndarray, shape (n,)
                The probability histogram of the sample of size n.
            b : ndarray, shape (m,)
                The probability histogram of the sample of size m.
            f : ndarray, shape (n,)
                The initial Kantorovich potential f.
            g : ndarray, shape (m,)
                The initial Kantorovich potential g.
            epsilon : float
                      The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
            rho : float
                  Damping factor for the line search update step.
            rho_inc : float
                      Factor to increase the line search update step
            c1 :  float
                  Damping factor for the slope in the Armijo's condition.
            z : float
                Initial update step.
                
        """
        self.K = K
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.x = np.concatenate( ( f, g ), axis = None )
        self.rho = rho
        self.rho_inc = rho_inc
        self.c = c
        self.z = z
        self.alpha = []
        self.err_a = []
        self.err_b = []
        self.objvalues = []
      
      def _computegradientf( self, f ):
        """

        Parameters:
        -----------
            f : ndarray, shape (n,)
                Input value of the potential f.
            
        Returns:
        --------
            ndarray, shape (n,)
            The gradient of the objective function with respect to potential f.
        """
        return ( self.a - ( np.exp( f/self.epsilon ) * np.dot( self.K, np.exp( self.x[self.a.shape[0]:]/self.epsilon ) ) )   )

      def _computegradientg( self, g ):
        """

        Parameters:
        -----------
            g : ndarray, shape (m,)
                Input value of the potential g.

        Returns:
        --------
            ndarray, shape (m,)
            The gradient of the objective function with respect to potential g.
        """
        return ( self.b - ( np.exp( g/self.epsilon ) * np.dot( self.K.T, np.exp( self.x[: self.a.shape[0]]/self.epsilon ) ) ) )

      def _objectivefunction( self, x ):
        """

        Parameters:
        -----------
            x : ndarray, shape (n+m,)
                The vector containing the potentials f and g.
                
        Returns:
        --------
            Q(f,g) :  float
                      The value of objective function obtained by evaluating the formula Q(f,g) = < f, a > + < g, b > - epsilon*< u, Kv >,
                      where u = exp( f/epsilon ), v = exp( g/epsilon ). 
        """
        f = x[: self.a.shape[0]]
        g = x[self.a.shape[0]:]
        return np.dot( f, self.a ) + np.dot( g, self.b ) - self.epsilon * np.dot( np.exp( f/self.epsilon ).T, np.dot( self.K,np.exp( g/self.epsilon ) ) )
      
    
      def _wolfe1( self, alpha, p, slope ):#Armijo Condition
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
            condition = self._objectivefunction( self.x + alpha * p ) < self._objectivefunction( self.x ) + self.c * alpha * slope
            if condition:
                alpha = self.rho * alpha
                reduction_count += 1
            else:
                break

        condition_inc = self._objectivefunction( self.x + ( self.rho_inc * alpha ) * p ) >= self._objectivefunction( self.x ) + self.c * ( self.rho_inc * alpha) * slope
        if reduction_count == 0 and condition_inc:
            alpha = self.rho_inc * alpha

        return alpha

      def _update( self, tol = 1e-12, maxiter = 1000 ):
        """
        
        Parameters:
        -----------
            tol : float
                  The tolerance limit for the error. Defaults to 1e-12.
            maxiter : float
                      The maximum iteration for the optimization algorithm. Defaults to 1000.
            
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
        i = 0
        while True :
            grad_f = self._computegradientf( self.x[:self.a.shape[0]] )
            grad_g = self._computegradientg( self.x[self.a.shape[0]:] )
        
            gradient = np.concatenate( ( grad_f, grad_g ), axis = None )
            
            slope = np.dot( gradient.T, gradient )
            p_k = np.concatenate( ( grad_f, grad_g ), axis = None )

            self.alpha.append( self.z )
            if i != 0:
              self.alpha[i] = self.alpha[i-1]

            # Wolfe Condition 1:Armijo Condition  
            self.alpha[i] = self._wolfe1( self.alpha[i], p_k, slope )

            self.x = self.x + self.alpha[i] * p_k
           
            # error computation 1
            s = np.exp( self.x[self.a.shape[0]:]/self.epsilon ) * np.dot( self.K, np.exp( self.x[self.a.shape[0]:]/self.epsilon ) )
            self.err_a.append( np.linalg.norm( s - self.a ) )
            
            # error computation 2
            r = np.exp( self.x[:self.a.shape[0]]/self.epsilon ) * np.dot( self.K .T, np.exp( self.x[:self.a.shape[0]] /self.epsilon ) )
            self.err_b.append( np.linalg.norm( r - self.b ) )

            #Calculating Objective values
            self.objvalues.append( self._objectivefunction( self.x ) )
            
            if i < maxiter and ( self.err_a[-1] > tol or self.err_b[-1] > tol ) :
                 i += 1
            else:
              print( "Terminating after iteration: ", i )
              break 
      
        # end for    
        return {
          'potential_f' : self.x[: self.a.shape[0]],
          'potential_g' : self.x[self.a.shape[0] : ],
          'error_a' : self.err_a,
          'error_b' : self.err_b,
          'objectives' : self.objvalues,
          'linesearch_steps' : self.alpha
           
        }
        