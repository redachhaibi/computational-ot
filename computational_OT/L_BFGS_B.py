import numpy as np
import scipy as sc

class l_bfgs_b:
    def __init__( self, K, a, b, f, g, epsilon ):
        """

        Parameters:
        -----------
            K : ndarray, shape (n,m)
                The Gibb's kernel of size n by m.
            a : ndarray, shape (n,1)
                The probability histogram of the sample of size n.
            b : ndarray, shape (m,1)
                The probability histogram of the sample of size m.
            f : ndarray, shape (n,1) 
                The initial Kantorovich potential f.
            g : ndarray, shape (m,1) 
                The initial Kantorovich potential g.
            epsilon : float
                      The regularization parameter.    
        """
        self.K = K
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.err_a = []
        self.err_b = []
        self.x = np.vstack( ( f, g ) )
        
        self.objvalues = []

    def _computegradient( self, x ):
        """

        Parameters:
        -----------
            x : ndarray, (n+m,1)
                The vector stack containing the potentials f and g.

        Returns:
        --------
            gradient : ndarray, shape (n+m,1)
                       The vector stack of the gradients of the objective function w.r.t. f and g.    
        """
        u = np.exp( x[:self.a.shape[0]]/self.epsilon )
        v = np.exp( x[self.a.shape[0]:]/self.epsilon )
        gradient = np.vstack( 
                             ( 
                                - ( self.a - ( u * np.dot( self.K, v ) ).reshape( x[:self.a.shape[0]].shape[0], -1 )),
                                - ( self.b - ( v * np.dot( self.K.T, u ) ).reshape( x[self.a.shape[0]:].shape[0], -1 ) ) 
                            ) 
                             )     
        return gradient

    def _objectivefunction( self, x ):
        """

        Parameters:
        -----------
            x : ndarray, (n+m,1)
                The vector stack containing the potentials f and g.

        Returns:
        --------
          Q(f,g) : float
                   The value of objective function obtained by evaluating the formula Q(f,g) = < f, a > + < g, b > - epsilon*< u, Kv >,
                   where u = exp( f/epsilon ), v = exp( g/epsilon ). 
        """
        f = x[:self.a.shape[0]]
        g = x[self.a.shape[0]:]
        u = np.exp( f/self.epsilon )
        v = np.exp( g/self.epsilon )
        objective = - np.dot( f, self.a ) - np.dot( g, self.b )
        regulariser = self.epsilon * np.dot( u.T, np.dot( self.K, v ) )
        return objective + regulariser

    
    def _update( self, tol = 1e-12, max_iterations = 1000, maxf = 1, maxiter_lbgfs = 1, pgt = 1e-20, maxl = 10, factor = 100000000.0 ):
        """

        Parameters:
        -----------
            tol : float
                  The tolerance limit for the error. Defaults to 1e-12.
            max_iterations : int
                             The maximum iteration for the optimization algorithm. Defaults to 1000.
            maxf : int
                   Maximum number of function evaluations. Defaults to 1.
            maxiter_lbgfs : int
                            Maximum number of iterations. Defaults to 1.
            pgt : float
                  The iteration will stop when max{|proj g_i | i = 1, ..., n} <= pgtol where proj g_i is the i-th component of the projected gradient. Defaults to 1e-20.
            maxl : float 
                   Maximum number of line search steps (per iteration). Defaults to 10.
            factor : float 
                     The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps. Defaults to 100000000.0.

        Returns:
        --------            
        Returns a dictionary where the keys are strings and the values are ndarrays or lists.
        The following are the keys of the dictionary and the descriptions of their values:
            potential_f : ndarray, shape (n,)
                          The optimal Kantorovich potential f.
            potential_g : ndarray, shape (m,)
                          The optimal Kantorovich potential g.
            error_a : list
                      The list of errors observed in estimating a over the iteratations.
            error_b : list
                      The list of errors observed in estimating b over the iterations
            objective_values : list
                               The list of objective values observed after each ascent update.
        """
        i = 0
        while True :
    
            out = sc.optimize.fmin_l_bfgs_b(    self._objectivefunction, 
                                                self.x,
                                                self._computegradient,
                                                maxfun = maxf,
                                                maxiter = maxiter_lbgfs,
                                                pgtol = pgt,
                                                maxls = maxl,
                                                factr = factor )

            self.x = out[0]            
            f = self.x[:self.a.shape[0]]
            g = self.x[self.a.shape[0]:]
            u = np.exp( f/self.epsilon )
            v = np.exp( g/self.epsilon )
            # Error computation 1
            s = u * np.dot( self.K, v )
            self.err_a.append( np.linalg.norm( s - self.a ) )
            # Error computation 2
            r = v * np.dot( self.K .T, u )
            self.err_b.append( np.linalg.norm( r - self.b ) )
            # Evaluating objective function after the ascent update
            self.objvalues.append( self._objectivefunction( self.x ) )
            # Checks to termminate the iterations
            iter_condition = ( self.err_a[-1] > tol or self.err_b[-1] > tol )
            if iter_condition and i < max_iterations :
                i += 1
            else:
                print( "Terminating after iteration: ", i )
                break  
        # end for  
        return {
            'potential_f'        : self.x[:self.a.shape[0]],
            'potential_g'        : self.x[self.a.shape[0]:],
            'error_a'            : self.err_a,
            'error_b'            : self.err_b,
            'objective_values'   : self.objvalues

        }