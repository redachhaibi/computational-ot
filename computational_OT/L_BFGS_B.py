import numpy as np
import scipy as sc

class L_BFGS_B:
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
                      The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
                    
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
                The vector containing the potentials f and g.

        Returns:
        --------
            gradient : ndarray, shape (n+m,1)
                       The gradient of the objective function.
                       
        """
        gradient = np.vstack( ( - ( self.a - ( np.exp( x[:self.a.shape[0]]/self.epsilon ) * np.dot( self.K, np.exp(x[self.a.shape[0]:]/self.epsilon ) ) ).reshape( x[:self.a.shape[0]].shape[0], -1 ) ),
                    - ( self.b - ( np.exp( x[self.a.shape[0]:]/self.epsilon ) * np.dot( self.K.T, np.exp( x[:self.a.shape[0]]/self.epsilon ) ) ).reshape( x[self.a.shape[0]:].shape[0], -1 ) ) ) )
        return gradient

    def _objectivefunction( self, x ):
        """

        Parameters:
        -----------
            x : ndarray, (n+m,1)
                The vector containing the potentials f and g.

        Returns:
        --------
          Q(f,g) :  float
                    The value of objective function obtained by evaluating the formula Q(f,g) = < f, a > + < g, b > - epsilon*< u, Kv >,
                    where u = exp( f/epsilon ), v = exp( g/epsilon ). 
            
        """
        objective = - np.dot( x[:self.a.shape[0]].T, self.a ) - np.dot( x[self.a.shape[0]:].T, self.b )
        regulariser = self.epsilon * np.dot( np.exp( x[:self.a.shape[0]]/self.epsilon ).T, np.dot( self.K, np.exp( x[self.a.shape[0]:]/self.epsilon ) ) )
        return objective + regulariser

    
    def _update( self, tol = 1e-12, maxiter = 1000, maxf = 1, maxiter_lbgfs = 1, pgt = 1e-20, maxl = 10, factor = 100000000.0 ):
        """

        Parameters:
        -----------
            tol  :  float
                    The tolerance limit for the error. Defaults to 1e-12.
            maxiter  :  int
                        The maximum iteration for the optimization algorithm. Defaults to 1000.
            maxf :  int
                    Maximum number of function evaluations. Defaults to 1.
            maxiter_lbgfs : int
                            Maximum number of iterations. Defaults to 1.
            pgt :   float
                    The iteration will stop when max{|proj g_i | i = 1, ..., n} <= pgtol where proj g_i is the i-th component of the projected gradient. Defaults to 1e-20.
            maxl :  float 
                    Maximum number of line search steps (per iteration). Defaults to 10.
            factor :    float 
                        The iteration stops when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps. Defaults to 100000000.0.

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
        """
        i = 0
        while True :
    
          out = sc.optimize.fmin_l_bfgs_b(  self._objectivefunction, 
                                            self.x,
                                            self._computegradient,
                                            maxfun = maxf,
                                            maxiter = maxiter_lbgfs,
                                            pgtol = pgt,
                                            maxls = maxl,
                                            factr = factor )

          self.x = out[0]            
          
          # error computation 1
          s = np.exp( self.x[:self.a.shape[0]]/self.epsilon )*np.dot( self.K,np.exp( self.x[self.a.shape[0]:]/self.epsilon ) )
          self.err_a.append( np.linalg.norm( s - self.a ) )


          # error computation 2
          
          r = np.exp( self.x[self.a.shape[0]:]/self.epsilon )*np.dot( self.K .T, np.exp( self.x[:self.a.shape[0]]/self.epsilon ) )
          self.err_b.append(np.linalg.norm(r - self.b))

          #Calculating Objective values
          self.objvalues.append( ( np.dot( self.x[:self.a.shape[0]].T, self.a ) 
                                  + 
                                  np.dot( self.x[self.a.shape[0]:].T,self.b )
                                  - 
                                  self.epsilon*np.dot( np.exp( self.x[:self.a.shape[0]]/self.epsilon ).T, np.dot( self.K, np.exp( self.x[self.a.shape[0]:]/self.epsilon ) ) )
                                  ) )

          if i < maxiter and ( self.err_a[-1] > tol or self.err_b[-1] > tol ) :
                 i += 1
          else:
            print( "Terminating after iteration: ", i )
            break 
      

        # end for  
        return {
            'potential_f' : self.x[:self.a.shape[0]],
            'potential_g' : self.x[self.a.shape[0]:],
            'error_a' : self.err_a,
            'error_b' : self.err_b,
            'objectives' : self.objvalues

        }