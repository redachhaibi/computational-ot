import numpy as np

class Sinkhorn:

  def __init__( self, K, a, b, u, v, epsilon ):
    """
    
    Parameters:
    -----------
        K : ndarray, shape (n,m)
            The Gibb's kernel.
        a : ndarray, shape (n,1)
            The probability histogram of the sample of size n.
        b : ndarray, shape (m,1)
            The probability histogram of the sample of size m.
        u : ndarray, shape (n,1)
            The initial left marginal of the coupling.
        v : ndarray, shape (m,1)
            The initial right marginal of the coupling.
        epsilon : float
                  The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
    """
    self.K = K
    self.a = a
    self.b = b
    self.u = u
    self.v = v
    self.epsilon = epsilon
    self.err_a = []
    self.err_b = []
    self.obj = []
    
  def _objectivefunction( self ):
        """
        
        Returns:
        --------
          Q(f,g) :  float
                    The value of objective function obtained by evaluating the formula Q(f,g) = < f, a > + < g, b > - epsilon*< u, Kv >,
                    where u = exp( f/epsilon ), v = exp( g/epsilon ). 

        """
        f = np.log( self.u ) * self.epsilon
        g = np.log( self.v ) * self.epsilon
        target = np.dot( f.T, self.a ) + np.dot( g.T, self.b )
        penalization = -self.epsilon*np.dot( np.exp( f/self.epsilon ).T, np.dot( self.K, np.exp( g/self.epsilon ) ) )
        return target+ penalization


  def _update( self, tol = 1e-12, maxiter = 1000 ):
    """

    Parameters:
    -----------
      tol  : float
             The tolerance limit for the error. Defaults to 1e-12.
      maxiter  : int
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
        error_a : ndarray, shape (k,), where k is the number of iterations
                  The list of error of the estimation of the measure 'a' over the iteration of the algorithm.
        error_b : ndarray, shape (k,), where k is the number of iterations
                  The list of error of the estimation of the measure 'b' over the iteration of the algorithm.
        objectives  : ndarray, shape (k,), where k is the number of iterations
                      The list of objective function values over the iterations of the algorithm.
    """
    i = 0
    while True :
      self.obj.append( self._objectivefunction() )

      # sinkhorn step 1
      self.u = self.a / np.dot( self.K, self.v )
      # error computation 1
      r = self.v*np.dot( self.K.T, self.u)
      self.err_b.append( np.linalg.norm( r - self.b ) )
      
      # sinkhorn step 2
      self.v = self.b / np.dot( self.K.T, self.u )
      
      # error computation 2
      s = self.u*np.dot( self.K, self.v )
      self.err_a.append( np.linalg.norm( s - self.a ) )
      iter_condition = ( self.err_a[-1] > tol or self.err_b[-1] > tol )
      if iter_condition and i < maxiter :
          i += 1
      elif np.isnan( self.err_a[-1] ) or np.isnan( self.err_b[-1] ):
        print( "Sinkhorn is unstable for epsilon: ", self.epsilon, " as there occurs divison by exponentially small values while performing the alternative projection. " )
        break
      else:
        print( "Terminating after iteration: ",i )
        break   

    # end for
    return {
      'potential_f' : self.epsilon * np.log( self.u ).reshape( self.a.shape[0], ),
      'potential_g' : self.epsilon * np.log( self.v ).reshape( self.b.shape[0], ),
      'error_a'     : self.err_a,
      'error_b'     : self.err_b,
      'objectives'  : self.obj
    }
