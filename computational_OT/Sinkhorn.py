import numpy as np

class sinkhorn:

  def __init__( self, K, a, b, u, v, epsilon ):
    """
    
    Parameters:
    -----------
        K : ndarray, shape (n,m)
            The Gibb's kernel.
        a : ndarray, shape (n,)
            The probability histogram of the sample of size n.
        b : ndarray, shape (m,)
            The probability histogram of the sample of size m.
        u : ndarray, shape (n,)
            The initial u.
        v : ndarray, shape (m,)
            The initial v.
        epsilon : float
                  The regularization parameter.
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
        target = np.dot( f, self.a ) + np.dot( g, self.b )
        penalization = -self.epsilon*np.dot( np.exp( f/self.epsilon ).T, np.dot( self.K, np.exp( g/self.epsilon ) ) )
        return target + penalization


  def _update( self, tol = 1e-12, max_iterations = 1000 ):
    """

    Parameters:
    -----------
      tol : float
            The tolerance limit for the error. Defaults to 1e-12.
      max_iterations  : int
                        The maximum iteration for the optimization algorithm. Defaults to 1000.

    Returns:
    --------
      Returns a dictionary where the keys are strings and the values are ndarrays or list.
      The following are the keys of the dictionary and the descriptions of their values:
        potential_f : ndarray, shape (n,)
                      The optimal Kantorovich potential f.
        potential_g : ndarray, shape (m,)
                      The optimal Kantorovich potential g.
        error_a : list
                  The list of error of the estimation of the measure 'a' over the iteration of the algorithm.
        error_b : list
                  The list of error of the estimation of the measure 'b' over the iteration of the algorithm.
        objective_values : list
                           The list of objective values observed after each ascent update.
    """
    i = 0
    while True :
      self.obj.append( self._objectivefunction() )
      # Sinkhorn step 1
      self.u = self.a / np.dot( self.K, self.v )
      # Error computation 1
      r = self.v * np.dot( self.K.T, self.u)
      self.err_b.append( np.linalg.norm( r - self.b ) )
      # Sinkhorn step 2
      self.v = self.b / np.dot( self.K.T, self.u )
      # Error computation 2
      s = self.u * np.dot( self.K, self.v )
      self.err_a.append( np.linalg.norm( s - self.a ) )
      iter_condition = ( self.err_a[-1] > tol or self.err_b[-1] > tol )
      if iter_condition and i < max_iterations :
          i += 1
      elif np.isnan( self.err_a[-1] ) or np.isnan( self.err_b[-1] ):
        print( "Sinkhorn is unstable for epsilon: ", self.epsilon, " as there occurs divison by exponentially small values while performing the alternative projection. " )
        break
      else:
        print( "Terminating after iteration: ",i )
        break   
    # end for
    return {
      'potential_f'       : self.epsilon * np.log( self.u ),
      'potential_g'       : self.epsilon * np.log( self.v ),
      'error_a'           : self.err_a,
      'error_b'           : self.err_b,
      'objective_values'  : self.obj
    }
