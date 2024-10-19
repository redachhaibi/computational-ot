import numpy as np

class gradient_ascent:
  def __init__( self, K, a, b, f, g, epsilon, learning_rate ):
    """
    
    Parameters:
    -----------
        K : ndarray, shape (n,m)
            The Gibb's kernel of size n by m.
        a : ndarray, shape (n,)
            The probability histogram of the sample of size n.
        b : ndarray, shape (m,)
            The probability histogram of the sample of size m.
        f : ndarray, shape (n,) 
            The initial Kantorovich potential f.
        g : ndarray, shape (m,) 
            The initial Kantorovich potential g.
        epsilon : float
                  The regularization parameter.
        learning_rate : float
                        The step size of the update towards optimal direction.              
    """
    self.K = K
    self.a = a
    self.b = b
    self.f = f
    self.g = g
    self.epsilon = epsilon
    self.err_a = []
    self.err_b = []
    self.learning_rate = learning_rate
    self.obj = []
  
  def _computegradient( self ):
    """ 
    
    Returns:
    --------
      grad_f : ndarray, shape: (n,)
               The gradient of the objective function w.r.t. f.
      grad_g : ndarray, shape: (m,)
               The gradient of the objective function w.r.t. g.
    """
    grad_f = self.a - np.exp( self.f/self.epsilon ) * np.dot( self.K, np.exp( self.g/self.epsilon ) )
    grad_g = self.b - np.exp( self.g/self.epsilon ) * np.dot( self.K.T, np.exp( self.f/self.epsilon ) )
    return grad_f, grad_g
    
  def _objectivefunction( self ):
        """
        
        Returns:
        --------
          Q(f,g) : float
                   The value of objective function obtained by evaluating the formula Q(f,g) = < f, a > + < g, b > - epsilon*< u, Kv >,
                   where u = exp( f/epsilon ), v = exp( g/epsilon ). 
        """

        Q = np.dot( self.f, self.a ) + np.dot( self.g, self.b ) - self.epsilon  * np.dot( 
                                                                                             np.exp( self.f/self.epsilon ).T, 
                                                                                             np.dot( self.K, np.exp( self.g/self.epsilon ) ) 
                                                                                             )
        return Q
      

  def _update( self, tol = 1e-12, max_iterations = 1000 ):
    """
    
    Parameters:
    -----------
      tol : float
            The tolerance limit for the error. Defaults to 1e-12.
      max_iterations : float
                       The maximum iteration for the optimization algorithm. Defaults to 1000.

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
          objective_values :  list
                              The list of objective values observed after each ascent update.
    """
    i = 0
    while True :
      grad_f, grad_g = self._computegradient()
      # Update f
      self.f = self.f + self.learning_rate * grad_f
      # Error computation 1
      s = np.exp( self.f/self.epsilon ) * np.dot( self.K, np.exp( self.g/self.epsilon ) )
      self.err_a.append( np.linalg.norm( s - self.a ) )
      # Update g
      self.g = self.g + self.learning_rate * grad_g
      # Error computation 2
      r = np.exp( self.g/self.epsilon ) * np.dot( self.K .T, np.exp( self.f/self.epsilon ) )
      self.err_b.append( np.linalg.norm( r - self.b ) )
      # Evaluating objective function after the ascent update
      self.obj.append( self._objectivefunction() )
      # Checks to termminate the iterations
      iter_condition = ( self.err_a[-1] > tol or self.err_b[-1] > tol )
      if iter_condition and i < max_iterations :
          i += 1
      else:
        print( "Terminating after iteration: ", i )
        break  
    # end for
    return {
      'potential_f'       : self.f,
      'potential_g'       : self.g,
      'error_a'           : self.err_a,
      'error_b'           : self.err_b,
      'objective_values'  : self.obj
      
    }