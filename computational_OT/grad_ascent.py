import numpy as np


class Gradient_Ascent:

  def __init__(self,K,a,b,f,g,epsilon,learning_rate):
    """
    
    Args:
        K : The Gibb's kernel of size n by m.
        a : The measure a.
        b : The measure b.
        f : The initial Kantorovich potential f.
        g : The initial Kantorovich potential g.
        epsilon : The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
        learning_rate : The step size of the update towards optimal direction.
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
  
  def _computegradient(self):
    """ Computes the gradient"""
    grad_f = self.a-np.exp(self.f/self.epsilon)*np.dot(self.K,np.exp(self.g/self.epsilon))
    grad_g = self.b-np.exp(self.g/self.epsilon)*np.dot(self.K.T,np.exp(self.f/self.epsilon))
    return grad_f, grad_g
    
  def _objectivefunction(self):
        """Computes the value of the objective function at x"""
        return np.dot(self.f.T,self.a)+np.dot(self.g.T,self.b)-self.epsilon*np.dot(np.exp(self.f/self.epsilon).T,np.dot(self.K,np.exp(self.g/self.epsilon)))
      

  def _update(self, tol=1e-12, maxiter=1000):
    """
    
    Args:
      tol  : The tolerance limit for the error. Defaults to 1e-12.
      maxiter  : The maximum iteration for the optimization algorithm. Defaults to 1000.

    Returns:
      potential_f : The optimal Kantorovich potential f.
      potential_g : The optimal Kantorovich potential g.
      error_a : The list of error of the estimation of the measure 'a' over the iteration of the algorithm.
      error_b : The list of error of the estimation of the measure 'b' over the iteration of the algorithm.
      objectives  : The list of objective function values over the iterations of the algorithm.
    """
    i = 0
    while True :
      grad_f,grad_g = self._computegradient()
      # Update f
      self.f = self.f+self.learning_rate*grad_f
      # error computation 1
      s = np.exp(self.f/self.epsilon)*np.dot(self.K,np.exp(self.g/self.epsilon))
      self.err_a.append(np.linalg.norm(s - self.a))
      # Update g
      self.g = self.g+self.learning_rate*grad_g
      # error computation 2
      r = np.exp(self.g/self.epsilon)*np.dot(self.K .T, np.exp(self.f/self.epsilon))
      self.err_b.append(np.linalg.norm(r - self.b))
      self.obj.append(self._objectivefunction())

      iter_condition = (self.err_a[-1]>tol or self.err_b[-1]>tol)
      if iter_condition and i<maxiter :
          i += 1
      else:
        print("Terminating after iteration: ",i)
        break  
    # end for
    return {
      'potential_f' : self.f,
      'potential_g' : self.g,
      'error_a' : self.err_a,
      'error_b' : self.err_b,
      'objectives' : self.obj
      
    }