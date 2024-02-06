import numpy as np


class Gradient_Ascent:

  def __init__(self,K,a,b,f,g,epsilon,learning_rate):
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
    grad_f = self.a-np.exp(self.f/self.epsilon)*np.dot(self.K,np.exp(self.g/self.epsilon))
    grad_g = self.b-np.exp(self.g/self.epsilon)*np.dot(self.K.T,np.exp(self.f/self.epsilon))
    return grad_f,grad_g
    
  def _objectivefunction(self):
        """Computes the value of the objective function at x"""
        return np.dot(self.f.T,self.a)+np.dot(self.g.T,self.b)-self.epsilon*np.dot(np.exp(self.f/self.epsilon).T,np.dot(self.K,np.exp(self.g/self.epsilon)))
      

  def _update(self, tol=1e-12, maxiter=1000):
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
    # return self.f,self.g, self.err_a,self.err_b,self.obj
    return {
      'potential_f' : self.f,
      'potential_g' : self.g,
      'error_a' : self.err_a,
      'error_b' : self.err_b,
      'objectives' : self.obj
      
    }