from turtle import pen
import numpy as np
from numpy import linalg as Lin


class Sinkhorn:

  def __init__(self,N,K,a,b,u,v,epsilon):
    self.K=K
    self.a=a
    self.b=b
    self.u=u
    self.v=v
    self.epsilon=epsilon
    self.err_a=[]
    self.err_b=[]
    self.obj=[]
    
  def _objectivefunction(self):
        """Computes the value of the objective function at x"""
        f=np.log(self.u)*self.epsilon
        g=np.log(self.v)*self.epsilon
        target=np.dot(f.T,self.a)+np.dot(g.T,self.b)
        penalization=-self.epsilon*np.dot(np.exp(f/self.epsilon).T,np.dot(self.K,np.exp(g/self.epsilon)))
        return target, penalization


  def _update(self, tol=1e-12, maxiter=1000):
    i=0
    while True :
      # sinkhorn step 1
      self.u = self.a / np.dot( self.K, self.v )
      
      # error computation 1
      r = self.v*np.dot( self.K.T, self.u)
      self.err_b.append(Lin.norm(r - self.b))

      # sinkhorn step 2
      self.v = self.b / np.dot( self.K.T, self.u )
      
      # error computation 2
      s = self.u*np.dot( self.K, self.v )
      self.err_a.append(Lin.norm(s - self.a))
      self.obj.append(self._objectivefunction())
    
      if i<maxiter and (self.err_a[-1]>tol or self.err_b[-1]>tol) :
          i+=1
      else:
        break
    # end for
    return self.u,self.v,self.err_a,self.err_b,self.obj
    