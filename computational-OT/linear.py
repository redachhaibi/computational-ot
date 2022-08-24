import numpy as np
import cvxpy as cp

class LinearOptimize:

  def __init__(self,N1,N2,a,b,Cost_mat):
    self.N1=N1
    self.N2=N2
    self.a=a.reshape(a.shape[0],-1)
    self.b=b.reshape(b.shape[0],-1)
    self.C=Cost_mat
  
  def solve(self):
    P = cp.Variable((self.N1,self.N2))
    v = np.ones((self.N2,self.a.shape[1]))
    u = np.ones((self.N1,self.b.shape[1]))
    
    U = [0 <= P, cp.matmul(P,v)==self.a, cp.matmul(P.T,u)==self.b]

    objective = cp.Minimize( cp.sum(cp.multiply(P,self.C)) )

    prob = cp.Problem(objective, U)
    result = prob.solve()
    return P
