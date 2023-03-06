import numpy as np
from numpy import linalg as Lin
import scipy as sc

class L_BFGS_B:
    def __init__(self,K,a,b,f,g,epsilon):
        self.K = K
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.err_a = []
        self.err_b = []
        self.x = np.vstack((f,g))
        
        self.objvalues=[]

    def _computegradient(self,x):
        """Computes Gradient with respect to g"""
        gradient = np.vstack( (-(self.a-(np.exp(x[:self.a.shape[0]]/self.epsilon)*np.dot(self.K,np.exp(x[self.a.shape[0]:]/self.epsilon))).reshape(x[:self.a.shape[0]].shape[0],-1)),
                    -(self.b-(np.exp(x[self.a.shape[0]:]/self.epsilon)*np.dot(self.K.T,np.exp(x[:self.a.shape[0]]/self.epsilon))).reshape(x[self.a.shape[0]:].shape[0],-1))) )
        return gradient

    def _objectivefunction(self,x):
        """Computes the value of the objective function at x"""
        objective = -np.dot(x[:self.a.shape[0]].T,self.a)-np.dot(x[self.a.shape[0]:].T,self.b)
        regulariser = self.epsilon*np.dot(np.exp(x[:self.a.shape[0]]/self.epsilon).T,np.dot(self.K,np.exp(x[self.a.shape[0]:]/self.epsilon)))
        return objective+regulariser

    
    def _update(self, tol = 1e-12,maxiter = 1000, maxf = 1,maxiter_lbgfs = 1,pgt = 1e-20,maxl = 10,factor = 100000000.0):
        
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
          s = np.exp(self.x[:self.a.shape[0]]/self.epsilon)*np.dot(self.K,np.exp(self.x[self.a.shape[0]:]/self.epsilon))
          self.err_a.append(Lin.norm(s - self.a))


          # error computation 2
          
          r = np.exp(self.x[self.a.shape[0]:]/self.epsilon)*np.dot(self.K .T, np.exp(self.x[:self.a.shape[0]]/self.epsilon))
          self.err_b.append(Lin.norm(r - self.b))

          #Calculating Objective values
          self.objvalues.append((np.dot(self.x[:self.a.shape[0]].T,self.a)+np.dot(self.x[self.a.shape[0]:].T,self.b)-self.epsilon*np.dot(np.exp(self.x[:self.a.shape[0]]/self.epsilon).T,np.dot(self.K,np.exp(self.x[self.a.shape[0]:]/self.epsilon)))))

          if i<maxiter and (self.err_a[-1]>tol or self.err_b[-1]>tol) :
                 i += 1
          else:
                break
      

        # end for  
        return {
            'potential_f' : self.x[:self.a.shape[0]],
            'potential_g' : self.x[self.a.shape[0]:],
            'error_a' : self.err_a,
            'error_b' : self.err_b,
            'objectives' : self.objvalues

        }