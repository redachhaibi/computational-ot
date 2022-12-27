import numpy as np
from numpy import linalg as Lin
import logging
class DampedNewton:
      def __init__(self,K,a,b,f,g,epsilon,rho,c,reg_matrices):
        self.K=K
        self.a=a
        self.b=b
        self.epsilon=epsilon
        self.x=np.hstack((f,g))
        self.rho=rho
        self.c=c
        self._reg_matrices=reg_matrices
        self.alpha=[]
        self.err_a=[]
        self.err_b=[] 
        self.objvalues=[]


      def _computegradientf(self,f):
        """Computes Gradient with respect to f"""
        u = np.exp(f/self.epsilon)
        v = np.exp(self.x[:,1]/self.epsilon)
        return self.a-(u*np.dot(self.K,v)).reshape(f.shape[0],-1)

 
      def _computegradientg(self,g):
        """Computes Gradient with respect to g"""
        return (self.b-(np.exp(g/self.epsilon)*np.dot(self.K.T,np.exp(self.x[:,0]/self.epsilon))).reshape(g.shape[0],-1))

      def _objectivefunction(self,x):
        """Computes the value of the objective function at x"""
        f=x[:,0]
        g=x[:,1]
        return np.dot(f.T,self.a)+np.dot(g.T,self.b)-self.epsilon*np.dot(np.exp(f/self.epsilon).T,np.dot(self.K,np.exp(g/self.epsilon)))

      def _wolfe1(self,alpha,p,slope):#Armijo Condition
          """Backtracking""" 
          
          reduction_count = 0
          while True:
            condition = self._objectivefunction(self.x+alpha*p)<self._objectivefunction(self.x)+self.c*alpha*slope
            if condition:
              alpha = self.rho*alpha
              reduction_count += 1
            else:
              break
          return alpha

      

      def _update(self,tol=1e-12, maxiter=100):
        
        i=0
        while True :
            grad_f=self._computegradientf(self.x[:,0])
            grad_g=self._computegradientg(self.x[:,1])
        
            gradient=np.vstack((grad_f,grad_g))
            
            
            # Regularize
            u = np.exp(self.x[:,0]/self.epsilon)
            v = np.exp(self.x[:,1]/self.epsilon)
            
            r1 = u*np.dot(self.K,v)
            r2 = v*np.dot(self.K.T,u)
            P  = u*self.K*(v.T)

            A = np.diag( np.array(r1.reshape(r1.shape[0],)) )
            B = P
            C = P.T
            D = np.diag( np.array(r2.reshape(r2.shape[0],)) )
            result = np.vstack( ( np.hstack((A,B)), np.hstack((C,D)) ) )

            self.Hessian = -result/self.epsilon
            # Inflating the corresponding direction
            mean_eig = -(0.5*np.mean( r1 ) + 0.5*np.mean( r2 ))/self.epsilon
            self.Hessian_stabilized = self.Hessian + mean_eig*self._reg_matrices[0]
            
            # Preconditioning step
            self.Hessian_stabilized=np.dot(np.dot( self._reg_matrices[1],self.Hessian_stabilized), self._reg_matrices[1])
            gradient=np.dot( self._reg_matrices[1],gradient)
              
            try:
              p_k=-np.linalg.solve( self.Hessian_stabilized,gradient)

            except:
              print("Inverse does not exist at epsilon:",self.epsilon)
              return np.zeros(6)

            # Stacked
            p_k_stacked = np.hstack((p_k[:self.a.shape[0]],p_k[self.a.shape[0]:]))

            # Wolfe Condition 1: Armijo Condition  
            slope = np.dot( p_k.T, gradient)[0][0]
            alpha = 1
            alpha = self._wolfe1( alpha, p_k_stacked, slope)
            self.alpha.append( alpha )

            # Update x = f and g
            self.x = self.x + alpha*p_k_stacked
          
            # error computation 1
            s = np.exp(self.x[:,0]/self.epsilon)*np.dot(self.K,np.exp(self.x[:,1]/self.epsilon))
            self.err_a.append(Lin.norm(s - self.a))

            # error computation 2
            r = np.exp(self.x[:,1]/self.epsilon)*np.dot(self.K .T, np.exp(self.x[:,0]/self.epsilon))
            self.err_b.append(Lin.norm(r - self.b))

            # Calculating Objective values
            value = self._objectivefunction(self.x)
            self.objvalues.append(value)
            
            if i<maxiter and (self.err_a[-1]>tol or self.err_b[-1]>tol) :
                 i+=1
            else:
                print("Terminating after iteration: ",i+1)
                break
      
        # end for    
        return self.x[:,0],self.x[:,1],self.err_a,self.err_b ,self.objvalues,self.alpha
        
#Footer
