import numpy as np
from numpy import linalg as Lin

class LineSearch:


      def __init__(self,K,a,b,f,g,epsilon,rho,rho_inc,c1,c2,z):
        self.K=K
        self.a=a
        self.b=b
        self.epsilon=epsilon
        self.x=np.hstack((f,g))
        self.rho=rho
        self.rho_inc=rho_inc
        self.c1=c1
        self.c2=c2
        self.z=z
        self.alpha=[]
        self.err_a=[]
        self.err_b=[]
        self.objvalues=[]
      
      def _computegradientf(self,f):
        """Computes Gradient with respect to f"""
        return (self.a-(np.exp(f/self.epsilon)*np.dot(self.K,np.exp(self.x[:,1]/self.epsilon))).reshape(f.shape[0],-1))

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
            condition = self._objectivefunction(self.x+alpha*p)<self._objectivefunction(self.x)+self.c1*alpha*slope
            if condition:
              alpha = self.rho*alpha
              reduction_count += 1
            else:
              break

          condition_inc = self._objectivefunction(self.x+(self.rho_inc*alpha)*p)>=self._objectivefunction(self.x)+self.c1*(self.rho_inc*alpha)*slope
          if reduction_count==0 and condition_inc:
            alpha=self.rho_inc*alpha

          return alpha

        

      

            

      def _update(self, tol=1e-12, maxiter=1000):
        
        i=0
        while True :
            grad_f=self._computegradientf(self.x[:,0])
            grad_g=self._computegradientg(self.x[:,1])
        
            gradient=np.vstack((grad_f,grad_g))
            
            slope=np.dot(gradient.T,gradient)
            p_k=np.hstack((grad_f,grad_g))

            self.alpha.append(self.z)
            if i!=0:
              self.alpha[i]=self.alpha[i-1]

            # Wolfe Condition 1:Armijo Condition  
            self.alpha[i]=self._wolfe1(self.alpha[i],p_k,slope)

            #Updating f
            self.x[:,0]=self.x[:,0]+self.alpha[i]*p_k[:,0]
           
            # error computation 1
            s = np.exp(self.x[:,0]/self.epsilon)*np.dot(self.K,np.exp(self.x[:,1]/self.epsilon))
            self.err_a.append(Lin.norm(s - self.a))


            #updating g
            self.x[:,1]=self.x[:,1]+self.alpha[i]*p_k[:,1]
            # error computation 2
            r = np.exp(self.x[:,1]/self.epsilon)*np.dot(self.K .T, np.exp(self.x[:,0]/self.epsilon))
            self.err_b.append(Lin.norm(r - self.b))

            #Calculating Objective values
            self.objvalues.append(self._objectivefunction(self.x))
            
            if i<maxiter and (self.err_a[-1]>tol or self.err_b[-1]>tol) :
                 i+=1
            else:
                break
      
        # end for    
        return self.x[:,0],self.x[:,1],self.err_a,self.err_b ,self.objvalues,self.alpha
        