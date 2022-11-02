import numpy as np
from numpy import linalg as Lin

class LineSearchNewton:


      def __init__(self,K,a,b,f,g,epsilon,rho,rho_inc,c1,c2,z):
        self.K=K
        self.a=a
        self.b=b
        self.epsilon=epsilon
        self.x=np.hstack((f,g))
        self.f=f
        self.g=g
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

      def _objectivefunction(self,c):
        """Computes the value of the objective function at x"""
        f=c[:self.a.shape[0]].reshape(self.a.shape[0],-1)
        g=c[self.a.shape[0]:].reshape(self.b.shape[0],-1)

        return np.dot(f.T,self.a)+np.dot(g.T,self.b)-self.epsilon*np.dot(np.exp(f/self.epsilon).T,np.dot(self.K,np.exp(g/self.epsilon)))
      
      def _wolfe1(self,alpha,p,slope):#Armijo Condition
          """Backtracking""" 
          
          X=np.hstack((self.x[:,0],self.x[:,1]))
          X=X.reshape(X.shape[0],-1)
          reduction_count = 0
          while True:
            condition = self._objectivefunction(X+alpha*p)<self._objectivefunction(X)+self.c1*alpha*slope
            if condition:
              alpha = self.rho*alpha
              reduction_count += 1
            else:
              break

          condition_inc = self._objectivefunction(X+(self.rho_inc*alpha)*p)>=self._objectivefunction(X)+self.c1*(self.rho_inc*alpha)*slope
          if reduction_count==0 and condition_inc:
            alpha=self.rho_inc*alpha
          return alpha
      
      def  _getHesianQ(self):
        f,g=self.x[:,0],self.x[:,1]

        Q11=(-1.0/self.epsilon)*np.diag(np.exp(f/self.epsilon)*np.dot(self.K,np.exp(g/self.epsilon)))
        Q12=(-1.0/self.epsilon)*(np.exp(f/self.epsilon)*self.K*(np.exp(g/self.epsilon).T))
        Q21=Q12.T
        Q22=(-1.0/self.epsilon)*np.diag(np.exp(g/self.epsilon)*np.dot(self.K.T,np.exp(f/self.epsilon)))
        
        HessianQ=np.zeros((Q11.shape[0]+Q21.shape[0],Q11.shape[1]+Q12.shape[1]))
    
        HessianQ[:Q11.shape[0],:Q11.shape[1]]=Q11
        HessianQ[Q11.shape[0]:,:Q11.shape[1]]=Q12
        HessianQ[:Q11.shape[0],Q11.shape[1]:]=Q21
        HessianQ[Q11.shape[0]:,Q11.shape[1]:]=Q22
        
        return HessianQ 
                 

      def _update(self, tol=1e-12, maxiter=1000):
        
        i=0
        while True :
            grad_f=self._computegradientf(self.x[:,0])
            grad_g=self._computegradientg(self.x[:,1])
        
            gradient=np.vstack((grad_f,grad_g))
            
            slope=np.dot(gradient.T,gradient)
            Hessian=self._getHesianQ()
            

            eig_vector = np.hstack( (np.ones(self.a.shape[0]), -np.ones(self.b.shape[0])) )/np.sqrt( self.a.shape[0] + self.b.shape[0])
            eig_vector = np.reshape( eig_vector, (self.a.shape[0] + self.b.shape[0], 1) )

            
            u = np.exp(self.f/self.epsilon)
            v = np.exp(self.g/self.epsilon)
            #
          
            r1 = np.dot((u*self.K),v)
            r2 = np.dot((v*self.K.T),u)
            P  = u*self.K*(v.T)
             
            
            
            # Inflating the corresponding direction
            mean_eig = 0.5*np.mean( r1 ) + 0.5*np.mean( r2 )
            Hessianstabilized = Hessian + mean_eig*np.dot( eig_vector, eig_vector.T)
           

            p_k=np.linalg.solve(Hessianstabilized,-np.vstack((grad_f,grad_g)))

            self.alpha.append(self.z)
            if i!=0:
              self.alpha[i]=self.alpha[i-1]

            # Wolfe Condition 1:Armijo Condition  
            self.alpha[i]=self._wolfe1(self.alpha[i],p_k,slope)

            #Updating f
            self.x[:,0]=self.x[:,0]+self.alpha[i]*p_k[:self.a.shape[0]].reshape(self.a.shape[0],)
           
            # error computation 1
            s = np.exp(self.x[:,0]/self.epsilon)*np.dot(self.K,np.exp(self.x[:,1]/self.epsilon))
            self.err_a.append(Lin.norm(s - self.a))


            #updating g
            self.x[:,1]=self.x[:,1]+self.alpha[i]*p_k[self.a.shape[0]:].reshape(self.b.shape[0],)
            # error computation 2
            r = np.exp(self.x[:,1]/self.epsilon)*np.dot(self.K .T, np.exp(self.x[:,0]/self.epsilon))
            self.err_b.append(Lin.norm(r - self.b))
            #Calculating Objective values
            self.objvalues.append(self._objectivefunction(np.hstack((self.x[:,0],self.x[:,1]))))
            
            if i<maxiter and (self.err_a[-1]>tol or self.err_b[-1]>tol) :
                 i+=1
            else:
                print("Terminating after teration: ",i+1)
                break
      
        # end for    
        return self.x[:,0],self.x[:,1],self.err_a,self.err_b ,self.objvalues,self.alpha
        