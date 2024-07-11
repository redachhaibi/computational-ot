import numpy as np

class LineSearch:

      def __init__(self,K,a,b,f,g,epsilon,rho,rho_inc,c1,z):
        """
        
        Args:
            K : The Gibb's kernel of size n by m.
            a : The measure a.
            b : The measure b.
            f : The initial Kantorovich potential f.
            g : The initial Kantorovich potential g.
            epsilon : The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
            rho : Damping factor for the line search update step.
            rho_inc (_type_): Factor to increase the line search update step
            c1 (_type_): Damping factor for the slope in the Armijo's condition.
            z (_type_): Initial update step.
        """
        self.K = K
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.x = np.hstack((f,g))
        self.rho = rho
        self.rho_inc = rho_inc
        self.c1 = c1
        self.z = z
        self.alpha = []
        self.err_a = []
        self.err_b = []
        self.objvalues = []
      
      def _computegradientf(self,f):
        """Computes Gradient with respect to f"""
        return (self.a-(np.exp(f/self.epsilon)*np.dot(self.K,np.exp(self.x[:,1]/self.epsilon))).reshape(f.shape[0],-1))

      def _computegradientg(self,g):
        """Computes Gradient with respect to g"""
        return (self.b-(np.exp(g/self.epsilon)*np.dot(self.K.T,np.exp(self.x[:,0]/self.epsilon))).reshape(g.shape[0],-1))

      def _objectivefunction(self,x):
        """Computes the value of the objective function at x"""
        f = x[:,0]
        g = x[:,1]
        return np.dot(f.T,self.a)+np.dot(g.T,self.b)-self.epsilon*np.dot(np.exp(f/self.epsilon).T,np.dot(self.K,np.exp(g/self.epsilon)))
      
      def _wolfe1(self,alpha,p,slope):#Armijo Condition
          """
          
            Backtracking
            Args:
              alpha : The step size to update the potentials towards the optimal direction.
              p : The optimal direction.
              slope : It is the inner product between the gradient and p.
            Returns:
              alpha: The updated step size. 
          """ 
          reduction_count = 0
          while True:
            condition = self._objectivefunction(self.x+alpha*p) < self._objectivefunction(self.x)+self.c1*alpha*slope
            if condition:
              alpha = self.rho*alpha
              reduction_count += 1
            else:
              break

          condition_inc = self._objectivefunction(self.x+(self.rho_inc*alpha)*p) >= self._objectivefunction(self.x)+self.c1*(self.rho_inc*alpha)*slope
          if reduction_count == 0 and condition_inc:
            alpha=self.rho_inc*alpha

          return alpha

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
            linesearch_steps : The list of step size along the iterations of the algorithm.

        """
        i = 0
        while True :
            grad_f = self._computegradientf(self.x[:,0])
            grad_g = self._computegradientg(self.x[:,1])
        
            gradient = np.vstack((grad_f,grad_g))
            
            slope = np.dot(gradient.T,gradient)
            p_k = np.hstack((grad_f,grad_g))

            self.alpha.append(self.z)
            if i != 0:
              self.alpha[i] = self.alpha[i-1]

            # Wolfe Condition 1:Armijo Condition  
            self.alpha[i] = self._wolfe1(self.alpha[i],p_k,slope)

            #Updating f
            self.x[:,0] = self.x[:,0]+self.alpha[i]*p_k[:,0]
           
            # error computation 1
            s = np.exp(self.x[:,0]/self.epsilon)*np.dot(self.K,np.exp(self.x[:,1]/self.epsilon))
            self.err_a.append(np.linalg.norm(s - self.a))


            #updating g
            self.x[:,1] = self.x[:,1]+self.alpha[i]*p_k[:,1]
            # error computation 2
            r = np.exp(self.x[:,1]/self.epsilon)*np.dot(self.K .T, np.exp(self.x[:,0]/self.epsilon))
            self.err_b.append(np.linalg.norm(r - self.b))

            #Calculating Objective values
            self.objvalues.append(self._objectivefunction(self.x))
            
            if i < maxiter and (self.err_a[-1]>tol or self.err_b[-1]>tol) :
                 i += 1
            else:
              print("Terminating after iteration: ",i)
              break 
      
        # end for    
        return {
          'potential_f' : self.x[:,0],
          'potential_g' : self.x[:,1],
          'error_a' : self.err_a,
          'error_b' : self.err_b,
          'objectives' : self.objvalues,
          'linesearch_steps' : self.alpha
           
        }
        