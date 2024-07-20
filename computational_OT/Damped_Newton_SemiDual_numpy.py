import numpy as np

class DampedNewton_SemiDual_np:
    def __init__(self, C, a, b, f, epsilon, rho, c):
        """
        
        Args:
            C : Cost matrix of size n by m.
            (a,b) : The two measures of the OT problem, the shape of which is (n,1) and (m,1) respectively.
            f : Kantorovich potential f, which is of shape (n,1).
            rho : Damping factor for the line search update step.
            epsilon : The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
            c : Damping factor for the slope in the Armijo's condition.
        """
        self.C = C
        self.a = a
        self.b = b
        self.f = f
        self.rho = rho
        self.epsilon = epsilon  
        self.c = c
        self.alpha_list = []
        self.err = []
        self.objvalues = [] 
        null_vector = np.ones(self.a.shape[0])/np.sqrt(self.a.shape[0])
        self.null_vector = np.reshape(null_vector, (self.a.shape[0],1))
        self.reg_matrix = np.dot( self.null_vector, self.null_vector.T )

    def _objectivefunction(self,f):
        """ 
        Args:
          f: The Kantorovich potential f.
        Returns : Q_semi(f) =  <f,a> + <g(f,C,epsilon),b>.
        """
        # Computing minimum of  C-f for each column of this difference matrix.
        min_f = np.min(self.C-f[:,None],0)# Shape: (m,)
        H = f[:,None]-self.C+min_f[None,:]
        g = -self.epsilon*np.log(np.sum(self.a[:,None]*np.exp(H/self.epsilon),0))+min_f # Shape: (m,)
        Q_semi = np.dot(f, self.a) + np.dot(g, self.b) 
        return Q_semi
      
    def _computegradientf(self):
        """ 
            Compute gradient with respect to f of the objective function Q_semi(.).
        """
        # Here self.g + self.min_f completes the log domain regularization of self.g.
        exponent = (-(self.C-self.f[:,None])+self.min_f[None,:] )+self.g[None,:] 
        gradient = self.a-np.sum( self.a[:,None]*np.exp(exponent/self.epsilon)*self.b[None,:], 1)
        return gradient

    def _wolfe1(self,alpha,p,slope):#Armijo Condition
          """
          
            Backtracking
            Args:
                alpha : The step size to update the potentials towards the optimal direction.
                p : The optimal direction.
                slope : It is the inner product of the gradient and p.
            Returns:
              alpha: The updated step size. 
          """ 
          reduction_count = 0           
          while True:   
            condition = self._objectivefunction( self.f+alpha*p )< self._objectivefunction( self.f )+self.c*alpha*slope
            if condition or np.isnan(self._objectivefunction( self.f+alpha*p )):
              alpha = self.rho*alpha                                                     
              reduction_count += 1
            else:
              break
          return alpha
        
      
    def _update(self, tol=1e-12, maxiter = 100, debug = False):
        """
        
        Args:
            tol  : The tolerance limit for the error. Defaults to 1e-12.
            maxiter  : The maximum iteration for the optimization algorithm. Defaults to 100.
            debug : To add a debug any step of the implementation when needed. Defaults to False.

        Returns:
            potential_f : The optimal Kantorovich potential f.
            potential_g : The optimal Kantorovich potential g.
            error : The list of error values over the iteration of the algorithm.
            objectives  : The list of objective function values over the iterations of the algorithm.
            linesearch_steps : The list of step size along the iterations of the algorithm.
        """
        # Computing minimum of  C-f for each column of this difference matrix.
        self.min_f = np.min(self.C-self.f[:,None],0)
        # We know e^((-(C-f)+self.min_f)/epsilon)<1, therefore the value of self.g below is bounded.
        H = -self.C+self.f[:,None]+self.min_f[None,:]
        self.g = -self.epsilon*np.log( np.sum(self.a[:,None]*np.exp(H/self.epsilon),0) )
        i = 0
        while True: 
            # Compute gradient w.r.t f:
            grad_f = self._computegradientf()
            # Compute the Hessian:
            ### Adding self.min_f in the exponents in M completes the  log-domain regularization of the Hessian.
            exponent = (-(self.C-self.f[:,None])+self.min_f[None,:] )+self.g[None,:] 
            M =  self.a[:,None]*np.exp(exponent/self.epsilon)*np.sqrt( self.b)[None,:]
            S = np.sum(M*np.sqrt(self.b)[None,:],1)
            self.Hessian = S[:,None]*np.identity(self.a.shape[0])-np.dot( M , M.T )  
            ### Regularizing the Hessian using the regularization vector with the factor being being the mean of eigenvalues of the Hessian 
            mean_eig = np.mean(np.linalg.eigh(self.Hessian)[0])
            self.Hessian =  self.Hessian + mean_eig*self.reg_matrix
            self.Hessian = -self.Hessian/self.epsilon
            # Compute solution of Ax = b:
            try:    
                p_k = -np.linalg.solve(self.Hessian, grad_f)
            except:
                print("Inverse does not exist at epsilon:", self.epsilon)   
                return np.zeros(6)
            p_k = p_k - self.null_vector.flatten()*np.dot( self.null_vector.flatten(), p_k )
            # Wolfe condition 1: Armijo Condition:  
            slope = np.dot(p_k, grad_f)
            alpha = 1
            alpha = self._wolfe1(alpha, p_k, slope)
            self.alpha_list.append(alpha)
            # Update f and g:
            self.f = self.f + alpha*p_k
            self.min_f = np.min(self.C-self.f[:,None],0)    
            # Updating the new self.g in the similar way as we did before starting the while loop.
            H = -self.C+self.f[:,None]+self.min_f[None,:]
            self.g = -self.epsilon*np.log( np.sum(self.a[:,None]*np.exp(H/self.epsilon),0) )
            ### Here similar to the Hessian the computation of the coupling P involves addition of the minimum self.min_f completing the log-domian regularization of self.g.
            exponent = (-(self.C-self.f[:,None])+self.min_f[None,:] )+self.g[None,:] 
            P  =   self.a[:,None]*(np.exp(exponent/self.epsilon))*self.b[None,:]
            # Error computation:
            self.err.append(np.linalg.norm(np.sum(P,1)- self.a,1))
            # Calculating objective function:
            value = self._objectivefunction(self.f)
            self.objvalues.append(value)
            # Check error:
            if i< maxiter and ( self.err[-1]>tol ):
                i+=1
            else:   
                print("Terminating after iteration: ",i)
                break 
        # end for                                                                                                            
        return {
            "potential_f"       : self.f.reshape(self.a.shape[0],)+self.epsilon*np.log(self.a).reshape(self.a.shape[0],),
            "potential_g"       : self.g.reshape(self.b.shape[0],)+self.epsilon*np.log(self.b).reshape(self.b.shape[0],)+self.min_f,
            "error"             : self.err,
            "objectives"        : self.objvalues,
            "linesearch_steps"  : self.alpha_list
        }
