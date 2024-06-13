import numpy as np

class DampedNewton_SemiDual_np:
    def __init__(self, C, a, b, f, epsilon, rho, c):
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
        null_vector = np.hstack(np.ones(self.a.shape[0]))/np.sqrt(self.a.shape[0])
        self.null_vector = np.reshape(null_vector, (self.a.shape[0],1))
        self.reg_matrix = np.dot( self.null_vector, self.null_vector.T )
    
    def _objectivefunction(self,x):
        a_ = self.a.reshape(self.a.shape[0],)
        min_x = np.min(self.C-x,0)
        x = x.reshape(self.a.shape[0],)
        y = -self.epsilon*np.log(np.sum(a_[:,None]*np.exp((x[:,None]-self.C+min_x[None,:])/self.epsilon),0))+min_x[None,:]
        return np.dot(x.T, self.a) + np.dot(y, self.b) 
     
    def _computegradientf(self):
        a_ = self.a.reshape(self.a.shape[0],)
        b_ = self.b.reshape(self.b.shape[0],)
        f_ = self.f.reshape(self.a.shape[0],)
        gradient = self.a-np.sum(a_[:,None]*np.exp((f_[:,None]+self.g[None,:]-self.C+self.min_f[None,:])/self.epsilon)*b_[None,:], 1).reshape(self.a.shape[0],-1)
        return gradient
    
    def _wolfe1(self,alpha,p,slope):#Armijo Condition
          """Backtracking""" 
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
        a_ = self.a.reshape(self.a.shape[0],)
        b_ = self.b.reshape(self.b.shape[0],)
        self.min_f = np.min(self.C-self.f,0)
        f_ = self.f.reshape(self.a.shape[0],)
        self.g = -self.epsilon*np.log(np.sum(a_[:,None]*np.exp((f_[:,None]-self.C+self.min_f[None,:])/self.epsilon),0))
        i = 0
        while True: 
            # Compute gradient w.r.t f:
            grad_f = self._computegradientf()
            # Compute the Hessian:
            M = a_[:,None]*np.exp((f_[:,None]+self.g[None,:]-self.C+self.min_f[None,:])/self.epsilon)*np.sqrt(b_)[None,:]
            self.Hessian = np.sum(M*np.sqrt(b_)[None,:],1)[:,None]*np.identity(self.a.shape[0])-np.dot( M , M.T )   
            mean_eig = -np.mean(np.linalg.eigh(self.Hessian)[0])/self.epsilon
            self.Hessian = -self.Hessian/self.epsilon
            self.Hessian =  self.Hessian + mean_eig*self.reg_matrix
            # Compute solution of Ax = b:
            try:    
                p_k = -np.linalg.solve(self.Hessian, grad_f)
            except:
                print("Inverse does not exist at epsilon:", self.epsilon)
                return np.zeros(6)
            
            p_k = p_k - self.null_vector*np.dot( self.null_vector.flatten(), p_k.flatten() )
            # Wolfe condition 1: Armijo Condition:  
            slope = np.dot(p_k.T, grad_f)[0][0]
            alpha = 1
            alpha = self._wolfe1(alpha, p_k, slope)
            self.alpha_list.append(alpha)
            # Update f and g:
            self.f = self.f + alpha*p_k
            self.min_f = np.min(self.C-self.f,0)
            f_ = self.f.reshape(self.a.shape[0],)
            self.g = -self.epsilon*np.log(np.sum(a_[:,None]*np.exp((f_[:,None]-self.C+self.min_f[None,:])/self.epsilon),0))
            # Error computation:
            P  =  a_[:,None]*(np.exp((f_[:,None]+self.g[None,:]-self.C+self.min_f[None,:])/self.epsilon))*b_[None,:]
            self.err.append(np.linalg.norm(np.sum(P,1)-a_,1))
            # Calculating objective function:
            value = self._objectivefunction(self.f)
            self.objvalues.append(value[0])
            # Check error:
            if i< maxiter and ( self.err[-1]>tol ):
                i+=1
            else:   
                print("Terminating after iteration: ",i)
                break
        return {
            "potential_f"       : self.f.reshape(self.a.shape[0],)+self.epsilon*np.log(self.a).reshape(self.a.shape[0],),
            "potential_g"       : self.g.reshape(self.b.shape[0],)+self.epsilon*np.log(self.b).reshape(self.b.shape[0],)+self.min_f,
            "error"             : self.err,
            "objectives"        : self.objvalues,
            "linesearch_steps"  : self.alpha_list
        }
