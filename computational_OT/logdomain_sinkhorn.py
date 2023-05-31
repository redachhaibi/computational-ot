import numpy as np
import time
class Log_domainSinkhorn:
    def __init__(self, a, b, C, epsilon):
        self.a       = a
        self.b       = b
        self.C       = C
        self.epsilon = epsilon
        self.error   = []
    
    def mina_u(self,H):
        return -self.epsilon*np.log( np.sum(self.a[:,None] * np.exp(-H/self.epsilon),0) )
    
    def minb_u(self,H):
        return -self.epsilon*np.log( np.sum(self.b[None,:] * np.exp(-H/self.epsilon),1) )  
    
    def mina(self,H):
        return self.mina_u(H-np.min(H,0)) + np.min(H,0);
    def minb(self,H):
        return self.minb_u(H-np.min(H,1)[:,None]) + np.min(H,1)
    
    def update(self, tol = 1e-14, niter = 500):   
        f,g = self.a, self.b
        for i in range(niter):
            g = self.mina(self.C-f[:,None])
            f = self.minb(self.C-g[None,:]) 
            # generate the coupling
            start = time.time()
            P = self.a[:,None]*np.exp((f[:,None]+g[None,:]-self.C)/self.epsilon)*self.b[None,:] # line (*)
            end = time.time()
            print("P time: ", 1e-3*(end-start))
            # check conservation of mass
            self.error.append( np.linalg.norm(np.sum(P,0)-self.b,1) )
            if self.error[i] < tol:
                print("Terminating after iteration: ",i)
                break
        return {
            'error'      : self.error,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
            'potential_f': f+self.epsilon*np.log(self.a).reshape(self.a.shape[0],),
            'potential_g': g+self.epsilon*np.log(self.b).reshape(self.b.shape[0],)  #Change of convention because of line (*)
        }


