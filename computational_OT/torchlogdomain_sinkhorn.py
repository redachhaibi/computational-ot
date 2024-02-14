import torch
torch.set_default_dtype(torch.float64)
class torchLog_domainSinkhorn:
    def __init__(self,a,b,C,epsilon):
        self.a = a
        self.b = b
        self.C = C
        self.epsilon = epsilon
        self.error = []
    
    def mina_u(self,H):
        return -self.epsilon*torch.log(torch.sum(self.a[:,None]*torch.exp(-H/self.epsilon),0))
    def minb_u(self,H): 
        return -self.epsilon*torch.log(torch.sum(self.b[None,:]*torch.exp(-H/self.epsilon),1))
    def mina(self,H):
        return self.mina_u(H-torch.min(H,0).values)+torch.min(H,0).values
    def minb(self,H):
        return self.minb_u(H-torch.min(H,1).values[:,None])+torch.min(H,1).values
    
    def update(self,tol=1e-12,niter=500):
        f,g = self.a.clone(),self.b.clone()
        for i in range(niter):
            g = self.mina(self.C-f[:,None])
            f = self.minb(self.C-g[None,:])
            #generate the coupling
            P = self.a[:,None]*torch.exp((f[:,None]+g[None,:]-self.C)/self.epsilon)*self.b[None,:]
            self.error.append(torch.linalg.norm(torch.sum(P,0)-self.b,1))
            if self.error[i]<tol:
                print("Terminating after iteration: ",i)
                break
        
        #end for
        if i+1>=niter:
            print("Terminating after maximal number of iterations: ",niter)
        return {
            'error'       : self.error,
            'potential_f' : f+self.epsilon*torch.log(self.a).reshape(self.a.shape[0],),
            'potential_g' : g+self.epsilon*torch.log(self.b).reshape(self.b.shape[0],)
        }

