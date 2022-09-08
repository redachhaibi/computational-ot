import numpy as np
import traceback
class Newtonbasincheck:
    def __init__(self,K,a,b,f,g,epsilon):
        self.K=K
        self.a=a
        self.b=b
        self.f=f
        self.g=g
        self.epsilon=epsilon
        self.err_a=[]
        self.err_b=[]
        self.obj=[]


    def  _getHesianQ(self):
        Q11=(-1/self.epsilon)*np.diag(np.exp(self.f/self.epsilon)*np.dot(self.K,np.exp(self.g/self/self.epsilon)))
        Q12=(-1/self.epsilon)*(np.exp(self.f/self.epsilon)*K*(np.exp(self.g/self.epsilon).T))
        Q21=Q12.T
        Q22=(-1/self.epsilon)*np.diag(np.exp(self.g/self.epsilon)*np.dot(self.K.T,np.exp(self.f/self/self.epsilon)))
        
        HessianQ=np.zeros((Q11.shape[0]+Q21.shape[0],Q11.shape[1]+Q12.shape[1]))
        HessianQ[:Q11.shape[0]][:Q11.shape[1]]=Q11
        HessianQ[Q11.shape[0]:][:Q11.shape[1]]=Q12
        HessianQ[:Q11.shape[0]][Q11.shape[1]:]=Q21
        HessianQ[Q11.shape[0]:][Q11.shape[1]:]=Q22
        
        return HessianQ 

    def _getOperatorNorm(self,A):
        singularvalues,_=np.sqrt(np.linalg.eig(np.dot(A.T,A)))
        return np.max(singularvalues)
    
    def _Checkbasinofattraction(self):
        HessianQ=self._getHesianQ()
        try:
            invHessianQ=np.linalg.inv(HessianQ)
        except:
            print("Matrix is Singular.")
            return False

        Lambda=self._getOperatorNorm(HessianQ)
        kappa=self._getOperatorNorm((invHessianQ))
        delta=self._getOperatorNorm(np.dot(invHessianQ,HessianQ))
        if 2*kappa*Lambda*delta<=1:
          


        



    


    



    