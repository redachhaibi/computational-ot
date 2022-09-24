import numpy as np
from scipy import spatial
class Newtonbasincheck:
    def __init__(self,K,a,b,u,v,epsilon):
        self.K=K
        self.a=a
        self.b=b
        self.f=epsilon*np.log(u)
        self.g=epsilon*np.log(v)
        self.epsilon=epsilon
        self.err_a=[]
        self.err_b=[]
        self.obj=[]

    def _getgradient(self):
        grad_f=self.a-np.exp(self.f/self.epsilon)*np.dot(self.K,np.exp(self.g/self.epsilon))
        grad_g=self.b-np.exp(self.g/self.epsilon)*np.dot(self.K.T,np.exp(self.f/self.epsilon))
        return grad_f,grad_g
    
    def _objectivefunction(self):
        """Computes the value of the objective function at x"""
        return np.dot(self.f.T,self.a)+np.dot(self.g.T,self.b)-self.epsilon*np.dot(np.exp(self.f/self.epsilon).T,np.dot(self.K,np.exp(self.g/self.epsilon)))
      


    def  _getHesianQ(self):
        Q11=(-1.0/self.epsilon)*np.diag(np.exp(self.f/self.epsilon)*np.dot(self.K,np.exp(self.g/self.epsilon)))
        Q12=(-1.0/self.epsilon)*(np.exp(self.f/self.epsilon)*self.K*(np.exp(self.g/self.epsilon).T))
        Q21=Q12.T
        Q22=(-1.0/self.epsilon)*np.diag(np.exp(self.g/self.epsilon)*np.dot(self.K.T,np.exp(self.f/self.epsilon)))
        
        HessianQ=np.zeros((Q11.shape[0]+Q21.shape[0],Q11.shape[1]+Q12.shape[1]))
    
        HessianQ[:Q11.shape[0],:Q11.shape[1]]=Q11
        HessianQ[Q11.shape[0]:,:Q11.shape[1]]=Q12
        HessianQ[:Q11.shape[0],Q11.shape[1]:]=Q21
        HessianQ[Q11.shape[0]:,Q11.shape[1]:]=Q22
        
        return HessianQ 
    
    def _getthirdderivative(self):
        Q11=(-1.0/self.epsilon**2)*np.diag(np.exp(self.f/self.epsilon)*np.dot(self.K,np.exp(self.g/self.epsilon)))
        Q12=(-1.0/self.epsilon**2)*(np.exp(self.f/self.epsilon)*self.K*(np.exp(self.g/self.epsilon).T))
        Q13=(-1.0/self.epsilon**2)*(np.exp(self.f/self.epsilon)*self.K*(np.exp(self.g/self.epsilon).T))
        Q14=(-1.0/self.epsilon**2)*(np.exp(self.f/self.epsilon)*self.K*(np.exp(self.g/self.epsilon).T))
        Q21= (-1.0/self.epsilon**2)*(np.exp(self.g/self.epsilon)*(self.K.T)*(np.exp(self.f/self.epsilon).T))
        Q22=(-1.0/self.epsilon**2)*(np.exp(self.g/self.epsilon)*(self.K.T)*(np.exp(self.f/self.epsilon).T))
        Q22=((-1.0/self.epsilon**2)*(np.exp(self.g/self.epsilon)*(self.K.T)*(np.exp(self.f/self.epsilon).T)))
        Q23=(-1.0/self.epsilon**2)*np.diag(np.exp(self.g/self.epsilon)*np.dot(self.K.T,np.exp(self.f/self.epsilon)))
        Q24=(-1.0/self.epsilon**2)*(np.exp(self.g/self.epsilon)*(self.K.T)*(np.exp(self.f/self.epsilon).T))

        Derivative=np.zeros((Q11.shape[0]+Q12.shape[0]+Q13.shape[0]+Q14.shape[0],Q11.shape[1]+Q21.shape[1]))
        Derivative[:Q11.shape[0],:Q11.shape[1]]=Q11
        Derivative[Q11.shape[0]:Q11.shape[0]+Q12.shape[0],:Q11.shape[1]]=Q12
        Derivative[Q11.shape[0]+Q12.shape[0]:Q11.shape[0]+Q12.shape[0]+Q13.shape[0],:Q11.shape[1]]=Q13
        Derivative[Q11.shape[0]+Q12.shape[0]+Q13.shape[0]:,:Q11.shape[1]]=Q14
        Derivative[:Q21.shape[0],Q11.shape[1]:]=Q21
        Derivative[Q21.shape[0]:Q21.shape[0]+Q22.shape[0],Q11.shape[1]:]=Q22
        Derivative[Q21.shape[0]+Q22.shape[0]:Q21.shape[0]+Q22.shape[0]+Q23.shape[0],Q11.shape[1]:]=Q23
        Derivative[Q21.shape[0]+Q22.shape[0]+Q23.shape[0]:,Q11.shape[1]:]=Q24

        return Derivative





    def _getOperatorNorm(self,A):
        singularvalues=np.sqrt(np.linalg.eig(np.dot(A.T,A))[0])
        return np.min(singularvalues)
    
    def _Checkbasinofattraction(self):

        HessianQ=self._getHesianQ()
        HessianDerivative=self._getthirdderivative()
        gradf,gradg=self._getgradient()
        gradQ=np.hstack((gradf,gradg)).reshape(self.f.shape[0]+self.g.shape[0],1)
        try:
            invHessianQ=np.linalg.inv(HessianQ)
        except:
            print("Matrix is Singular.")
            return False
        
        Lambda=self._getOperatorNorm(HessianDerivative)
        kappa=self._getOperatorNorm((invHessianQ))
        delta=self._getOperatorNorm(np.dot(invHessianQ,gradQ))
        h=2*Lambda*kappa*delta
        print("h: ", np.abs(h), " Lambda: ",Lambda, " kappa: ",kappa, " delta: ",delta)
        if np.abs(h)<=1:
            print("Hi")
            d=(2/h)*((1-np.sqrt(1-h))*delta)

            if self._getOperatorNorm(-np.dot(invHessianQ,HessianQ))<np.abs(d):
                return True

        else:
            False


        






           

          


        



    


    



    