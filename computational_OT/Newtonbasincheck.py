import numpy as np
import traceback
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
    
    def _getthirdderivative(self):
        Q11=(-1/self.epsilon**2)*np.diag(np.exp(self.f/self.epsilon)*np.dot(self.K,np.exp(self.g/self.epsilon)))
        Q12=(-1/self.epsilon**2)*(np.exp(self.f/self.epsilon)*self.K*(np.exp(self.g/self.epsilon).T))
        Q13=(-1/self.epsilon**2)*(np.exp(self.f/self.epsilon)*self.K*(np.exp(self.g/self.epsilon).T))
        Q14=(-1/self.epsilon**2)*(np.exp(self.f/self.epsilon)*self.K*(np.exp(self.g/self.epsilon).T))
        Q21= (-1/self.epsilon**2)*(np.exp(self.g/self.epsilon)*(self.K.T)*(np.exp(self.f/self.epsilon).T))
        Q22=(-1/self.epsilon**2)*(np.exp(self.g/self.epsilon)*(self.K.T)*(np.exp(self.f/self.epsilon).T))
        Q22=((-1/self.epsilon**2)*(np.exp(self.g/self.epsilon)*(self.K.T)*(np.exp(self.f/self.epsilon).T)))
        Q23=(-1/self.epsilon**2)*np.diag(np.exp(self.g/self.epsilon)*np.dot(self.K.T,np.exp(self.f/self/self.epsilon)))
        Q24=(-1/self.epsilon**2)*(np.exp(self.g/self.epsilon)*(self.K.T)*(np.exp(self.f/self.epsilon).T))

        Derivative=np.zeros((Q11.shape[0]+Q12.shape[0]+Q13.shape[0]+Q14.shape[0].Q11.shape[1]+Q21.shape[1]))
        Derivative[:Q11.shape[0]][:Q11.shape[[1]]]=Q11
        Derivative[Q11.shape[0]:Q12.shape[0]][:Q11.shape[[1]]]=Q12
        Derivative[Q12.shape[0]:Q13.shape[0]][:Q11.shape[[1]]]=Q13
        Derivative[Q13.shape[0]:][:Q11.shape[[1]]]=Q14
        Derivative[:Q21.shape[0]][Q11.shape[[1]]:]=Q21
        Derivative[Q21.shape[0]:Q22.shape[0]][Q11.shape[[1]]:]=Q22
        Derivative[Q22.shape[0]:Q23.shape[0]][Q11.shape[[1]]:]=Q23
        Derivative[Q23.shape[0]:][Q11.shape[[1]]:]=Q24

        return Derivative





    def _getOperatorNorm(self,A):
        singularvalues,_=np.sqrt(np.linalg.eig(np.dot(A.T,A)))
        return np.max(singularvalues)
    
    def _Checkbasinofattraction(self):
        rng = np.random.default_rng()
        pointsa = rng.random((self.a.shape[0], self.a.shape[0])) 
        pointsa =np.concatenate((pointsa,self.f))
        Da=spatial.ConvexHull(pointsa)

        
        pointsb = rng.random((self.b.shape[0], self.b.shape[0])) 
        pointsb=np.concatenate((pointsb,self.g))
        Db=spatial.ConvexHull(pointsb)



        HessianQ=self._getHesianQ()
        HessianDerivative=self._getthirdderivative()
        try:
            invHessianQ=np.linalg.inv(HessianQ)
        except:
            print("Matrix is Singular.")
            return False

        Lambda=self._getOperatorNorm(HessianDerivative)
        kappa=self._getOperatorNorm((invHessianQ))
        delta=self._getOperatorNorm(np.dot(invHessianQ,HessianQ))
        if delta <=1/(2*Lambda*kappa):
            h=2*Lambda*kappa*delta
            d=2*((1-np.sqrt(1-h))*delta)/h
            # max=2*((1-np.sqrt(1-h))*delta)/h
            # for simplex in Da.simplices:
            #     if max < np.linalg.norm(pointsa[simplex, :],self.f):
            #         return False

            # for simplex in Db.simplices:
            #     if max < np.linalg.norm(pointsa[simplex, :],self.g):
            #         return False 

            return True


        






           

          


        



    


    



    