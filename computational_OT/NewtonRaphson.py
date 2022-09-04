
from distutils import errors
import numpy as np

class NewtonRaphson:
    def __init__(self,x,K,a,b,epsilon):
        self.x=x
        self.K=K
        self.a=a
        self.b=b
        self.N1=a.shape[0]
        self.N2=b.shape[0]
        self.epsilon=epsilon
        self.err_a=[]
        self.err_b=[]
    
    def _func_phi(self):
        y = np.exp(self.x/self.epsilon)
        u = y[:self.N1]
        v = y[self.N1:]
        r1 = np.dot(np.dot(np.diag(u),self.K),v)
        r2 = np.dot(np.dot(np.diag(v),self.K.T),u)
        return np.hstack( (r1-self.a, r2-self.b) )

    def _func_jacobian(self, debug=False):
        eig_vector = np.hstack( (np.ones(self.N1), -np.ones(self.N2)) )/np.sqrt( self.N1 + self.N2)

        y = np.exp(self.x/self.epsilon)
        u = y[:self.N1]
        v = y[self.N1:]
        #
        r1 = np.dot(np.dot(np.diag(u),self.K),v)
        r2 = np.dot(np.dot(np.diag(v),self.K.T),u)
        P  = np.dot(np.dot(np.diag(u),self.K),np.diag(v))
        # Form matrix 
        # [ A,B
        #   C,D ]
        # = 
        # [ diag(r1),P
        #   P.T,diag(r2) ]    
        A = np.diag( r1 )
        B = P
        C = P.T
        D = np.diag( r2 )
        result = np.vstack( ( np.hstack((A,B)), np.hstack((C,D)) ) )/self.epsilon
        # Conjecture: Smallest eigenvalue in absolute value has eigenvector approx (\mathds{1}_n, -\mathds{1}_m)
        if debug:
            eig, v = np.linalg.eig( result )
            min_index = np.argmin( np.abs(eig) )
            min_value = eig[ min_index ]
            min_vector = v[:, min_index]
            min_vector = min_vector/min_vector[0]
            #
            print("Min absolute eigenvalues: ", min_value)
            print("Norm of v-1: ", np.linalg.norm(min_vector-eig_vector))
            print("")
        # Inflating the corresponding direction
        result = result + np.dot( eig_vector, eig_vector.T)
        return result
    
    def _update(self,maxiter=200,tol=1e-15):
        i=0
        while True:
            target   = self._func_phi()
            jacobian = self._func_jacobian()
            e = [np.linalg.norm( target[:self.N1]),np.linalg.norm( target[self.N1:])]
            self.x=self.x - np.linalg.solve( jacobian, target )
            # inv_jac  = np.linalg.inv( jacobian)
            # x = x - np.dot( inv_jac, target )
            self.err_a.append( e[0])
            self.err_b.append(e[1])

            iter_condition=(e[0]>tol or e[1]>tol)
            if iter_condition and i< maxiter:
                i+=1
            else:
                break

        return self.err_a,self.err_b

        
        

    
            