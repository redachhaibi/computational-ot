import numpy as np
from numpy import linalg as Lin
import logging
import time

class DampedNewton_With_Preconditioner:
      def __init__(self,K,a,b,f,g,epsilon,rho,c,null_vector,precond_vectors):
        self.K=K
        self.a=a
        self.b=b
        self.epsilon=epsilon
        self.x=np.vstack((f,g))
        self.rho=rho
        self.c=c
        self.null_vector =null_vector
        self.precond_vectors=precond_vectors
        self.alpha=[]
        self.err_a=[]
        self.err_b=[] 
        self.objvalues=[]


      def _computegradientf(self,f):
        """Computes Gradient with respect to f"""
        u = np.exp(f/self.epsilon)
        v = np.exp(self.x[self.a.shape[0]:]/self.epsilon)
        return self.a-(u*np.dot(self.K,v)).reshape(f.shape[0],-1)

 
      def _computegradientg(self,g):
        u = np.exp(self.x[:self.a.shape[0]]/self.epsilon)
        v = np.exp(g/self.epsilon)
        return self.b-(v*np.dot(self.K.T,u)).reshape(g.shape[0],-1)

      def _objectivefunction(self,x):
        """Computes the value of the objective function at x"""
        f=x[:self.a.shape[0]]
        g=x[self.a.shape[0]:]
        regularizer=-self.epsilon*np.dot(np.exp(f/self.epsilon).T,np.dot(self.K,np.exp(g/self.epsilon)))
        return np.dot(f.T,self.a)+np.dot(g.T,self.b)+regularizer


      def _wolfe1(self,alpha,p,slope):#Armijo Condition
          """Backtracking""" 
          
          reduction_count = 0
          while True:
            condition = self._objectivefunction(self.x+alpha*p)<self._objectivefunction(self.x)+self.c*alpha*slope
            if condition:
              alpha = self.rho*alpha
              reduction_count += 1
            else:
              break
          return alpha
      
      # First implementation
      # To be made faster using
      # - vector operations only
      # - C code following 
      #   |- https://medium.com/spikelab/calling-c-functions-from-python-104e609f2804
      #   |- https://scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html
      #
      # TODO: Complete refactor.
      def _precond_inversion( self, unnormalized_Hessian, gradient, iterative_inversion=-1, debug=False ):

        # Record list of unwinding transformations on final result
        unwinding_transformations = []

        # Construct modified Hessian
        diag = 1/np.sqrt( np.diag(unnormalized_Hessian).flatten() )
        self.modified_Hessian = diag[:,None]*unnormalized_Hessian*diag[None,:]
        
        # Dummy variable to work on
        matrix = self.modified_Hessian

        # Preconditioning along null vector
        vector = self.null_vector
        vector = vector/diag
        vector = vector/np.linalg.norm(vector)
        vector = vector.reshape( (len(vector), 1) )
        matrix = matrix + np.dot( vector, vector.T)
        # Transformations
        gradient = diag[:,None]*gradient
        unwinding_transformations.append( [diag, lambda d,x : d[:,None]*x])

        # Conditioning with other vectors
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        for i in range(k):
            vector = self.precond_vectors[i]
            value  = np.dot( np.dot( matrix, vector ), vector)
            vector = vector.reshape( (vector.shape[0], 1) )
            P_matrix = np.identity(n) + (1/np.sqrt(value)-1)*np.dot( vector, vector.T)
            # Transforms
            matrix = np.dot( P_matrix, np.dot(matrix, P_matrix) )
            gradient = np.dot( P_matrix, gradient)
            unwinding_transformations.append( [P_matrix, lambda P,x : np.dot(P, x)] )
        # end for

        # Debug
        if debug:
          eig, v = np.linalg.eigh( matrix )
          sorting_indices = np.argsort(eig)
          eig = eig[sorting_indices]
          v   = v[:, sorting_indices]
          #
          print( "List of smallest eigenvalues: ", eig[:5])
          print( "List of largest  eigenvalues: ", eig[-5:])
        
        # Solve
        self.Hessian_stabilized = -matrix/self.epsilon
        if iterative_inversion >= 0:
          accumulator = gradient
          inverse = gradient
          delta   = -(matrix-np.identity(n)) # Unipotent part
          for i in range(iterative_inversion):
            accumulator = np.dot(delta, accumulator)
            inverse = inverse + accumulator
          p_k = self.epsilon*inverse
        else:
          p_k=-np.linalg.solve( self.Hessian_stabilized, gradient)
        

        # Unwind
        for transform in unwinding_transformations:
          P,f = transform
          p_k = f(P,p_k)

        return p_k

      ## Records time
      def _precond_inversion_debug( self, unnormalized_Hessian, gradient, iterative_inversion=-1, debug=False ):

        # Record list of unwinding transformations on final result
        unwinding_transformations = []

        # Construct modified Hessian
        diag = 1/np.sqrt( np.diag(unnormalized_Hessian).flatten() )
        self.modified_Hessian = diag[:,None]*unnormalized_Hessian*diag[None,:]
        
        # Dummy variable to work on
        matrix = self.modified_Hessian

        # Preconditioning along null vector
        vector = self.null_vector
        vector = vector/diag
        vector = vector/np.linalg.norm(vector)
        vector = vector.reshape( (len(vector), 1) )
        matrix = matrix + np.dot( vector, vector.T)
        # Transformations
        gradient = diag[:,None]*gradient
        unwinding_transformations.append( [diag, lambda d,x : d[:,None]*x])

        # Conditioning with other vectors
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        start=time.time()
        for i in range(k):
            vector = self.precond_vectors[i]
            value  = np.dot( np.dot( matrix, vector ), vector)
            vector = vector.reshape( (vector.shape[0], 1) )
            P_matrix = np.identity(n) + (1/np.sqrt(value)-1)*np.dot( vector, vector.T)
            # Transforms
            matrix = np.dot( P_matrix, np.dot(matrix, P_matrix) )
            gradient = np.dot( P_matrix, gradient)
            unwinding_transformations.append( [P_matrix, lambda P,x : np.dot(P, x)] )
        end1=time.time()
        print("\n|--- Time required for preconditioning matrix formation: ", np.round(end1-start,5) ,"s---|")
        # end for

        start2=time.time()
        # Debug
        if debug:
          eig, v = np.linalg.eigh( matrix )
          sorting_indices = np.argsort(eig)
          eig = eig[sorting_indices]
          v   = v[:, sorting_indices]
          #
          print( "List of smallest eigenvalues: ", eig[:5])
          print( "List of largest  eigenvalues: ", eig[-5:])
        end2=time.time()

        print("\n|--- Time taken for the debug step: ", np.round(end2-start2,5),"s---|")

        start3=time.time()
        # Solve
        self.Hessian_stabilized = -matrix/self.epsilon
        if iterative_inversion >= 0:
          accumulator = gradient
          inverse = gradient
          delta   = -(matrix-np.identity(n)) # Unipotent part
          for i in range(iterative_inversion):
            accumulator = np.dot(delta, accumulator)
            inverse = inverse + accumulator
          p_k = self.epsilon*inverse
        else:
          p_k=-np.linalg.solve( self.Hessian_stabilized, gradient)
        end3=time.time()
        print("\n|--- Time taken to evaluate p_k: ",np.round(end3-start3,5),"s---|")

        start4=time.time()
        # Unwind
        for transform in unwinding_transformations:
          P,f = transform
          p_k = f(P,p_k)
        end=time.time()
        print("\n|--- Time taken for unwinding: ",np.round(end-start4,5),"s---|")

        print("\n|--- Time taken for the complete code block: ",np.round(end-start,2),"s---|")
        return p_k




      def _update(self,tol=1e-11, maxiter=100, iterative_inversion=-1, timedebug=False, debug=False):
        
        i=0
        while True :
            grad_f=self._computegradientf(self.x[:self.a.shape[0]])
            grad_g=self._computegradientg(self.x[self.a.shape[0]:])
        
            gradient=np.vstack((grad_f,grad_g))
            
            
            # Compute Hessian
            u = np.exp(self.x[:self.a.shape[0]]/self.epsilon)
            v = np.exp(self.x[self.a.shape[0]:]/self.epsilon)
            r1 = u*np.dot(self.K,v)
            r2 = v*np.dot(self.K.T,u)
            # P  = u*self.K*(v.T) # WRONG AGAIN: DANGEROUS CODE!!
            # P = u[:,None]*self.K*v[None,:]
            P  = u*self.K*(v.T)

            A = np.diag( np.array(r1.reshape(r1.shape[0],)) )
            B = P
            C = P.T
            D = np.diag( np.array(r2.reshape(r2.shape[0],)) )
            result = np.vstack( ( np.hstack((A,B)), np.hstack((C,D)) ) )

            self.Hessian = -result/self.epsilon

            # Inverting Hessian against gradient with preconditioning
            if not timedebug:
              p_k  = self._precond_inversion( result, gradient, iterative_inversion=iterative_inversion, debug=debug )

            else:
              print("\n At iteration: ",i)
              p_k  = self._precond_inversion_debug( result, gradient, iterative_inversion=iterative_inversion, debug=debug)

            # print("Outputs")
            # print( np.linalg.norm(p_k-p_k2))
            # print( np.linalg.norm(p_k))
            # print( np.linalg.norm(p_k2))

            #p_k = p_k2
            
            # try:
            #   p_k = self._precond_inversion( result, gradient )

            # except:
            #   print("Inverse does not exist at epsilon:",self.epsilon)
            #   return np.zeros(6)

           # Stacked
            p_k_stacked = np.vstack((p_k[:self.a.shape[0]],p_k[self.a.shape[0]:]))

            # Wolfe Condition 1: Armijo Condition  
            slope = np.dot( p_k.T, gradient)[0][0]
            alpha = 1
            alpha = self._wolfe1( alpha, p_k_stacked, slope)
            self.alpha.append( alpha )

            # Update x = f and g
            self.x = self.x + alpha*p_k_stacked
          
            # error computation 1
            s = np.exp(self.x[:self.a.shape[0]]/self.epsilon)*np.dot(self.K,np.exp(self.x[self.a.shape[0]:]/self.epsilon))
            self.err_a.append(Lin.norm(s - self.a))

            # error computation 2
            r = np.exp(self.x[self.a.shape[0]:]/self.epsilon)*np.dot(self.K .T, np.exp(self.x[:self.a.shape[0]]/self.epsilon))
            self.err_b.append(Lin.norm(r - self.b))

            # Calculating Objective values
            value = self._objectivefunction(self.x)
            self.objvalues.append(value[0])
            
            if i<maxiter and (self.err_a[-1]>tol or self.err_b[-1]>tol) :
                 i+=1
            else:
                print("Terminating after iteration: ",i+1)
                break
      
        # end for    
        return self.x[:self.a.shape[0]],self.x[self.a.shape[0]:],self.err_a,self.err_b ,self.objvalues,self.alpha
        
#Footer
