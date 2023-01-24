import numpy as np
from numpy import linalg as Lin
import scipy
import logging
import time
from scipy.sparse.linalg import cg,LinearOperator,gmres

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
        self.m=None
        self.t=[]
       


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
      def _precond_inversion_v0( self, unnormalized_Hessian, gradient, iterative_inversion=-1, debug=False ):

        start=time.time()
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
        start1=time.time()
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
        end1=time.time()
        t1=np.round(1e3*(end1-start1),5)
        print("\n|--- Time required for preconditioning matrix formation: ", np.round(1e3*(end1-start1),5) ,"ms---|")
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
        t2=np.round(1e3*(end2-start2),5)

        print("|--- Time taken for the debug step: ", np.round(1e3*(end2-start2),5),"ms---|")

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
        t3=np.round(1e3*(end3-start3),5)
        print("|--- Time taken to invert the linear system for p_k: ",np.round( 1e3*(end3-start3),5),"ms---|")

        start4=time.time()
        # Unwind
        for transform in unwinding_transformations:
          P,f = transform
          p_k = f(P,p_k)
        end=time.time()
        t4=np.round(1e3*(end-start4),5)
        print("|--- Time taken for unwinding: ",np.round( 1e3*(end-start4),5),"ms---|")
        t_all=np.round(1e3*(end-start),5)

        print("|--- Time taken for the complete code block: ",np.round( 1e3*(end-start),2),"ms---|\n")
        return p_k,(t1,t2,t3,t4,t_all)

      def _precond_inversion_v1( self, unnormalized_Hessian, gradient, iterative_inversion=-1, debug=False ):
        start=time.time()
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
        end0=time.time()
        t1=np.round(1e3*(end0-start),5)
        print("\n|--- Time required for initial preconditioning: ", np.round(1e3*(end0-start),5) ,"ms---|")


        # Conditioning with other vectors
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        start1 = time.time()
        P_matrix = np.identity(n)
        for i in range(k):
          vector = self.precond_vectors[i]
          value  = np.dot( np.dot( matrix, vector ), vector)
          vector = vector.reshape( (vector.shape[0], 1) )
          P_matrix = P_matrix+ (1/np.sqrt(value)-1)*np.dot( vector, vector.T)
        # end for
        unwinding_transformations.append( [P_matrix, lambda P,x : np.dot(P, x)] )
        
        matrix = np.dot( P_matrix, np.dot(matrix, P_matrix) )
        gradient = np.dot( P_matrix, gradient)
        end1 = time.time()
        t2 = np.round(1e3*(end1-start1),5)
        print("|--- Time required for preconditioning matrix formation: ", np.round(1e3*(end1-start1),5) ,"ms---|")

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
        t3=np.round(1e3*(end2-start2),5)

        print("|--- Time taken for the debug step: ", np.round(1e3*(end2-start2),5),"ms---|")

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
        t4=np.round(1e3*(end3-start3),5)
        print("|--- Time taken to invert the linear system for p_k: ",np.round( 1e3*(end3-start3),5),"ms---|")

        start4=time.time()
        # Unwind
        for transform in unwinding_transformations:
          P,f = transform
          p_k = f(P,p_k)
        end=time.time()
        t5=np.round(1e3*(end-start4),5)
        print("|--- Time taken for unwinding: ",np.round( 1e3*(end-start4),5),"ms---|")
        t_all=np.round(1e3*(end-start),5)

        print("|--- Time taken for the complete code block: ",np.round( 1e3*(end-start),2),"ms---|\n")
        return p_k,(t1,t2,t3,t4,t5,t_all)

      def _precond_inversion_v2( self, unnormalized_Hessian, gradient, iterative_inversion=-1, debug=False ):
        start=time.time()
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
        end0=time.time()
        t1=np.round(1e3*(end0-start),5)
        print("\n|--- Time required for initial preconditioning: ", np.round(1e3*(end0-start),5) ,"ms---|")


        # Conditioning with other vectors
        #  Naming conventions:
        #  y = Preconditioning vectors as a numpy matrix n by k
        #  matrix = our matrix A to precondition
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        start1=time.time()
        y = np.array( self.precond_vectors ).T # Matrix of size n by k
        # Compute eigenvalues
        Ay = np.dot( matrix, y)
        eigenvalues = np.sum( y * Ay, axis=0 )
        # Compute P_matrix = id + y*diag(values)*y.T
        values = (1/np.sqrt(eigenvalues)-1)    # Vector of size k
        z = y*values[None,:]
        P_matrix = np.identity(n) + np.dot( z, y.T)
        # Old version
        # P_matrix = np.identity(n)
        # for i in range(k):
        #   vector = self.precond_vectors[i]
        #   #value  = np.dot( np.dot( matrix, vector ), vector)
        #   vector = vector.reshape( (vector.shape[0], 1) )
        #   P_matrix = P_matrix+ (1/np.sqrt(eigenvalues[i])-1)*np.dot( vector, vector.T)
        # # end for
        end1=time.time()
        unwinding_transformations.append( [P_matrix, lambda P,x : np.dot(P, x)] )
        t2=np.round(1e3*(end1-start1),5)
        print("|--- Time required for preconditioning matrix formation: ", np.round(1e3*(end1-start1),5) ,"ms---|")

        # Changing A=matrix to PAP
        start2=time.time()
        # Old version -- O(n^3)
        # matrix = np.dot( P_matrix, np.dot(matrix, P_matrix) )
        # New version -- O(n^2)
        # PAP = (id + y*diag(values)*y.T)*A*(id + y*diag(values)*y.T)
        #     = A + A*y*diag(values)*y.T + y*diag(values)*(A*y).T + y*diag(values)*y.T*A*y*diag(values)*y.T
        B = np.dot( Ay, z.T)
        C = z @ np.dot(y.T, Ay) @ z.T
        matrix = matrix + B + B.T + C
        gradient = np.dot( P_matrix, gradient)
        end2=time.time()
        t3= np.round(1e3*(end2-start2),5)
        print("|--- Time required for changing A to PAP: ", np.round(1e3*(end2-start2),5) ,"ms---|")

        start3=time.time()
        # Debug
        if debug:
          eig, v = np.linalg.eigh( matrix )
          sorting_indices = np.argsort(eig)
          eig = eig[sorting_indices]
          v   = v[:, sorting_indices]
          #
          print( "List of smallest eigenvalues: ", eig[:5])
          print( "List of largest  eigenvalues: ", eig[-5:])
        end3=time.time()
        t4=np.round( 1e3*(end3-start3),5)

        print("|--- Time taken for the debug step: ", np.round(1e3*(end3-start3),5),"ms---|")

        start4=time.time()      
        end1=time.time()

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
        end4=time.time()
        t5=np.round( 1e3*(end4-start4),5)
        print("|--- Time taken to invert the linear system for p_k: ",np.round( 1e3*(end4-start4),5),"ms---|")

        start5=time.time()
        # Unwind
        for transform in unwinding_transformations:
          P,f = transform
          p_k = f(P,p_k)
        end=time.time()
        t6=np.round( 1e3*(end-start5),5)

        print("|--- Time taken for unwinding: ",np.round( 1e3*(end-start4),5),"ms---|")
        t_all=np.round( 1e3*(end-start),5)

        print("|--- Time taken for the complete code block: ",np.round( 1e3*(end-start),2),"ms---|\n")
        return p_k,(t1,t2,t3,t4,t5,t6,t_all)

      def _precond_inversion_v3( self, unnormalized_Hessian, gradient, iterative_inversion=-1, debug=False ):
        start=time.time()
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
        end0=time.time()
        t1=np.round(1e3*(end0-start),5)
        print("\n|--- Time required for initial preconditioning: ", np.round(1e3*(end0-start),5) ,"ms---|")


        # Conditioning with other vectors
        #  Naming conventions:
        #  y = Preconditioning vectors as a numpy matrix n by k
        #  matrix = our matrix A to precondition
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        start1=time.time()
        y = np.array( self.precond_vectors ).T # Matrix of size n by k
        # Compute eigenvalues
        Ay = np.dot( matrix, y)
        eigenvalues = np.sum( y * Ay, axis=0 )
        # Compute P_matrix = id + y*diag(values)*y.T
        values = (1/np.sqrt(eigenvalues)-1)    # Vector of size k
        z = y*values[None,:]
        P_matrix = np.identity(n) + np.dot( z, y.T)
        # Done
        end1=time.time()
        unwinding_transformations.append( [P_matrix, lambda P,x : np.dot(P, x)] )
        t2=np.round(1e3*(end1-start1),5)
        print("|--- Time required for preconditioning matrix formation: ", np.round(1e3*(end1-start1),5) ,"ms---|")

        # Changing A=matrix to PAP
        start11=time.time()
        # PAP = (id + y*diag(values)*y.T)*A*(id + y*diag(values)*y.T)
        #     = A + A*y*diag(values)*y.T + y*diag(values)*(A*y).T + y*diag(values)*y.T*A*y*diag(values)*y.T
        B = np.dot( Ay, z.T)
        C = z @ np.dot(y.T, Ay) @ z.T
        matrix = matrix + B + B.T + C
        gradient = np.dot( P_matrix, gradient)
        end11=time.time()
        t3= np.round(1e3*(end11-start11),5)
        print("|--- Time required for changing A to PAP: ", np.round(1e3*(end11-start11),5) ,"ms---|")

        start3=time.time()
        # Solve either iteratively using CG or exactly
        self.Hessian_stabilized = -matrix/self.epsilon
        if iterative_inversion >= 0:
          inverse, exit_code = cg(matrix, gradient, x0=gradient, maxiter=iterative_inversion, tol=1e-10)
          print( "  --- CG exit code: ", exit_code)
          p_k = self.epsilon*inverse
          p_k = p_k.reshape( (p_k.shape[0], 1) ) # For some reason, this outputs (n,) and the next line outputs (n,1)
        else:
          p_k=-np.linalg.solve( self.Hessian_stabilized, gradient)
        end3=time.time()
        t4=np.round( 1e3*(end3-start3),5)
        print("|--- Time taken to invert the linear system for p_k: ",np.round( 1e3*(end3-start3),5),"ms---|")

        start4=time.time()
        # Unwind
        for transform in unwinding_transformations:
          P,f = transform
          p_k = f(P,p_k)
        end=time.time()
        t5=np.round( 1e3*(end-start4),5)
        print("|--- Time taken for unwinding: ",np.round( 1e3*(end-start4),5),"ms---|")
        t_all=np.round( 1e3*(end-start),2)
        print("|--- Time taken for the complete code block: ",np.round( 1e3*(end-start),2),"ms---|\n")
        return p_k,(t1,t2,t3,t4,t5,t_all)

      def _precond_inversion_v3_1( self, unnormalized_Hessian, gradient, iterative_inversion=-1, debug=False ):
        
        def mv(vector):
            return np.dot(self.m,vector) 

        start = time.time()
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
        end0=time.time()
        t1=np.round(1e3*(end0-start),5)
        print("\n|--- Time required for initial preconditioning: ", np.round(1e3*(end0-start),5) ,"ms---|")


        # Conditioning with other vectors
        #  Naming conventions:
        #  y = Preconditioning vectors as a numpy matrix n by k
        #  matrix = our matrix A to precondition
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        start1=time.time()
        y = np.array( self.precond_vectors ).T # Matrix of size n by k
        # Compute eigenvalues
        Ay = np.dot( matrix, y)
        eigenvalues = np.sum( y * Ay, axis=0 )
        # Compute P_matrix = id + y*diag(values)*y.T
        values = (1/np.sqrt(eigenvalues)-1)    # Vector of size k
        z = y*values[None,:]
        P_matrix = np.identity(n) + np.dot( z, y.T)
        # Done
        end1 = time.time()
        unwinding_transformations.append( [P_matrix, lambda P,x : np.dot(P, x)] )
        t2=np.round(1e3*(end1-start1),5)
        print("|--- Time required for preconditioning matrix formation: ", np.round(1e3*(end1-start1),5) ,"ms---|")

        # Changing A=matrix to PAP
        start11 = time.time()
        # PAP = (id + y*diag(values)*y.T)*A*(id + y*diag(values)*y.T)
        #     = A + A*y*diag(values)*y.T + y*diag(values)*(A*y).T + y*diag(values)*y.T*A*y*diag(values)*y.T
        B = np.dot( Ay, z.T)
        C = z @ np.dot(y.T, Ay) @ z.T
        matrix = matrix + B + B.T + C
        gradient = np.dot( P_matrix, gradient)
        end11=time.time()
        t3= np.round(1e3*(end11-start11),5)
        print("|--- Time required for changing A to PAP: ", np.round(1e3*(end11-start11),5) ,"ms---|")

        start3=time.time()
        # Solve either iteratively using CG or exactly
        self.Hessian_stabilized = -matrix/self.epsilon
        if iterative_inversion >= 0:
          self.m = matrix
          A = LinearOperator((self.m.shape[0],self.m.shape[1]), matvec=mv)
          inverse, exit_code = cg(A, gradient, x0=gradient, maxiter=iterative_inversion, tol=1e-10)
          print( "  --- CG exit code: ", exit_code)
          p_k = self.epsilon*inverse
          p_k = p_k.reshape( (p_k.shape[0], 1) ) # For some reason, this outputs (n,) and the next line outputs (n,1)
        else:
          p_k=-np.linalg.solve( self.Hessian_stabilized, gradient)
        end3=time.time()
        t4=np.round( 1e3*(end3-start3),5)
        print("|--- Time taken to invert the linear system for p_k: ",np.round( 1e3*(end3-start3),5),"ms---|")

        start4=time.time()
        # Unwind
        for transform in unwinding_transformations:
          P,f = transform
          p_k = f(P,p_k)
        end=time.time()
        t5=np.round( 1e3*(end-start4),5)
        print("|--- Time taken for unwinding: ",np.round( 1e3*(end-start4),5),"ms---|")
        t_all=np.round( 1e3*(end-start),2)

        print("|--- Time taken for the complete code block: ",np.round( 1e3*(end-start),2),"ms---|\n")
        return p_k,(t1,t2,t3,t4,t5,t_all)

      def _precond_inversion_v3_2( self, unnormalized_Hessian, gradient, iterative_inversion=-1, debug=False ):
        
        def mv(vector):
            return np.dot(self.m,vector) 

        start = time.time()
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
        end0=time.time()
        t1=np.round(1e3*(end0-start),5)
        print("\n|--- Time required for initial preconditioning: ", np.round(1e3*(end0-start),5) ,"ms---|")


        # Conditioning with other vectors
        #  Naming conventions:
        #  y = Preconditioning vectors as a numpy matrix n by k
        #  matrix = our matrix A to precondition
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        start1=time.time()
        y = np.array( self.precond_vectors ).T # Matrix of size n by k
        # Compute eigenvalues
        Ay = np.dot( matrix, y)
        eigenvalues = np.sum( y * Ay, axis=0 )
        # Compute P_matrix = id + y*diag(values)*y.T
        values = (1/np.sqrt(eigenvalues)-1)    # Vector of size k
        z = y*values[None,:]
        P_matrix = np.identity(n) + np.dot( z, y.T)
        # Done
        end1 = time.time()
        unwinding_transformations.append( [P_matrix, lambda P,x : np.dot(P, x)] )
        t2=np.round(1e3*(end1-start1),5)
        print("|--- Time required for preconditioning matrix formation: ", np.round(1e3*(end1-start1),5) ,"ms---|")

        # Changing A=matrix to PAP
        start11 = time.time()
        # PAP = (id + y*diag(values)*y.T)*A*(id + y*diag(values)*y.T)
        #     = A + A*y*diag(values)*y.T + y*diag(values)*(A*y).T + y*diag(values)*y.T*A*y*diag(values)*y.T
        B = np.dot( Ay, z.T)
        C = z @ np.dot(y.T, Ay) @ z.T
        matrix = matrix + B + B.T + C
        # gradient = np.dot( P_matrix, gradient)
        gradient = np.dot( P_matrix, gradient)
        end11=time.time()
        t3= np.round(1e3*(end11-start11),5)
        print("|--- Time required for changing A to PAP: ", np.round(1e3*(end11-start11),5) ,"ms---|")

        start3=time.time()
        # Solve either iteratively using CG or exactly
        self.Hessian_stabilized = -matrix/self.epsilon
        if iterative_inversion >= 0:
          self.m = matrix
          A = LinearOperator((self.m.shape[0],self.m.shape[1]), matvec=mv)
          inverse, exit_code = gmres(A, gradient, x0=gradient, maxiter=iterative_inversion, tol=1e-10)
          print( "  --- CG exit code: ", exit_code)
          p_k = self.epsilon*inverse
          p_k = p_k.reshape( (p_k.shape[0], 1) ) # For some reason, this outputs (n,) and the next line outputs (n,1)
        else:
          p_k=-np.linalg.solve( self.Hessian_stabilized, gradient)
        end3=time.time()
        t4=np.round( 1e3*(end3-start3),5)
        print("|--- Time taken to invert the linear system for p_k: ",np.round( 1e3*(end3-start3),5),"ms---|")

        start4=time.time()
        # Unwind
        for transform in unwinding_transformations:
          P,f = transform
          p_k = f(P,p_k)
        end=time.time()
        t5=np.round( 1e3*(end-start4),5)
        print("|--- Time taken for unwinding: ",np.round( 1e3*(end-start4),5),"ms---|")
        t_all=np.round( 1e3*(end-start),2)

        print("|--- Time taken for the complete code block: ",np.round( 1e3*(end-start),2),"ms---|\n")
        return p_k,(t1,t2,t3,t4,t5,t_all)

      def _precond_inversion_v4( self, unnormalized_Hessian, gradient, iterative_inversion=-1, debug=False ):
        start=time.time()
        # Record list of unwinding transformations on final result
        unwinding_transformations = []

        
        # Construct modified Hessian
        diag = 1/np.sqrt( np.diag(unnormalized_Hessian).flatten() )
        self.modified_Hessian = diag[:,None]*unnormalized_Hessian*diag[None,:]
        # Transformations
        gradient = diag[:,None]*gradient
        unwinding_transformations.append( [diag, lambda d,x : d[:,None]*x])
       
        # Dummy variable to work on
        matrix = self.modified_Hessian

        # Preconditioning along null vector
        vector = self.null_vector
        vector = vector/diag
        vector = vector/np.linalg.norm(vector)
        vector = vector.reshape( (len(vector), 1) )
        matrix = matrix + np.dot( vector, vector.T)

        # Conditioning with other vectors
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        self.y = np.array( self.precond_vectors ).T # Matrix of size n by k
        # Compute eigenvalues
        Ay = np.dot( matrix, self.y)
        eigenvalues = np.sum( self.y * Ay, axis=0 )

        end0 = time.time()
        t1=np.round(1e3*(end0-start),5)
        print("\n|--- Time required for all preconditioning: ", np.round(1e3*(end0-start),5) ,"ms---|")

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
        t2=np.round(1e3*(end2-start2),5)

        print("|--- Time taken for the debug step: ", np.round(1e3*(end2-start2),5),"ms---|")
        # precond_map=lambda x,y,z: x-
        start3=time.time()
        # Solve
        self.Hessian_stabilized = -matrix/self.epsilon
        if iterative_inversion >= 0:
          accumulator = gradient
          inverse = gradient
          self.eigvalues = 1-(1/np.sqrt(eigenvalues))#Correct
          self.eigvalues = self.eigvalues.reshape(self.eigvalues.shape[0],)
          self.eigscalar_vec = self.eigvalues[None,:]*self.y
          self.m = matrix

          x = self.preconditioned_map(inverse)
          print(x.shape)
          # for i in range(iterative_inversion):           
          #   z=self._iterative_precondition(x,accumulator)
          #   accumu=np.dot(matrix,accumulator)
          #   accumulator=accumulator-self._iterative_precondition(x,accumulator)
          #   inverse=self._iterative_precondition(x,inverse)
          #   inverse = inverse + accumulator
          # p_k = self.epsilon*inverse
        else:
          p_k=-np.linalg.solve( self.Hessian_stabilized, gradient)
        end3=time.time()
        t3=np.round(1e3*(end3-start3),5)
        print("|--- Time taken to invert the linear system for p_k: ",np.round( 1e3*(end3-start3),5),"ms---|")

        start4=time.time()
        # Unwind
        for transform in unwinding_transformations:
          P,f = transform
          p_k = f(P,p_k)
        end=time.time()
        t4=np.round(1e3*(end-start4),5)
        print("|--- Time taken for unwinding: ",np.round( 1e3*(end-start4),5),"ms---|")
        t_all=np.round(1e3*(end-start),5)

        print("|--- Time taken for the complete code block: ",np.round( 1e3*(end-start),2),"ms---|\n")
        return p_k,(t1,t2,t3,t4,t_all)

      # Function mapping v to P(A+E)Pv
      def _preconditioned_map(self, vector):
        vector = self._apply_P( vector ) 
        vector = np.dot( self.m, vector)  
        vector = self._apply_P( vector ) 
        return vector
      
      def _apply_P(self, vector):

        vector = vector - (self.eigscalar_vec*np.dot(self.y ,vector)).sum(axis=1)
        vector = vector*self.diag_part
        return vector    


      def _iterative_precondition(self,x,y):
        eig_vec_scalar=np.dot(np.vstack(self.precond_vectors),y)
        eig_vec_scalar=eig_vec_scalar.reshape(eig_vec_scalar.shape[0],)
        z=x*eig_vec_scalar[None,:]
        z=np.sum(z,axis=1)[:,None]
        y=y-z
        return y

      def _update(self,tol=1e-11, maxiter=100, iterative_inversion=-1, version=1, debug=False):
        
        i = 0
        t_1,t_2,t_3,t_4,t_5,t_6,tall = [],[],[],[],[],[],[]
        while True :
            grad_f = self._computegradientf(self.x[:self.a.shape[0]])
            grad_g = self._computegradientg(self.x[self.a.shape[0]:])
        
            gradient=np.vstack((grad_f,grad_g))
            
            
            # Compute Hessian
            u = np.exp(self.x[:self.a.shape[0]]/self.epsilon)
            v = np.exp(self.x[self.a.shape[0]:]/self.epsilon)
            r1 = u*np.dot(self.K,v)
            r2 = v*np.dot(self.K.T,u)
            # P  = u*self.K*(v.T) # WRONG AGAIN: DANGEROUS CODE!!
            u = u.reshape(u.shape[0],)
            v = v.reshape(v.shape[0],)
            P = u[:,None]*self.K*v[None,:]
            A = np.diag( np.array(r1.reshape(r1.shape[0],)) )
            B = P
            C = P.T
            D = np.diag( np.array(r2.reshape(r2.shape[0],)) )
            result = np.vstack( ( np.hstack((A,B)), np.hstack((C,D)) ) )

            self.Hessian = -result/self.epsilon
            # Inverting Hessian against gradient with preconditioning
            if version==4:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v4( result, gradient, iterative_inversion=iterative_inversion, debug=debug )
              t_1.append(temp[0]),t_2.append(temp[1]),t_3.append(temp[2]),t_4.append(temp[3]),tall.append(temp[-1])

            elif version==3.1:
              print("\n At iteration: ",i)
              p_k,temp = self._precond_inversion_v3_1( result, gradient, iterative_inversion=iterative_inversion, debug=debug )
              t_1.append(temp[0]),t_2.append(temp[1]),t_3.append(temp[2]),t_4.append(temp[3]),t_5.append(temp[4]),tall.append(temp[-1])

            elif version==3:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v3( result, gradient, iterative_inversion=iterative_inversion, debug=debug )
              t_1.append(temp[0]),t_2.append(temp[1]),t_3.append(temp[2]),t_4.append(temp[3]),t_5.append(temp[4]),tall.append(temp[-1])
            elif version == 2:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v2( result, gradient, iterative_inversion=iterative_inversion, debug=debug )
              t_1.append(temp[0]),t_2.append(temp[1]),t_3.append(temp[2]),t_4.append(temp[3]),t_5.append(temp[4]),t_6.append(temp[5]),tall.append(temp[-1])
            elif version == 1:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v1( result, gradient, iterative_inversion=iterative_inversion, debug=debug )
              t_1.append(temp[0]),t_2.append(temp[1]),t_3.append(temp[2]),t_4.append(temp[3]),t_5.append(temp[4]),t_6.append(temp[5]),tall.append(temp[-1])
            elif version == 0:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v0( result, gradient, iterative_inversion=iterative_inversion, debug=debug)
              t_1.append(temp[0]),t_2.append(temp[1]),t_3.append(temp[2]),t_4.append(temp[3]),t_5.append(temp[4]),tall.append(temp[-1])

            else:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v2( result, gradient, iterative_inversion=iterative_inversion, debug=debug)
              t_1.append(temp[0]),t_2.append(temp[1]),t_3.append(temp[2]),t_4.append(temp[3]),t_5.append(temp[4]),t_6.append(temp[5]),tall.append(temp[-1])


            
            end=time.time()
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

        self.t=[t_1,t_2,t_3,t_4,t_5,t_6,tall]
        return self.x[:self.a.shape[0]],self.x[self.a.shape[0]:],self.err_a,self.err_b ,self.objvalues,self.alpha,self.t

 # end for    
