import numpy as np
import scipy
import time
from typing import List, Dict, Tuple

class DampedNewton_with_precodonditioner_SemiDual_np:
    def __init__(self,  C : List[List[float]],
                        a : List[float], 
                        b : List[float], 
                        f : List[float],
                        epsilon : float, 
                        rho : float, 
                        c : float, 
                        null_vector : List[float],
                        precond_vectors : List[float]):
        """
        
        Args:
          C : Cost matrix of size n by m.
          ( a, b ) : The two measures of the OT problem, the shape of which is (n,) and (m,) respectively.
          f: Kantorovich potential f, which is of shape (n,).
          rho : Damping factor for the line search update step.
          epsilon : The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
          c : Damping factor for the slope in the Armijo's condition.
          null_vector : null vector of the Hessian obtained from the unpreconditioned iteration of semi-dual damped Newton.
          precond_vector : preconditioning vectors from the selected eigenvalues obtained from the unpreconditioned iteration of semi-dual damped Newton to be used for preconditioning the system <Hessian,p> = gradient, where p is the optimization direction vector.
        """
        self.C = C
        self.a = a
        self.b = b
        self.f = f
        self.epsilon = epsilon
        self.rho = rho           
        self.c = c
        self.null_vector = null_vector
        self.precond_vectors = precond_vectors
        self.alpha_list = []
        self.err = []
        self.objvalues = [] 
        self.timing = []
        self.out = []
        # Computing minimum of  C-f for each column of this difference matrix.
        self.f_C = np.min( self.C - self.f[:,None], axis = 0 )# The C-transform of f, shape : (m,).
        # We know e^((-(C-f)+self.min_f)/epsilon)<1, therefore the value of self.g below is bounded.
        self.H = self.C - self.f[:,None] - self.f_C[None,:] # Shape : (n,m)     
        self.g = self.f_C -self.epsilon*np.log( np.sum( self.a[:,None]*np.exp( -self.H /self.epsilon ), axis = 0 ) )# Shape : (m,)
        self.z = self.C - self.f[:,None] - self.g[None,:]# Shape : (n,m)

    def _objectivefunction( self, f : List[float] )-> float :
        """ 
        
        Args:
          f : The Kantorovich potential f.
        Returns : Q_semi(f) =  <f,a> + <g(f,C,epsilon),b>.
        """
        # Computing minimum of  C-f for each column of this difference matrix.
        f_C = np.min( self.C - f[:,None], axis = 0 )# The C-transform of f, shape : (m,).
        H = self.C - f[:,None] - f_C[None,:]# Shape : (n,m)
        g = f_C - self.epsilon*np.log( np.sum( self.a[:,None]*np.exp( -H /self.epsilon ), axis = 0 ) ) # Shape : (m,)
        Q_semi = np.dot( f, self.a ) + np.dot( g, self.b ) 
        return Q_semi
      
    def _computegradientf( self )->List[float]:
        """ 
            Compute gradient with respect to f of the objective function Q_semi(.).
        """
        gradient = self.a*(np.ones(self.a.shape[0]) - np.sum( np.exp( -self.z/self.epsilon )*self.b[None,:], axis = 1 ))# Shape : (n,)
        return gradient

    def _wolfe1( self,  alpha : float, 
                        p : List[float], 
                        slope :float 
                        )->float:
        #Armijo Condition
        """

        Args:
            alpha : The update step size.
            p :The optimal direction.
            slope : It is the inner product of the gradient and p.

        Returns:
            alpha : The updated step size.
        """
        reduction_count = 0           
        while True:   
            condition = self._objectivefunction( self.f + alpha*p )< self._objectivefunction( self.f ) + self.c*alpha*slope
            if condition or np.isnan(self._objectivefunction( self.f + alpha*p )):
                alpha = self.rho*alpha                                                     
                reduction_count += 1
            else:
                break
        return alpha
        
          
    def _precond_inversion_v0( self,  unnormalized_Hessian : List[List[float]], 
                                      gradient : List[float],
                                      iterative_inversion : int = -1, 
                                      debug : bool = False 
                                      )-> Tuple[ List[float], List[float] ]:
        """

          Args:
              unnormalized_Hessian : THe unnormalized Hessian, shape : (n,n).
              gradient : The gradient vector, shape : (n,).
              iterative_inversion : The number of iterative inversions. Defaults to -1.
              debug : To add a debug any step of the implementation when needed. Defaults to False.
              optType :  Input for the choice of iterative inversion algorithm, which here are Conjugate Gradient-'cg' and GMRES-'gmres. Defaults to 'cg'.

          Returns:
              p : The optimal direction vector.
              timings : The list of timestamps at each step.
        """
      
        timings = []  
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
        vector = vector/np.linalg.norm( vector )
        vector = vector.reshape( (len(vector), 1) )
        matrix = matrix + np.dot( vector, vector.T )
        # Transformations
        gradient = diag[:,None]*gradient[:,None]
        unwinding_transformations.append( [diag, lambda d,x : d[:,None]*x] )

        # Conditioning with other vectors
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        start1=time.time()
        for i in range( k ):
            vector = self.precond_vectors[i]
            value  = np.dot( np.dot( matrix, vector ), vector)
            vector = vector.reshape( (vector.shape[0], 1) ) 
            P_matrix = np.identity( n ) + ( 1/np.sqrt(value)-1 )*np.dot( vector, vector.T )
            # Transforms
            matrix = np.dot( P_matrix, np.dot( matrix, P_matrix ) )
            gradient = np.dot( P_matrix, gradient)
            unwinding_transformations.append( [P_matrix, lambda P,x : np.dot( P, x )] )
        # end for
        end = time.time()
        interval = 1e3*( end-start1 )
        timings.append( interval )
        print( "\n|--- Time required for preconditioning matrix formation: ", np.round( interval,5 ) ,"ms---|" )
        # end for

        start2 = time.time()
        # Debug
        if debug:
          eig, v = np.linalg.eigh( matrix )
          sorting_indices = np.argsort( eig )
          eig = eig[sorting_indices]
          v   = v[:, sorting_indices]
          #
          print( "List of smallest eigenvalues: ", eig[:5])
          print( "List of largest  eigenvalues: ", eig[-5:])
        end = time.time()
        interval = 1e3*( end-start2 )
        timings.append( interval )

        print( "|--- Time taken for the debug step: ", np.round( interval,5 ),"ms---|" )

        start3 = time.time()
        # Solve
        self.Hessian_stabilized = -matrix/self.epsilon
        if iterative_inversion >= 0:
          accumulator = gradient
          inverse = gradient
          delta   = -( matrix-np.identity( n ) ) # Unipotent part
          for i in range( iterative_inversion ):
            accumulator = np.dot( delta, accumulator )
            inverse = inverse + accumulator
          p_k = self.epsilon*inverse
        else:
          p_k = -np.linalg.solve( self.Hessian_stabilized, gradient )
        end = time.time()
        interval = 1e3*( end-start3 )
        timings.append( interval )
        print( "|--- Time taken to invert the linear system for p_k: ",np.round( interval,5 ),"ms---|" )

        start4 = time.time()
        # Unwind
        for transform in unwinding_transformations:
          P,f = transform
          p_k = f(P,p_k)    
        end = time.time()
        interval = 1e3*( end-start4 )
        timings.append( interval )
        print( "|--- Time taken for unwinding: ",np.round( interval,5),"ms---|" )
        interval = 1e3*( end-start )
        timings.append( interval )    

        print( "|--- Time taken for the complete code block: ",np.round( interval,2),"ms---|\n" )
        return p_k,timings

    def _precond_inversion_v1( self,  unnormalized_Hessian : List[List[float]], 
                                      gradient : List[float],
                                      iterative_inversion : int = -1, 
                                      debug : bool = False 
                                      )-> Tuple[ List[float], List[float] ]:
        """

          Args:
              unnormalized_Hessian : THe unnormalized Hessian, shape : (n,n).
              gradient : The gradient vector, shape : (n,).
              iterative_inversion : The number of iterative inversions. Defaults to -1.
              debug : To add a debug any step of the implementation when needed. Defaults to False.
              optType :  Input for the choice of iterative inversion algorithm, which here are Conjugate Gradient-'cg' and GMRES-'gmres. Defaults to 'cg'.

          Returns:
              p : The optimal direction vector.
              timings : The list of timestamps at each step.
        """
      
      
        timings = []
        start = time.time()
        # Record list of unwinding transformations on final result
        unwinding_transformations = []

        # Construct modified Hessian
        diag = 1/np.sqrt( np.diag( unnormalized_Hessian ).flatten() )
        self.modified_Hessian = diag[:,None]*unnormalized_Hessian*diag[None,:]
        
        # Dummy variable to work on
        matrix = self.modified_Hessian

        # Preconditioning along null vector
        vector = self.null_vector
        vector = vector/diag
        vector = vector/np.linalg.norm( vector )
        vector = vector.reshape( ( len( vector ), 1) )
        matrix = matrix + np.dot( vector, vector.T )
        # Transformations
        gradient = diag[:,None]*gradient[:,None]
        unwinding_transformations.append( [diag, lambda d,x : d[:,None]*x] )
        end = time.time()
        interval = 1e3*( end-start )
        timings.append( interval )
        print( "\n|--- Time required for initial preconditioning: ", np.round( interval,5 ) ,"ms---|" )


        # Conditioning with other vectors
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        start1 = time.time()
        P_matrix = np.identity(n)
        for i in range(k):
          vector = self.precond_vectors[i]
          value  = np.dot( np.dot( matrix, vector ), vector )
          vector = vector.reshape( (vector.shape[0], 1) )
          P_matrix = P_matrix + ( 1/np.sqrt( value )-1 )*np.dot( vector, vector.T )
        # end for
        unwinding_transformations.append( [P_matrix, lambda P,x : np.dot(P, x)] )
        
        matrix = np.dot( P_matrix, np.dot(matrix, P_matrix) )
        gradient = np.dot( P_matrix, gradient)
        end = time.time()
        interval = 1e3*( end-start1 )
        timings.append( interval ) 
        print( "|--- Time required for preconditioning matrix formation: ", np.round( interval,5 ) ,"ms---|" )

        start2 = time.time()
        # Debug
        if debug:
          eig, v = np.linalg.eigh( matrix )
          sorting_indices = np.argsort(eig)
          eig = eig[sorting_indices]
          v   = v[:, sorting_indices]
          print( "List of smallest eigenvalues: ", eig[:5])
          print( "List of largest  eigenvalues: ", eig[-5:])
        end = time.time()
        interval = 1e3*(end-start2)
        timings.append( interval )


        print( "|--- Time taken for the debug step: ", np.round( interval,5 ),"ms---|" )

        start3 = time.time()
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
          p_k = -np.linalg.solve( self.Hessian_stabilized, gradient)
        end = time.time()
        interval = 1e3*(end-start3)
        timings.append(interval)
        print( "|--- Time taken to invert the linear system for p_k: ",np.round( interval,5 ),"ms---|")

        start4 = time.time()
         # Unwind
        for transform in unwinding_transformations:
          P,f = transform
          p_k = f(P,p_k)
        end = time.time()
        interval = 1e3*(end-start4)
        timings.append(interval)
        print( "|--- Time taken for unwinding: ",np.round( interval,5 ),"ms---|" )
        
        interval = 1e3*(end-start)
        timings.append( interval )
        print( "|--- Time taken for the complete code block: ",np.round( interval,2 ),"ms---|\n" )
        return p_k,timings

    def _precond_inversion_v2( self,  unnormalized_Hessian : List[List[float]], 
                                      gradient : List[float], 
                                      iterative_inversion : int = -1, 
                                      debug : bool = False 
                                      )-> Tuple[ List[float], List[float] ]:
        """

          Args:
              unnormalized_Hessian : THe unnormalized Hessian, shape : (n,n).
              gradient : The gradient vector, shape : (n,).
              iterative_inversion : The number of iterative inversions. Defaults to -1.
              debug : To add a debug any step of the implementation when needed. Defaults to False.
              optType :  Input for the choice of iterative inversion algorithm, which here are Conjugate Gradient-'cg' and GMRES-'gmres. Defaults to 'cg'.

          Returns:
              p : The optimal direction vector.
              timings : The list of timestamps at each step.
        """
      
      
        timings = []
        start = time.time()
        # Record list of unwinding transformations on final result
        unwinding_transformations = []

        # Construct modified Hessian
        diag = 1/np.sqrt( np.diag( unnormalized_Hessian ).flatten() )
        self.modified_Hessian = diag[:,None]*unnormalized_Hessian*diag[None,:]
        
        # Dummy variable to work on
        matrix = self.modified_Hessian

        # Preconditioning along null vector
        vector = self.null_vector
        vector = vector/diag
        vector = vector/np.linalg.norm( vector )
        vector = vector.reshape( (len( vector ), 1) )
        matrix = matrix + np.dot( vector, vector.T )
        # Transformations
        gradient = diag[:,None]*gradient[:,None]
        unwinding_transformations.append( [diag, lambda d,x : d[:,None]*x] )
        end = time.time()
        interval = 1e3*( end-start )
        timings.append( interval )
        print( "\n|--- Time required for initial preconditioning: ", np.round( interval,5 ) ,"ms---|" )


        # Conditioning with other vectors
        #  Naming conventions:
        #  y = Preconditioning vectors as a numpy matrix n by k
        #  matrix = our matrix A to precondition
        k = len( self.precond_vectors )
        n = self.null_vector.shape[0]
        start1=time.time()
        y = np.array( self.precond_vectors ).T # Matrix of size n by k
        # Compute eigenvalues
        Ay = np.dot( matrix, y )
        eigenvalues = np.sum( y * Ay, axis=0 )
        # Compute P_matrix = id + y*diag(values)*y.T
        values = ( (1/np.sqrt(eigenvalues))-1 )    # Vector of size k
        z = y*values[None,:]
        P_matrix = np.identity( n ) + np.dot( z, y.T )
        # Old version
        # P_matrix = np.identity(n)
        # for i in range(k):
        #   vector = self.precond_vectors[i]
        #   #value  = np.dot( np.dot( matrix, vector ), vector)
        #   vector = vector.reshape( (vector.shape[0], 1) )
        #   P_matrix = P_matrix+ (1/np.sqrt(eigenvalues[i])-1)*np.dot( vector, vector.T)
        # # end for
        end = time.time()
        unwinding_transformations.append( [P_matrix, lambda P,x : np.dot(P, x)] )
        interval = 1e3*( end-start1 )
        timings.append( interval )
        print( "|--- Time required for preconditioning matrix formation: ", np.round( interval,5 ) ,"ms---|" )

        # Changing A=matrix to PAP
        start2 = time.time()
        # Old version -- O(n^3)
        # matrix = np.dot( P_matrix, np.dot(matrix, P_matrix) )
        # New version -- O(n^2)
        # PAP = (id + y*diag(values)*y.T)*A*(id + y*diag(values)*y.T)
        #     = A + A*y*diag(values)*y.T + y*diag(values)*(A*y).T + y*diag(values)*y.T*A*y*diag(values)*y.T
        B = np.dot( Ay, z.T )
        C = z @ np.dot( y.T, Ay ) @ z.T
        matrix = matrix + B + B.T + C
        gradient = np.dot( P_matrix, gradient )
        end = time.time()
        interval = 1e3*( end-start2 )
        timings.append( interval )
        print( "|--- Time required for changing A to PAP: ", np.round( interval,5 ) ,"ms---|" )

        start3 = time.time()
        # Debug
        if debug:
          eig, v = np.linalg.eigh( matrix )
          sorting_indices = np.argsort( eig )
          eig = eig[sorting_indices]
          v   = v[:, sorting_indices]
          #
          print( "List of smallest eigenvalues: ", eig[:5] )
          print( "List of largest  eigenvalues: ", eig[-5:] )
        end = time.time()
        interval = 1e3*( end-start3 )
        timings.append( interval )

        print( "|--- Time taken for the debug step: ", np.round( interval,5 ),"ms---|" )

        start4 = time.time()      

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
          p_k = -np.linalg.solve( self.Hessian_stabilized, gradient )
        end = time.time()
        interval = 1e3*( end-start4 )
        timings.append( interval )
        print( "|--- Time taken to invert the linear system for p_k: ",np.round( interval,5 ),"ms---|" )

        start5 = time.time()
        # Unwind
        for transform in unwinding_transformations:
          P,f = transform
          p_k = f(P,p_k)
        end = time.time()
        interval = 1e3*( end-start5 )
        timings.append( interval )

        print( "|--- Time taken for unwinding: ",np.round( interval,5 ),"ms---|" )
        interval = 1e3*( end - start )
        timings.append( interval )

        print( "|--- Time taken for the complete code block: ",np.round( interval ,2),"ms---|\n" ) 
        return p_k,timings

    def _precond_inversion_v3( self,  unnormalized_Hessian : List[List[float]], 
                                      gradient : List[float],
                                      iterative_inversion : int = -1, 
                                      optType : str = None 
                                      )-> Tuple[ List[float], List[float] ]:
        """

          Args:
              unnormalized_Hessian : THe unnormalized Hessian, shape : (n,n).
              gradient : The gradient vector, shape : (n,).
              iterative_inversion : The number of iterative inversions. Defaults to -1.
              debug : To add a debug any step of the implementation when needed. Defaults to False.
              optType :  Input for the choice of iterative inversion algorithm, which here are Conjugate Gradient-'cg' and GMRES-'gmres. Defaults to 'cg'.

          Returns:
              p : The optimal direction vector.
              timings : The list of timestamps at each step.
        """
      
        timings = []
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
        gradient = diag[:,None]*gradient[:,None]
        unwinding_transformations.append( [diag, lambda d,x : d[:,None]*x])
        # Record timing 
        end = time.time()
        interval = 1e3*(end-start)
        timings.append( interval )
        print("\n|--- Time required for initial preconditioning: ", np.round(interval,5) ,"ms---|")
        # Conditioning with other vectors
        #  Naming conventions:
        #  y = Preconditioning vectors as a numpy matrix n by k
        #  matrix = our matrix A to precondition
        n = self.null_vector.shape[0]
        start0 = time.time()
        y = np.array( self.precond_vectors ).T # Matrix of size n by k
        # Compute eigenvalues
        Ay = np.dot( matrix, y)
        eigenvalues = np.sum( y * Ay, axis=0 )
        # Compute P_matrix = id + y*diag(values)*y.T
        values = (1/np.sqrt(eigenvalues)-1)    # Vector of size k
        z = y*values[None,:]
        P_matrix = np.identity(n) + np.dot( z, y.T)
        # Done
        unwinding_transformations.append( [P_matrix, lambda P,x : np.dot(P, x)] )
        # Record timings
        end = time.time()
        interval = 1e3*(end-start0)
        timings.append( interval )
        print( "|--- Time required for preconditioning matrix formation: ", np.round(interval,5) ,"ms---|" )
        # Changing A=matrix to PAP
        start2 = time.time()
        # PAP = (id + y*diag(values)*y.T)*A*(id + y*diag(values)*y.T)
        #     = A + A*y*diag(values)*y.T + y*diag(values)*(A*y).T + y*diag(values)*y.T*A*y*diag(values)*y.T
        B = np.dot( Ay, z.T)
        C = z @ np.dot(y.T, Ay) @ z.T
        matrix = matrix + B + B.T + C
        gradient = np.dot( P_matrix, gradient)
        end = time.time()
        interval = 1e3*(end-start2)
        timings.append( interval )
        print( "|--- Time required for changing A to PAP: ", np.round( interval,5 ) ,"ms---|" )
        # Solve either iteratively using CG or exactly
        def mv(vector):
            return np.dot(matrix,vector) 
        start3 = time.time()
        if iterative_inversion >= 0:
          self.m = matrix
          A = scipy.sparse.linalg.LinearOperator( ( self.m.shape[0],self.m.shape[1] ), matvec = mv ) 
          if optType == 'cg':
            inverse, exit_code = scipy.sparse.linalg.cg( A, gradient, x0 = gradient, maxiter = iterative_inversion, tol = 1e-10 )
            print( "  --- CG exit code: ", exit_code)
          else:
            inverse, exit_code = scipy.sparse.linalg.gmres( A, gradient, x0 = gradient, maxiter =   iterative_inversion, tol = 1e-10 )
            print( "  --- GMRES exit code: ", exit_code)
          p_k = self.epsilon*inverse
          p_k = p_k.reshape( (p_k.shape[0], 1) ) # For some reason, this outputs (n,) and the next line outputs (n,1)
        else:
          self.Hessian_stabilized = -matrix/self.epsilon
          p_k = -np.linalg.solve( self.Hessian_stabilized, gradient )
        end = time.time()
        interval = 1e3*(end-start3)
        timings.append(interval)
        print( "|--- Time taken to invert the linear system for p_k: ",np.round( interval,5),"ms---|" )
        start4 = time.time()
        # Unwind
        for transform in unwinding_transformations:
          P,f = transform
          p_k = f(P,p_k)
        end = time.time()
        interval = 1e3*(end-start4)
        timings.append( interval )
        print("|--- Time taken for unwinding: ",np.round( interval ,5 ) ,"ms---|")
        interval = 1e3*(end-start)
        timings.append( interval )
        print("|--- Time taken for the complete code block: ",np.round( interval,2),"ms---|\n")
        return p_k, timings
      
    def _precond_inversion_v4( self,  unnormalized_Hessian : List[List[float]], 
                                      gradient : List[float],
                                      iterative_inversion : int = -1, 
                                      optType : str = None 
                                      )-> Tuple[ List[float], List[float] ]:
        """

          Args:
              unnormalized_Hessian : THe unnormalized Hessian, shape : (n,n).
              gradient : The gradient vector, shape : (n,).
              iterative_inversion : The number of iterative inversions. Defaults to -1.
              debug : To add a debug any step of the implementation when needed. Defaults to False.
              optType :  Input for the choice of iterative inversion algorithm, which here are Conjugate Gradient-'cg' and GMRES-'gmres. Defaults to 'cg'.

          Returns:
              p : The optimal direction vector.
              timings : The list of timestamps at each step.
        """
        timings = []
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
        vector_E = vector
        # Transformations (Initial on gradient and final on result)
        gradient = diag[:,None]*gradient[:,None]
        unwinding_transformations.append( lambda x : diag[:,None]*x )
        # Record timing
        end = time.time()
        interval = 1e3*(end-start)
        timings.append( interval )
        print("\n|--- Time required for initial preconditioning: ", np.round(interval,5) ,"ms---|")
        # Conditioning with other vectors
        #  Naming conventions:
        #  y = Preconditioning vectors as a numpy matrix n by k
        #  matrix = our matrix A to precondition
        #  We only form the data y and z such that
        #  P = id + z*y.T
        start0 = time.time()
        y = np.array( self.precond_vectors ).T # Matrix of size n by k
        # Compute eigenvalues
        Ay = np.dot( matrix, y)
        eigenvalues = np.sum( y * Ay, axis=0 )
        # Compute data for P = id + y*diag(values)*y.T
        values = (1/np.sqrt(eigenvalues)-1)    # Vector of size k
        z = y*values[None,:]
        # Record timings
        end = time.time()
        interval = 1e3*(end-start0)
        timings.append( interval )
        print( "|--- Time required for preconditioning matrix formation: ", np.round(interval,5) ,"ms---|" )

        # Changing A=matrix to PAP
        start2 = time.time()

        # Function mapping v to Pv
        # P = Id + z*y.T
        def _apply_P(vector):
          return vector + z @ ( y.T @ vector)
        
        # Function mapping v to P(A+E)Pv
        # A is matrix
        # E is vector_E*vector_E.T
        def _preconditioned_map(vector):
          vector = _apply_P( vector ) 
          vector = np.dot( matrix, vector)  + vector_E*np.dot(vector_E, vector)
          vector = _apply_P( vector ) 
          return vector
        # Apply P
        # At beginning on gradient
        # At the end 
        gradient = _apply_P(gradient) 
        unwinding_transformations.append( lambda x : _apply_P(x) )
        end=time.time()
        interval = 1e3*(end-start2)
        timings.append( interval )
        print( "|--- Time required for changing A to PAP: ", np.round( interval,5 ) ,"ms---|" )
        #
        # Solve either iteratively using CG or exactly
        start3 = time.time()
        if iterative_inversion >= 0:
          self.m = matrix
          A = scipy.sparse.linalg.LinearOperator( ( self.m.shape[0],self.m.shape[1] ), matvec=_preconditioned_map ) 
          if optType == 'cg':
            inverse, exit_code = scipy.sparse.linalg.cg( A, gradient, x0=gradient, maxiter=iterative_inversion, tol=1e-10 )
            print( "  --- CG exit code: ", exit_code)
          else: 
            inverse, exit_code = scipy.sparse.linalg.gmres( A, gradient, x0=gradient, maxiter=iterative_inversion, tol=1e-10 )
            print( "  --- GMRES exit code: ", exit_code)
          p_k = self.epsilon*inverse
          p_k = p_k.reshape( (p_k.shape[0], 1) ) # For some reason, this outputs (n,) and the next line outputs (n,1)
        else:
          B = np.dot( Ay, z.T)
          C = z @ np.dot(y.T, Ay) @ z.T
          matrix = matrix + B + B.T + C
          self.Hessian_stabilized = -matrix/self.epsilon
          p_k = -np.linalg.solve( self.Hessian_stabilized, gradient)        
        end = time.time()
        interval = 1e3*(end-start3)
        timings.append(interval)
        print( "|--- Time taken to invert the linear system for p_k: ",np.round( interval,5),"ms---|" )
        start4 = time.time()
        # Unwind

        for transform in unwinding_transformations:
          p_k = transform(p_k)
        end = time.time()
        interval = 1e3*(end-start4)
        timings.append( interval )
        print("|--- Time taken for unwinding: ",np.round( interval ,5 ) ,"ms---|")
        interval = 1e3*(end-start)
        timings.append( interval )

        print("|--- Time taken for the complete code block: ",np.round( interval,2),"ms---|\n")
        return p_k, timings
      
    def _precond_inversion( self, unnormalized_Hessian : List[List[float]], 
                                  gradient : List[float],
                                  iterative_inversion : int = -1, 
                                  optType : str = None 
                                  )-> Tuple[ List[float], List[float] ]:
        """

          Args:
              unnormalized_Hessian : THe unnormalized Hessian, shape : (n,n).
              gradient : The gradient vector, shape : (n,).
              iterative_inversion : The number of iterative inversions. Defaults to -1.
              debug : To add a debug any step of the implementation when needed. Defaults to False.
              optType :  Input for the choice of iterative inversion algorithm, which here are Conjugate Gradient-'cg' and GMRES-'gmres. Defaults to 'cg'.

          Returns:
              p : The optimal direction vector.
              timings : The list of timestamps at each step.
        """
        timings = []
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
        vector_E = vector
        if iterative_inversion < 0:
          vector = vector.reshape( (len(vector), 1) )
          matrix = matrix + np.dot( vector, vector.T)
        # Transformations (Initial on gradient and final on result)
        gradient = diag[:,None]*gradient[:,None]
        unwinding_transformations.append(lambda x : diag[:,None]*x)
        # Record timing
        end = time.time()
        interval = 1e3*(end-start)
        timings.append( interval )
        # Conditioning with other vectors
        #  Naming conventions:
        #  y = Preconditioning vectors as a numpy matrix n by k
        #  matrix = our matrix A to precondition
        #  We only form the data y and z such that
        #  P = id + z*y.T
        start0 = time.time()
        y = np.array( self.precond_vectors ).T # Matrix of size n by k
        # Compute eigenvalues
        Ay = np.dot( matrix, y)
        eigenvalues = np.sum( y * Ay, axis=0 )
        # Compute data for P = id + y*diag(values)*y.T
        values = (1/np.sqrt(eigenvalues)-1)    # Vector of size k
        z = y*values[None,:]
        # Record timings
        end = time.time()
        interval = 1e3*(end-start0)
        timings.append( interval )

        # Changing the A matrix to PAP
        start2 = time.time()

        # Function mapping v to Pv
        # P = Id + z*y.T
        def _apply_P(vector):
          return vector + z @ ( y.T @ vector)
        # Function mapping v to P(A+E)Pv
        # A is matrix
        # E is vector_E*vector_E.T
        def _preconditioned_map(vector):
          vector   = _apply_P( vector )
          vector   = np.dot(matrix,vector)  + vector_E*np.dot(vector_E, vector)
          vector   = _apply_P( vector ) 
          return vector
        # Apply P
        # At beginning on gradient
        # At the end 
        gradient = _apply_P(gradient)
        unwinding_transformations.append( lambda x : _apply_P(x) )
        end      = time.time()
        interval = 1e3*(end-start2)
        timings.append( interval )  
        #
        # Solve either iteratively using CG or exactly
        start3 = time.time()
       
        if iterative_inversion >= 0:
          self.m  = matrix
          A = scipy.sparse.linalg.LinearOperator( ( self.m.shape[0],self.m.shape[1] ), matvec=_preconditioned_map ) 
          if optType == 'cg':
            inverse, exit_code = scipy.sparse.linalg.cg( A, gradient, x0 = gradient, maxiter = iterative_inversion, tol=1e-10 )
            # print( "  --- CG exit code: ", exit_code)
          else:
            inverse, exit_code = scipy.sparse.linalg.gmres( A, gradient, x0 = gradient, maxiter = iterative_inversion, tol = 1e-10 )
            # print( "  --- GMRES exit code: ", exit_code)
          p_k = self.epsilon*inverse
          p_k = p_k.reshape( (p_k.shape[0], 1) ) # For some reason, this outputs (n,) and the next line outputs (n,1)
        else:
          B = np.dot( Ay, z.T)
          C = z @ np.dot(y.T, Ay) @ z.T
          matrix = matrix + B + B.T + C
          self.Hessian_stabilized = -matrix/self.epsilon
          p_k = -np.linalg.solve( self.Hessian_stabilized, gradient)
        end = time.time() 
        interval = 1e3*(end-start3)
        timings.append(interval)
        start4 = time.time()
    
        # Unwind
        for transform in unwinding_transformations:
          p_k = transform(p_k)
        end = time.time()
        interval = 1e3*(end-start4)
        timings.append( interval )
        interval = 1e3*(end-start)
        timings.append( interval )
        # print("|--- Time taken for the complete code block: ",np.round( interval,2),"ms---|\n")
        return p_k, timings
    
    def _update(self, tol : float = 1e-12, 
                      maxiter : int = 100, 
                      iterative_inversion : int = -1, 
                      version : int = 1, 
                      debug : bool = False, 
                      optType : str = 'cg'
                      )-> Dict[str, List[float]]:
        """

        Args:
            tol : The tolerance limit for the error. Defaults to 1e-12.
            maxiter : The maximum iteration for the optimization algorithm. Defaults to 100.
            iterative_inversion : The number of iterative inversions to be used. Defaults to -1.
            version: The version of the precondioned iterative inversion to be used. Defaults to 1.
            debug : To add a debug any step of the implementation when needed. Defaults to False.
            optType :  Input for the choice of iterative inversion algorithm, which here are Conjugate Gradient-'cg' and GMRES-'gmres. Defaults to 'cg'.

        Returns:
            potential_f : The optimal Kantorovich potential f.
            potential_g : The optimal Kantorovich potential g.
            error : The list of error values over the iteration of the algorithm.
            objectives : The list of objective function values over the iterations of the algorithm.
            linesearch_steps : The list of step size along the iterations of the algorithm.
            timings : The list of timestamps.
        """
        i = 0
        while True: 
            # Compute gradient w.r.t f:
            grad_f = self._computegradientf()# Shape : (n,)
            # Compute the Hessian:
            M = self.a[:,None]*np.exp( -self.z/self.epsilon )*np.sqrt( self.b )[None,:]# Shape : (n,m)
            Sum_M = np.sum( M*np.sqrt( self.b )[None,:], axis = 1 )# Shape : (n,)
            self.Hessian = Sum_M[:,None]*np.identity( self.a.shape[0] ) - np.dot( M, M.T )  # Shape : (n,n)
            # Compute solution of Ax = b:
            if version == 4:
              print("\n At iteration: ",i)
              p_k, temp = self._precond_inversion_v4( self.Hessian, 
                                                      grad_f, 
                                                      iterative_inversion = iterative_inversion, 
                                                      optType = optType )
              self.timing.append(temp)
            elif version == 3:
              print("\n At iteration: ",i)
              p_k, temp = self._precond_inversion_v3( self.Hessian, 
                                                      grad_f, 
                                                      iterative_inversion = iterative_inversion, 
                                                      optType = optType )
              self.timing.append(temp)
            elif version == 2:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v2( self.Hessian, 
                                                      grad_f, 
                                                      iterative_inversion = iterative_inversion, 
                                                      debug = debug )
              self.timing.append(temp)
            elif version == 1:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v1( self.Hessian, 
                                                      grad_f, 
                                                      iterative_inversion = iterative_inversion, 
                                                      debug = debug )
              self.timing.append(temp)
            elif version == 0:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v0( self.Hessian,
                                                      grad_f,
                                                      iterative_inversion = iterative_inversion,
                                                      debug = debug)
              self.timing.append(temp)
            else:
              #print("At iteration: ",i)
              p_k, temp = self._precond_inversion(  self.Hessian, 
                                                    grad_f, 
                                                    iterative_inversion = iterative_inversion, 
                                                    optType = optType )
              self.timing.append(temp)
            p_k = p_k.reshape(p_k.shape[0], )# Shape : (n,)
            # Wolfe condition 1: Armijo Condition:  
            slope = np.dot( p_k, grad_f )
            alpha = 1
            alpha = self._wolfe1( alpha, p_k, slope )
            self.alpha_list.append( alpha )
            # Update f and g:
            self.f = self.f + alpha*p_k
            self.f_C = np.min( self.C - self.f[:,None], axis = 0 )# The C-transform of f, shape : (m,). 
            self.H = self.C - self.f[:,None] - self.f_C[None,:]# Shape : (n,m)
            self.g = self.f_C - self.epsilon*np.log( np.sum( self.a[:,None]*np.exp( -self.H /self.epsilon ), axis = 0 ) )# Shape : (m,)
            self.z = self.C - self.f[:,None] - self.g[None,:]# Shape : (n,m)
            P = self.a[:,None]*( np.exp( -self.z/self.epsilon ) )*self.b[None,:]# Shape : (n,m)
            # Error computation:
            self.err.append( np.linalg.norm( np.sum( P, axis = 1 ) - self.a, ord = 1 ) )
            # Calculating objective function:
            value = self._objectivefunction( self.f )
            self.objvalues.append(value)
            # Check error:
            if i< maxiter and ( self.err[-1]>tol ):
                i += 1
            else:   
                print( "Terminating after iteration: ", i )
                break  
        # end for                                                                                                            
        return {
            "potential_f"       : self.f.reshape( self.a.shape[0], ) + self.epsilon*np.log( self.a ).reshape( self.a.shape[0], ),
            "potential_g"       : self.g.reshape( self.b.shape[0], ) + self.epsilon*np.log( self.b ).reshape( self.b.shape[0], ) ,
            "error"             : self.err,
            "objectives"        : self.objvalues,
            "linesearch_steps"  : self.alpha_list,
            "timings"           : self.timing  

        }
