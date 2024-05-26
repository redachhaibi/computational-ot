import numpy as np
import scipy
import time

class DampedNewton_with_precodonditioner_SemiDual_np:
    def __init__(self,K,a,b,f,g,epsilon,rho,c,null_vector,precond_vectors):
        self.K = K
        self.a = a
        self.b = b
        self.f = f
        self.g = g
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
        
    def _log_regularise(self,min,a,u):
        return np.log(np.sum(a[:,None]*u[:,None]*self.K*np.exp(min/self.epsilon)[None,:],0))
    
    def _objectivefunction(self,x):
        a_ = self.a.reshape(self.a.shape[0],)
        u = np.exp(x/self.epsilon).reshape(x.shape[0],)
        min_x = np.min(-self.epsilon*np.log(self.K)-x,0)
        y = -self.epsilon*self._log_regularise(min_x,a_,u)+min_x[None,:]
        return np.dot(x.T, self.a) + np.dot(y, self.b)    
    
    def _computegradientf(self):
        a_ = self.a.reshape(self.a.shape[0],)
        b_ = self.b.reshape(self.b.shape[0],)
        gradient = self.a-np.sum(a_[:,None]*self.u[:,None]*self.K*self.v[None,:]*b_[None,:]*np.exp(self.min_f/self.epsilon)[None,:], 1).reshape(self.a.shape[0],-1)
        return gradient

        
    def _wolfe1(self,alpha,p,slope):
        "Bactracking"
        reduction_count = 0
        while True:
            condition = self._objectivefunction( self.f+ alpha*p )<self._objectivefunction( self.f ) + self.c*alpha*slope
            if condition or np.isnan(self._objectivefunction( self.f+alpha*p)):
                alpha = self.rho*alpha
                reduction_count += 1
            else:
                break
        return alpha
    
    def _precond_inversion_v0( self, unnormalized_Hessian, gradient, iterative_inversion=-1, debug=False ):
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
        gradient = diag[:,None]*gradient
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

    def _precond_inversion_v1( self, unnormalized_Hessian, gradient, iterative_inversion=-1, debug=False ):
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
        gradient = diag[:,None]*gradient
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

    def _precond_inversion_v2( self, unnormalized_Hessian, gradient, iterative_inversion=-1, debug=False ):
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
        gradient = diag[:,None]*gradient
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

    def _precond_inversion_v3( self, unnormalized_Hessian, gradient, iterative_inversion=-1, debug=False,optType =None ):
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
        gradient = diag[:,None]*gradient
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
        self.Hessian_stabilized = -matrix/self.epsilon
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
          start = time.time()
          p_k = -np.linalg.solve( self.Hessian_stabilized, gradient )
          end = time.time()
          print("p_k inverse time stamp: ", 1e-3*(end-start))        
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
      
    def _precond_inversion_v4( self, unnormalized_Hessian, gradient, iterative_inversion=-1, debug=False, optType =None ):
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
        gradient = diag[:,None]*gradient
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
        self.Hessian_stabilized = -matrix/self.epsilon
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
          start = time.time()
          p_k = -np.linalg.solve( self.Hessian_stabilized, gradient)
          end = time.time()
          print("p_k inverse time stamp: ", 1e-3*(end-start))
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
      
    def _precond_inversion( self, unnormalized_Hessian, gradient, iterative_inversion = -1, debug = False, optType = None ):
        
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
        gradient = diag[:,None]*gradient
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
      

    def _update(self, tol = 1e-12, maxiter = 100, iterative_inversion = -1, version = 1, debug = False, optType = 'cg'):
        a_ = self.a.reshape(self.a.shape[0],)
        b_ = self.b.reshape(self.b.shape[0],)
        self.min_f = np.min(-self.epsilon*np.log(self.K)-self.f,0)
        self.u = np.exp(self.f/self.epsilon).reshape(self.f.shape[0],)
        self.g = -self.epsilon*self._log_regularise(self.min_f,a_,self.u)
        self.v = np.exp(self.g/self.epsilon).reshape(self.g.shape[0],)
        i = 0 
        while True:
            # Compute gradient w.r.t f:   
            grad_f = self._computegradientf()
            # Compute the Hessian:
            M = a_[:,None]*self.u[:,None]*self.K*self.v[None,:]*np.exp(self.min_f/self.epsilon)[None,:]*np.sqrt(b_)[None,:]
            self.Hessian = np.sum(M*np.sqrt(b_)[None,:],1)[:,None]*np.identity(self.a.shape[0])-np.dot( M , M.T ) 
            # Compute solution of Ax = b:
            if version == 4:
              print("\n At iteration: ",i)
              p_k, temp = self._precond_inversion_v4( self.Hessian, 
                                                      grad_f, 
                                                      iterative_inversion=iterative_inversion, 
                                                      debug = debug, 
                                                      optType = optType )
              self.timing.append(temp)
            elif version == 3:
              print("\n At iteration: ",i)
              p_k, temp = self._precond_inversion_v3( self.Hessian, 
                                                      grad_f, 
                                                      iterative_inversion=iterative_inversion, 
                                                      debug = debug, 
                                                      optType = optType )
              self.timing.append(temp)
            elif version == 2:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v2( self.Hessian, 
                                                      grad_f, 
                                                      iterative_inversion=iterative_inversion, 
                                                      debug=debug )
              self.timing.append(temp)
            elif version == 1:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v1( self.Hessian, 
                                                      grad_f, 
                                                      iterative_inversion=iterative_inversion, 
                                                      debug=debug )
              self.timing.append(temp)
            elif version == 0:
              print("\n At iteration: ",i)
              p_k,temp  = self._precond_inversion_v0( self.Hessian,
                                                      grad_f,
                                                      iterative_inversion=iterative_inversion,
                                                      debug=debug)
              self.timing.append(temp)
            else:
              #print("At iteration: ",i)
              p_k, temp = self._precond_inversion(  self.Hessian, 
                                                    grad_f, 
                                                    iterative_inversion=iterative_inversion, 
                                                    debug = debug, 
                                                    optType = optType )
              self.timing.append(temp)

            start = time.time()
            # Wolfe condition 1( Armijo condition ):
            slope  = np.dot(p_k.T, grad_f)[0][0]
            alpha = 1
            alpha = self._wolfe1( alpha, p_k, slope )
            end  = time.time()
            self.alpha_list.append( alpha ) 
            # Update f and g:
            self.f = self.f + alpha*p_k
            self.min_f = np.min(-self.epsilon*np.log(self.K)-self.f,0)
            self.u  = np.exp(self.f/self.epsilon).reshape(self.f.shape[0],)
            self.g = -self.epsilon*self._log_regularise(self.min_f,a_,self.u)
            self.v  = np.exp(self.g/self.epsilon).reshape(self.g.shape[0],)
            # Error computation:
            P  =  a_[:,None]*(self.u[:,None]*self.K*self.v[None,:]*np.exp(self.min_f/self.epsilon)[None,:])*b_[None,:]
            self.err.append(np.linalg.norm(np.sum(P,1)-self.a,1))
            # Calculating objective function:
            value = self._objectivefunction(self.f)
            self.objvalues.append(value[0])
            # Check error:
            if i< maxiter and (self.err[-1]>tol ):
                i+=1
            else:   
                print("Terminating after iteration: ",i)
                break            
            
        return {
            "potential_f"       : self.f.reshape(self.a.shape[0],),
            "potential_g"       : self.g.reshape(self.b.shape[0],)+self.min_f,
            "error"             : self.err,
            "objectives"        : self.objvalues,
            "linesearch_steps"  : self.alpha_list,
            "timings"           : self.timing,
        }


