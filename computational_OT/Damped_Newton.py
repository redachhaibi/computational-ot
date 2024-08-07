import numpy as np
import logging
class DampedNewton:
      def __init__( self, K, a, b, f, g, epsilon, rho, c ):
        """
        
        Parameters:
        -----------
          K : ndarray, shape (n,m)
              The Gibb's kernel.
          a : ndarray, shape (n,)
                The probability histogram of the sample of size n.
          b : ndarray, shape (m,)
                The probability histogram of the sample of size m.
          f : ndarray, shape (n,)
              The initial Kantorovich potential f.
          g : ndarray, shape (m,)
              The initial Kantorovich potential g.
          rho : float
                Damping factor for the line search update step.
          c : float
              Damping factor for the slope in the Armijo's condition.
          epsilon : float
                    The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
                    
        """
        self.K = K
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.x = np.concatenate( ( f, g ), axis = None )
        self.rho = rho
        self.c = c
        # null vector
        null_vector = np.hstack( ( np.ones( a.shape[0] ), - np.ones( b.shape[0] ) ) )/np.sqrt( a.shape[0] + b.shape[0] )
        null_vector = np.reshape( null_vector, ( a.shape[0] + b.shape[0], 1 ) )
        self.null_vector = null_vector
        self.reg_matrix = np.dot( null_vector, null_vector.T )
        #
        self.alpha = []
        self.err_a = []
        self.err_b = [] 
        self.objvalues = []



      def _computegradientf( self, f ):
        """
        Computes Gradient with respect to f
        
        Parameters:
        -----------
          f : ndarray, shape (n,)
              The Kantorovich potential f.
        """
        u = np.exp( f/self.epsilon )
        v = np.exp( self.x[self.a.shape[0]:]/self.epsilon )
        return self.a - ( u * np.dot( self.K,v ) )

 
      def _computegradientg( self, g ):
        """
        Computes Gradient with respect to g
        
        Parameters:
        -----------
          g : ndarray, shape (m,)
              The Kantorovich potential g.

        """
        u = np.exp( self.x[:self.a.shape[0]]/self.epsilon )
        v = np.exp( g/self.epsilon )
        return self.b - ( v * np.dot( self.K.T, u ) )

      def _objectivefunction( self, x ):
        """
        
        Parameters:
        -----------
          x : ndarray, shape (n+m,)
              The stacked vector containing the potentials f and g.
            
        Returns:
        --------
          Q(f,g) :  float
                    The value of objective function obtained by evaluating the formula Q(f,g) = < f, a > + < g, b > - epsilon*< u, Kv >,
                    where u = exp( f/epsilon ), v = exp( g/epsilon ). 
          
        """
        f = x[:self.a.shape[0]]
        g = x[self.a.shape[0]:]
        regularizer = - self.epsilon * np.dot( np.exp( f/self.epsilon ).T, np.dot( self.K, np.exp( g/self.epsilon ) ) )
        return np.dot( f.T, self.a ) + np.dot( g.T, self.b ) + regularizer

      def _wolfe1( self, alpha, p, slope ):
        #Armijo Condition
        """

        Parameters:
        -----------
            alpha : float
                    The update step size.
            p : ndarray, shape (n,)
                The optimal direction.
            slope : float
                    It is the inner product of the gradient and p.

        Returns:
        --------
            alpha : float
                    The updated step size.
        """
        reduction_count = 0           
        while True:   
            condition = self._objectivefunction( self.x + alpha*p ) < self._objectivefunction( self.x ) + self.c * alpha * slope
            if condition or np.isnan( self._objectivefunction( self.x + alpha*p ) ):
                alpha = self.rho * alpha                                                     
                reduction_count += 1
            else:
                break
        return alpha
        


      def _update( self, tol = 1e-12, maxiter = 100 ):
        """

        Parameters:
        ----------
            tol : float
                  The tolerance limit for the error. Defaults to 1e-12.
            maxiter : int 
                      The maximum iteration for the optimization algorithm. Defaults to 100.
        Returns:
        --------
        Returns a dictionary where the keys are strings and the values are ndarrays.
        The following are the keys of the dictionary and the descriptions of their values:
            potential_f : ndarray, shape (n,)
                          The optimal Kantorovich potential f.
            potential_g : ndarray, shape (m,)
                          The optimal Kantorovich potential g.
            error : ndarray, shape (k,), where k is the number of iterations
                    Errors observed over the iteration of the algorithm.
            objectives : ndarray, shape (k,), where k is the number of iterations
                         Objective function values obtained over the iterations of the algorithm.
            linesearch_steps :  ndarray, shape (k,), where k is the number of iterations
                                Different step sizes obtained by using the Armijo's rule along the iterations of the algorithm.
        """
        i = 0
        while True :
            
            grad_f = self._computegradientf( self.x[:self.a.shape[0]] )
            grad_g = self._computegradientg( self.x[self.a.shape[0]:] )
        
            gradient = np.concatenate( ( grad_f, grad_g ), axis = None )
            
            
            # Compute Hessian
            u = np.exp( self.x[:self.a.shape[0]]/self.epsilon )
            v = np.exp( self.x[self.a.shape[0]:]/self.epsilon )
            r1 = u * np.dot( self.K,v )
            r2 = v * np.dot( self.K.T, u )
            # P  = u*self.K*(v.T) # WRONG AGAIN: DANGEROUS CODE!!
            u = u.reshape( u.shape[0], )
            v = v.reshape( v.shape[0], )
            P = u[:,None] * self.K * v[None,:]

            A = np.diag( np.array( r1.reshape( r1.shape[0], ) ) )
            B = P
            C = P.T
            D = np.diag( np.array( r2.reshape( r2.shape[0], ) ) )
            result = np.vstack( ( np.hstack( ( A, B ) ), np.hstack( ( C ,D ) ) ) )

            self.Hessian = - result/self.epsilon

            # Debug
            # if debug:
            #   scaling = 1.0/np.mean( np.diag(self.Hessian) )
            #   eig, v = np.linalg.eigh( self.Hessian * scaling )
            #   sorting_indices = np.argsort(eig)
            #   eig = eig[sorting_indices]
            #   v   = v[:, sorting_indices]
            #   #
            #   empirical_null_vector = v[:,0]
            #   empirical_null_vector = np.reshape( empirical_null_vector, (len(empirical_null_vector),1) )
            #   self.reg_matrix = np.dot( empirical_null_vector, empirical_null_vector.T)
            #   #
            #   print( "--- Unstabilized")
            #   print( "List of --smallest eigenvalues: ", eig[:3])
            #   print( "        |-largest  eigenvalues: ", eig[-3:])
            #   print( "        |- sum of  eigenvalues: ", eig[:3]+np.flip(eig[-3:]) )
              # print( "Null vector vs empirical:", np.dot(self.null_vector.flatten(), empirical_null_vector.flatten()) )
              # print( "Gradient along -- 0  :", np.dot(gradient.flatten(), v[:,0])/np.linalg.norm(gradient) )
              # print( "               |- 1  :", np.dot(gradient.flatten(), v[:,1])/np.linalg.norm(gradient) )
              # print( "               |- 2  :", np.dot(gradient.flatten(), v[:,2])/np.linalg.norm(gradient) )
              # print( "               |- m-1:", np.dot(gradient.flatten(), v[:,-2])/np.linalg.norm(gradient) )
              # print( "               |- m  :", np.dot(gradient.flatten(), v[:,-1])/np.linalg.norm(gradient) )
              #print( "Null vector vs gradient :", np.dot(self.null_vector.flatten(), gradient.flatten()) )
              #print( "Null vector vs p_k      :", np.dot(self.null_vector.flatten(), p_k.flatten()) )

            
            
            # Inflating the corresponding direction
            mean_eig = - ( 0.5 * np.mean( r1 ) + 0.5 * np.mean( r2 ) )/self.epsilon
            self.Hessian_stabilized = self.Hessian + mean_eig * self.reg_matrix
            # eig, v = np.linalg.eigh(self.Hessian_stabilized)
            # sorted_indices = np.argsort(eig)
            # v = v[:,sorted_indices]
            # print(len(np.where(eig < 1e-1000)[0]))
            # k=len(np.where(eig < 1e-1000)[0])
            # print("Condition number: ", np.max(eig)/np.min(eig))
            # if self.epsilon <= 0.02:
            #   for eigv in v[:k]:
            #       self.Hessian_stabilized = self.Hessian_stabilized + mean_eig*np.dot(eigv[:, None] , eigv[:, None].T)
            
            # Debug
            # if debug:
            #   scaling = 1.0/np.mean( np.diag(self.Hessian_stabilized) )
            #   eig, v = np.linalg.eigh( self.Hessian_stabilized * scaling )
            #   sorting_indices = np.argsort(eig)
            #   eig = eig[sorting_indices]
            #   v   = v[:, sorting_indices]
            #   #
            #   empirical_null_vector = v[:,0]
            #   empirical_null_vector = np.reshape( empirical_null_vector, (len(empirical_null_vector),1) )
            #   self.reg_matrix = np.dot( empirical_null_vector, empirical_null_vector.T)
            #   #
            #   print( "--- Stabilized")
            #   print( "List of --smallest eigenvalues:mean_eig ", eig[:3])
            #   print( "Null vector vs empirical:", np.dot(self.null_vector.flatten(), empirical_null_vector.flatten()) )
            #   print( "")
              #print( "Null vector vs gradient :", np.dot(self.null_vector.flatten(), gradient.flatten()) )
              #print( "Null vector vs p_k      :", np.dot(self.null_vector.flatten(), p_k.flatten()) )
            # eig,v = np.linalg.eigh(self.Hessian_stabilized)
            try:
              p_k = - np.linalg.solve( self.Hessian_stabilized,gradient )
            except:
              print( "Inverse does not exist at epsilon:", self.epsilon )
              return np.zeros( 6 )

            p_k = p_k - self.null_vector.flatten() * np.dot( self.null_vector.flatten(), p_k.flatten() )
            # 
            # if debug:
            #   cos_metric = np.dot( p_k.flatten(), gradient.flatten())/( np.linalg.norm(p_k)*np.linalg.norm(gradient) )
            #   print( "Cos metric: ", cos_metric)
            #   if cos_metric<0:
            #     p_k = -p_k

            # Stacked
            p_k_stacked = np.concatenate( ( p_k[:self.a.shape[0]], p_k[self.a.shape[0]:] ), axis = None )
 
            # Wolfe Condition 1: Armijo Condition  
            slope = np.dot( p_k.T, gradient )
            alpha = 1
            alpha = self._wolfe1( alpha, p_k_stacked, slope )
            self.alpha.append( alpha )

            # Update x = f and g
            self.x = self.x + alpha * p_k_stacked
          
            # error computation 1
            s = np.exp( self.x[:self.a.shape[0]]/self.epsilon ) * np.dot( self.K, np.exp( self.x[self.a.shape[0]:]/self.epsilon ) )
            self.err_a.append( np.linalg.norm( s - self.a ) )

            # error computation 2
            r = np.exp( self.x[self.a.shape[0]:]/self.epsilon ) * np.dot( self.K .T, np.exp(self.x[:self.a.shape[0]]/self.epsilon ) )
            self.err_b.append( np.linalg.norm( r - self.b ) )

            # Calculating Objective values
            value = self._objectivefunction( self.x )
            self.objvalues.append( value )
            
            if i < maxiter and ( self.err_a[-1] > tol or self.err_b[-1] > tol ) :
                 i += 1
            else:
                print( "Terminating after iteration: ", i )
                break
      
        # end for
        return {
          "potential_f"      : self.x[:self.a.shape[0]],
          "potential_g"      : self.x[self.a.shape[0]:],
          "error_a"          : self.err_a,
          "error_b"          : self.err_b,
          "objectives"       : self.objvalues,
          "linesearch_steps" : self.alpha       
        }    
        
#Footer
