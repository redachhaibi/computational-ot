import numpy as np
class Log_domainSinkhorn:
    def __init__( self, a, b, C, epsilon ):
        """
        
        Parameters:
        -----------
            C : ndarray, shape (n,m), 
                n and m are the sizes of the samples from the two point clouds.
                It is the cost matrix between the sample points of the two point clouds.
            a : ndarray, shape (n,)
                The probability histogram of the sample of size n.
            b : ndarray, shape (m,)
                The probability histogram of the sample of size m.
            epsilon : float
                      The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
        """
        self.a       = a
        self.b       = b
        self.C       = C
        self.epsilon = epsilon
        self.error   = []
    
    def mina_u( self, H ):
        """

        Parameters:
        -----------
            a : ndarray, shape (n,)
                The probability histogram of the sample of size n.
            H : ndarray, shape (n,m)
                It is the matrix obtained from C - f.
            epsilon :   float
                        The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
                    
        Returns:
        --------
            ndarray, shape (m,)
            The value of potential g obtained from the Schrodinger-bridge equation between the potentials f and g.
        """
        return -self.epsilon * np.log( np.sum( self.a[:,None] * np.exp( -H/self.epsilon ), 0 ) )
    def minb_u( self,H ):
        """

        Parameters:
        -----------
            b : ndarray, shape (m,)
                The probability histogram of the sample of size m.
            H : ndarray, shape (n,m)
                It is the matrix obtained from C - g.
            epsilon :   float
                        The regularization factor in the entropy regularized optimization setup of the optimal transport problem.
    
        Returns:
        --------
            ndarray, shape (n,)
            The value of potential f obtained from the Schrodinger-bridge equation between the potentials f and g.
        """
        return - self.epsilon * np.log( np.sum( self.b[None,:] * np.exp( -H/self.epsilon ), 1 ) )  
    def mina( self,H ):
        """

        Parameters:
        -----------
            a : ndarray, shape (n,)
                The probability histogram of the sample of size n.
            H : ndarray, shape (n,m)
                It is the matrix obtained from C - f.
            epsilon :  float
                       The regularization factor in the entropy regularized optimization setup of the optimal transport problem.

        Returns:
        --------
            ndarray, shape (m,)
            The value of potential g obtained from the Schrodinger-bridge equation between the potentials f and g after log-exp regularization.

        """
        return self.mina_u( H - np.min( H, 0 ) ) + np.min( H, 0 )
    def minb( self, H ):
        """

        Parameters:
        -----------
            b : ndarray, shape (m,)
                The probability histogram of the sample of size m.
            H : ndarray, shape (n,m)
                It is the matrix obtained from C - g.
            epsilon :  float
                       The regularization factor in the entropy regularized optimization setup of the optimal transport problem.

        Returns:
        --------
            ndarray, shape (n,)
            The value of potential g obtained from the Schrodinger-bridge equation between the potentials f and g after log-exp regularization.
        """
        return self.minb_u( H - np.min( H, 1 )[:,None] ) + np.min( H, 1 )
    
    def update( self, tol = 1e-12, niter = 500 ):     
        """
        
        Parameters:
        -----------
            tol  :  float
                    The tolerance limit for the error. Defaults to 1e-12.
            niter : int
                    The maximum iteration for the optimization algorithm. Defaults to 500.

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
        """
        f, g = self.a, self.b
        for i in range( niter ):
            g = self.mina( self.C - f[:,None] )
            f = self.minb( self.C - g[None,:] ) 
            # generate the coupling
            P = self.a[:,None] * np.exp( ( f[:,None] + g[None,:] - self.C )/self.epsilon ) * self.b[None,:] # line (*)
            # check conservation of mass
            self.error.append( np.linalg.norm( np.sum( P, 0 ) - self.b, 1 ) )
            if self.error[i] < tol:
                print( "Terminating after iteration: ", i )
                break
        #end for
        if i + 1 >= niter:
            print( "Terminating after maximal number of iterations: ", niter )
        return {
            'error'      : self.error,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
            'potential_f': f + self.epsilon * np.log(self.a).reshape(self.a.shape[0],),
            'potential_g': g + self.epsilon * np.log(self.b).reshape(self.b.shape[0],)  #Change of convention because of line (*)
        }


