import torch
torch.set_default_dtype(torch.float64)
class logdomain_Sinkhorn_torch:
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
                      The regularization parameter.
        """
        self.a = a
        self.b = b
        self.C = C
        self.epsilon = epsilon
        self.error = []

    def mina_u( self, H ):
        """

        Parameters:
        -----------
            H : ndarray, shape (n,m)
                It is the matrix obtained from C - f.                    
        Returns:
        --------
            ndarray, shape (m,)
            The value of potential g obtained from the Schrodinger-bridge equation between the potentials f and g.
        """
        return -self.epsilon * torch.log( torch.sum( self.a[:, None] * torch.exp( -H / self.epsilon ), 0 ) )

    def minb_u( self, H ):
        """

        Parameters:
        -----------
            H : ndarray, shape (n,m)
                It is the matrix obtained from C - g.    
        Returns:
        --------
            ndarray, shape (n,)
            The value of potential f obtained from the Schrodinger-bridge equation between the potentials f and g.
        """
        return -self.epsilon * torch.log( torch.sum( self.b[None, :] * torch.exp( -H / self.epsilon ), 1 ) )

    def mina( self, H ):
        """

        Parameters:
        -----------
            H : ndarray, shape (n,m)
                It is the matrix obtained from C - f.
        Returns:
        --------
            ndarray, shape (m,)
            The exp-log regularized g.
        """
        return self.mina_u( H - torch.min( H, 0 )[0] ) + torch.min( H, 0 )[0]

    def minb( self, H ):
        """

        Parameters:
        -----------
            H : ndarray, shape (n,m)
                It is the matrix obtained from C - g.
        Returns:
        --------
            ndarray, shape (n,)
            The exp-log regularized f.
        """
        return self.minb_u( H - torch.min( H, 1 )[0][:, None]) + torch.min( H, 1 )[0]

    def update(self, tol = 1e-12, max_iterations = 500):
        """
        
        Parameters:
        -----------
            tol : float
                  The tolerance limit for the error. Defaults to 1e-12.
            max_iterations : int
                             The maximum iteration for the optimization algorithm. Defaults to 500.

        Returns:
        --------
        Returns a dictionary where the keys are strings and the values are ndarrays or list.
        The following are the keys of the dictionary and the descriptions of their values:
            potential_f : ndarray, shape (n,)
                          The optimal Kantorovich potential f.
            potential_g : ndarray, shape (m,)
                          The optimal Kantorovich potential g.
            error : list
                    The list of errors observed when checking conservation of mass .
        """
        f, g = self.a, self.b
        for i in range( max_iterations ):
            g = self.mina( self.C - f[:, None] )
            f = self.minb( self.C - g[None, :] )
            # Generate the coupling
            P = self.a[:, None] * torch.exp(( f[:, None] + g[None, :] - self.C ) / self.epsilon) * self.b[None, :]
            # Check conservation of mass
            self.error.append(  torch.linalg.norm( torch.sum( P, 1 ) - self.a, 1 ).item()
                                +
                                torch.linalg.norm( torch.sum( P, 0 ) - self.b, 1 ).item() )
            if self.error[i] < tol:
                print( "Terminating after iteration: ", i )
                break
        # end for
        if i + 1 >= max_iterations:
            print( "Terminating after maximal number of iterations: ", max_iterations )
        return {
            "error"         : self.error,
            "potential_f"   : ( f + self.epsilon * torch.log( self.a ) ).reshape( self.a.shape[0], ),
            "potential_g"   : ( g + self.epsilon * torch.log( self.b ) ).reshape( self.b.shape[0], ),
        }


