import numpy as np
import cvxpy as cp

class linear_optimization:

  def __init__( self, N1, N2, a, b, Cost ):
    """
    
    Parameters:
    -----------
        (N1,N2) : int
                  Sizes of two sample point clouds
        a : ndarray, shape (N1,1)
            The probability histogram of the sample point cloud of size N1.
        b : ndarray, shape (N2,1)
            The probability histogram of the sample point cloud of size N2.
        Cost :  ndarray, shape (N1, N2)
                The cost matrix of size N1 by N2.
    """
    self.N1 = N1
    self.N2 = N2
    self.a = a.reshape( a.shape[0], -1 )
    self.b = b.reshape( b.shape[0], -1 )
    self.Cost = Cost
  
  def solve( self ):
    """
    
    Returns:
    --------
          P : ndarray, shape (N1,N2)
              The optimal coupling.
    """
    # Variables
    P = cp.Variable( ( self.N1, self.N2 ) )
    v = np.ones( ( self.N2, self.a.shape[1] ) )
    u = np.ones( ( self.N1, self.b.shape[1] ) )
    # Optimization constraint
    U = [ 0 <= P, cp.matmul( P, v ) == self.a, cp.matmul( P.T, u ) == self.b ]
    # Objective function
    objective = cp.Minimize( cp.sum( cp.multiply( P, self.Cost  ) ) )
    # Optimization problem
    prob = cp.Problem( objective, U )
    # Solving it
    result = prob.solve()
    # Error computation
    error = np.linalg.norm( np.sum( P.value, axis = 1 ) - self.a, ord = 1 )
    return {
      "Optimal coupling"  : P,
      "Error"             : error
    }
