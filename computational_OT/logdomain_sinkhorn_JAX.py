import jax.numpy as jnp
from jax import jit

# Define functions outside of the class
@jit
def mina_u(a, H, epsilon):
    return -epsilon * jnp.log(jnp.sum(a[:, None] * jnp.exp(-H / epsilon), axis=0))
@jit
def minb_u(b, H, epsilon):
    return -epsilon * jnp.log(jnp.sum(b[None, :] * jnp.exp(-H / epsilon), axis=1))
@jit
def mina(a, H, epsilon):
    return mina_u(a, H - jnp.min(H, axis=0), epsilon) + jnp.min(H, axis=0)
@jit
def minb(b, H, epsilon):
    return minb_u(b, H - jnp.min(H, axis=1)[:, None], epsilon) + jnp.min(H, axis=1)

def update(a, b, C, epsilon, tol=1e-12, niter=500):
    error = []
    f, g = a, b
    for i in range(niter):
        g = mina(a, C - f[:, None], epsilon)
        f = minb(b, C - g[None, :], epsilon) 
        # generate the coupling
        P = a[:, None] * jnp.exp((f[:, None] + g[None, :] - C) / epsilon) * b[None, :]
        # check conservation of mass
        error.append(jnp.linalg.norm(jnp.sum(P, axis=0) - b, 1))
        if error[i] <= tol:
            print("Terminating after iteration: ", i)
            break
    # end for
    if i + 1 >= niter:
        print("Terminating after maximal number of iterations: ", niter)
    return {
        'error'       : error,
        'potential_f' : f + epsilon * jnp.log(a).reshape(a.shape[0],),
        'potential_g' : g + epsilon * jnp.log(b).reshape(b.shape[0],)  # Change of convention because of line (*)
    }