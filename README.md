# computational-OT

This package is work in progress in the context of Anirban BOSE's phD thesis.
The goal is to study numerical methods for optimal transport.

## Content

The repository is structured as follows. We only describe the most important files for a new user.
```bash
./
|-- computational-OT: Core of package. 
|  |-- _Cuthill_Mckee.py        : Use of (Reverse) Cuthill-Mckee to fleshout sparsity
|  |-- Damped_Newton_precond.py : (Our method) Damped Newton with iterative inversion using preconditioning
|  |-- Damped_Newton.py         : The classical damped Newton-Raphson algorithm with exact inversion of the Hessian
|  |-- grad_ascent.py : Gradient ascent
|  |-- L_BFGS_B.py    : BFGS is the goto quasi-Newton method
|  |-- linear.py      : Linear programming using cvxpy
|  |-- linesearch.py  : Gradient ascent with line search using Armijo condition
|  |-- sinkhorn.py    : Sinkhorn iterations
|  |-- logdomain_sinkhorn.py    : exp-log regularized Sinkhorn iterations
|  |-- Damped_Newton_SemiDual_numpy.py    : Semi-dual damped Newton with exact inversion of the Hessian
|  |-- Damped_Newton_precond_SemiDual_numpy.py    : Semi-dual damped Newton with iterative inversion using preconditioning 

|-- ipynb: Contains Python notebooks which demonstrate how the code works
|  |-- Damped Newton: Notebooks related to damped Newton.
|  |  |-- DampedNewton.ipynb: Illustration of the method (Self explanatory).
|  |  |-- DampedNewtonPreconditioning.ipynb: Illustration of the method (Self explanatory).
|  |  |-- DampedNewtonPreconditionigFinal.ipynb : Illustration of the final version of the damped Newton with preconditioning method.
|  |  |-- DampedNewtonPreconditoningAllversions.ipynb : Illustration of all the version of the damped Newton with preconditoning method.
|  |  |-- DampedNewtonPreconditioning_fordiffparams.ipynb : Illustration of the damped Newton with preconditioning methond for different hyperparameters.
|  |  |-- SinkhornvsDampedNewton.ipynb : Illustration of the performance of the Sinkhorn algorithm and damped Newton algorithm.
|  |  |-- Spectralplots.ipynb : Illustration of the spectral plots from the Hessian obtained from algorithms: Sinkhorn, log-domain Sinkhorn and damped Newton.
|  |-- Sparsity: Notebooks related to the sparsity analysis.
|  |   |-- Multiscale.ipynb : Exploring multiscale resolution.
|  |   |-- NewtonSparsity-RCM-CAH.ipynb : Exploring sparsity as $\epsilon$ goes to zero.
|  |   |-- NewtonSparsityExperiments.ipynb : Illustration to understand the sparsity using Cuthill Mckee algorithm and nested dissection method.
|  |-- Sinkhorn: Notebooks related to Sinkhorn algorithm.
|  |   |-- Sinkhorn.ipynb  : Benchmarking of Sinkhorn.
|  |   |-- log_domain_Sinkhorn_versions.ipynb  : Benchmarking of exp-log regularized Sinkhorn.
|  |-- Others: Notebooks containing analysis across different algorithms.
|  |   |-- Correctness.ipynb : Notebook to compare the convergence of the Kantorovich potentials for different algorithm with respect ot the exp-log regularized Sinkhorn algorithm as the ground truth.
|  |   |-- Robutness.ipynb : Illustration of the robustness of the eigenvalues of the Hessian obtained from the log-domain Sinkhorn.
|  |   |-- Demo.ipynb : Illustrates the various methods, in particular, Sinkhorn, Gradient ascent (fixed or line search), L-BGFS, Newton...
|  |-- Semi-dual damped Newton: Notebooks related to semi-dual damped Newton.
|  |   |-- SemiDual_damped_Newton_jax.ipynb    : Benchmarking of Semi-dual damped Newton in jax.
|  |   |-- SemiDual_damped_Newton_numpy.ipynb  : Benchmarking of Semi-dual damped Newton in numpy.
|-- tests: Unit tests
|-- README.md: This file
```

## Installation

1. Create new virtual environment

```bash
$ python3 -m venv .venv_computational-OT
```

(Do
sudo apt install python3-venv
if needed)

3. Activate virtual environment

```bash
$ source .venv_computational-OT/bin/activate
```

4. Upgrade pip, wheel and setuptools 

```bash
$ pip install --upgrade pip
$ pip install --upgrade setuptools
$ pip install wheel
```

5. Install the `computational-OT` package.

```bash
python setup.py develop
```

6. (Optional) In order to use Jupyter with this virtual environment .venv
```bash
pip install ipykernel
python -m ipykernel install --user --name=.venv_computational-OT
```
(see https://janakiev.com/blog/jupyter-virtual-envs/ for details)

7. (Not needed if step 5 is used) Packages
```bash
pip install numpy matplotlib scipy sympy cvxpy torch
```

## Configuration
Nothing to do

## Credits
Later