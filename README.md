# computational-OT

## Content

The repository is structured as follows. We only describe the most important files for a new user.
```bash
./
|-- computational-OT: Core of package. 
|  |-- grad_ascent.py : Gradient ascent
|  |-- L_BFGS_B.py    : BFGS is the goto quasi-Newton method
|  |-- linear.py      : Linear programming using cvxpy
|  |-- linesearch.py  : Gradient ascent with line search using Armijo condition
|  |-- sinkhorn.py    : Sinkhorn iterations
|-- ipynb: Contains Python notebooks which demonstrate how the code works
|  |-- Demo.ipynb: Illustrates the various methods, in particular, Sinkhorn, Gradient ascent (fixed or line search), L-BGFS, Newton...
|  |-- Multiscale.ipynb: Exploring multiscale resolution
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
pip install numpy matplotlib scipy sympy cvxpy
```

## Configuration
Nothing to do

## Credits
Later