from setuptools import Extension, setup, find_packages
from os import path

local_path = path.abspath(path.dirname(__file__))

print("Local path: ", local_path)
print("")

print("Launching setup...")
# Setup
setup(
    name = 'computational_OT',

    version = '0.01',

    description = 'Computational aspects of optimal transport',
    long_description = """ Computational solutions to Free Deconvolution.
    In this module, we explore and benchmark various computational methods 
    for computing (regularized) optimal transport.
    """,
    url = '',

    author = 'Anonymous',
    author_email = 'Anonymous',

    license = 'MIT License',

    install_requires = ["numpy", "matplotlib", "scipy", "cvxpy", "scikit-learn"],

    keywords = '',

    packages = find_packages(),

    entry_points = {
        'console_scripts': [
            'sample=sample:main',
        ],
    },

    ext_modules = [],
)