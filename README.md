# Preconditioners

This library implements a set of useful algebraic (incomplete-factorization-based) preconditioners that can be used to accelerate the convergence of iterative linear-algebraic methods (such as the conjugate-gradient, MINRES, Lanczos, or LOBPCG methods).  This library builds upon the incomplete Crout symmetric indefinite LDL factorization provided by the [sym-ildl](https://cs.stanford.edu/people/paulliu/sym-ildl/html/index.html) library, implementing (optional) *inertia-correction* to ensure that the constructed preconditioners are *positive-definite* (which many inexact linear-algebraic methods require), as well as presenting an interface that permits easy integration with [Eigen](https://eigen.tuxfamily.org/index.php)

## Getting Started

This library can be built and exported as a CMake project.  The following installation instructions have been verified on Ubuntu 20.04:

*Step 1:*  Install dependencies
```
$ sudo apt-get install build-essential cmake-gui libeigen3-dev liblapack-dev libblas-dev libsuitesparse-dev
```

*Step 2:*  Clone the repository
```
$ git clone https://github.com/david-m-rosen/SE-Sync.git SESync
```

*Step 3:*  Initialize Git submodules
```
$ cd SESync
$ git submodule init
$ git submodule update
```

*Step 4:*  Create build directory
```
$ cd C++ && mkdir build
```

*Step 5:*  Configure build and generate Makefiles
```
$ cd build && cmake ..
```

*Step 6:*  Build code
```
$ make [-jN, where N is the number of cores to use for parallel compilation]
```

*Step 7:*  Run the example command-line utility on some tasty data :-D!
```
$ cd bin
$ ./SE-Sync ../../../data/sphere2500.g2o 
```

## References

We are making this software freely available in the hope that it will be useful to others. If you use SE-Sync in your own work, please cite our [paper](https://arxiv.org/abs/2207.05257), which describes the design of the inertia-corrected symmetric indefinite preconditioner implemented in the library:

```
@misc{Rosen2022Accelerating,
  title = {Accelerating Certifiable Estimation with Preconditioned Eigensolvers},
  author = {Rosen, David M.},
  month = may,
  year = {2022},
  publisher = {arXiv},
  doi = {10.48550/ARXIV.2207.05257},
  url = {https://arxiv.org/abs/2207.05257},
}
```

and the following [paper](https://dl.acm.org/doi/abs/10.1145/3054948) of Greif et al., which describes the design of the incomplete Crout symmetric indefinite factorization method that this library employs:

```
@article{Greif2017SymILDL,
title = {{SYM-ILDL}: Incomplete {$LDL\transpose$} Factorization of Symmetric Indefinite and Skew-Symmetric Matrices},
author = {Greif, C. and He, S. and Liu, P.},
journal = toms,
volume = {44},
number = {1},
month = apr,
year = {2017},
}
```

## Copyright and License 

The Preconditioners implementations contained herein are copyright (C) 2016-2022 by David M. Rosen, and are distributed under the terms of the GNU Lesser General Public License (LGPL) version 3 (or later).  Please see the [LICENSE] for more information.

The modified version of the [sym-ildl](https://cs.stanford.edu/people/paulliu/sym-ildl/html/index.html) library redistributed with this project is released under the [MIT license].

Contact: d.rosen@northeastern.edu
