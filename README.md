# LearningLowRankMatrices
Software supplement for the paper  "Predictive Low Rank Matrix Learning under
Partial Observations: Mixed-Projection ADMM"  by Dimitris Bertsimas and Nicholas
A. G. Johnson, for which a preprint is available
[here](https://arxiv.org/abs/2407.13731).

## Introduction

The software in this package is designed to provide high quality feasible
solutions to the predictive low rank matrix learning problem under partial
observations given by

`min_{X, alpha} \sum_{(i, j) \in \Omega}(X_{ij}-A_{ij})^2 + \lambda * ||Y - X * \alpha||_2^2 + \frac{\gamma}||X||_\star`

`s.t. rank(X) <= k`

using the Mixed-Projection ADMM algorithm described in the paper "Predictive Low
Rank Matrix Learning under Partial Observations: Mixed-Projection ADMM" by
Dimitris Bertsimas and Nicholas A. G. Johnson. We provide a multithreaded
implementation of the algorithm.

## Installation and set up (TODO)

In order to run this software, you must install a recent version of Julia from
http://julialang.org/downloads/. This code was developed using Julia 1.7.3.

Several packages must be installed in Julia before the code can be run.  These
packages can be found in "lowRankMatrixLearning.jl". The code was last tested
using the following package versions:

- Distributions v0.25.70
- LowRankApprox v0.5.2
- MatrixImpute v0.3.2
- PyCall v1.94.1
- TSVD v0.4.3

## Use of the admm()function

The key method in this package is admm(). This method accepts four required
arguments: `A`, `k`, `Y`, `lambda`, as well as several optional arguments which
are described in the function docstring. The four required arguments correspond
to the input data to the optimization problem.

## Citing lowRankMatrixLearning.jl

If you use lowRankMatrixLearning.jl, we ask that you please cite the following
[paper](https://arxiv.org/pdf/2306.04647.pdf):

```
@article{bertsimas2024predictive,
  title={Predictive Low Rank Matrix Learning under Partial Observations: Mixed-Projection ADMM},
  author={Bertsimas, Dimitris and Johnson, Nicholas AG},
  journal={arXiv preprint arXiv:2407.13731},
  year={2024}
}
```

## Thank you

Thank you for your interest in lowRankMatrixLearning. Please let us know if
you encounter any issues using this code, or have comments or questions.  Feel
free to email us anytime.

Dimitris Bertsimas
dbertsim@mit.edu

Nicholas A. G. Johnson
nagj@mit.edu
