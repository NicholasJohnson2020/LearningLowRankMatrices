using SparseArrays, LinearAlgebra, Random, Distributions, TSVD, OptimKit
using JuMP, Mosek, MosekTools
using MatrixImpute
using PyCall
using Distributed
using Dates

#using Conda
#Conda.pip_interop(true)
#Conda.pip("install", "fancyimpute")

fancyImpute = pyimport("fancyimpute")

include("src/scaledGD.jl")
include("src/altMin.jl")
include("src/admm.jl")
include("src/admmV2.jl")
include("src/perspectiveFormulations.jl")
include("src/benchmarkMethods.jl")
include("scripts/generateSyntheticData.jl")
;
