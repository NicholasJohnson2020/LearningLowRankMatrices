import Base: eltype, size, adjoint, *

struct LowRankMat
    U::Matrix{Float64}
    V::Matrix{Float64}
end

*(A::LowRankMat, x::Vector{Float64}) = A.U * (A.V' * x)

*(A::LowRankMat, X::Matrix{Float64}) = A.U * (A.V' * X)

*(X::Adjoint{Float64, Matrix{Float64}}, A::LowRankMat) = (X * A.U) * A.V'

function eltype(A::LowRankMat)
    return eltype(A.U)
end

function size(A::LowRankMat)
    return (size(A.U)[1], size(A.V)[1])
end

function size(A::LowRankMat, k::Int64)
    return size(A)[k]
end

function adjoint(A::LowRankMat)
    return LowRankMat(A.V, A.U)
end

function subSketch(A::LowRankMat, order::Int64)
    indices = rand(1:size(A, 2), order)
    return LowRankMat(A.U, A.V[indices, :])
end

struct SymLowRankMat
    L::Matrix{Float64}
end

*(A::SymLowRankMat, x::Vector{Float64}) = A.L * (A.L' * x)

*(A::SymLowRankMat, X::Matrix{Float64}) = A.L * (A.L' * X)

*(X::Adjoint{Float64, Matrix{Float64}}, A::SymLowRankMat) = (X * A.L) * A.L'

function eltype(A::SymLowRankMat)
    return eltype(A.L)
end

function size(A::SymLowRankMat)
    return (size(A.L)[1], size(A.L)[1])
end

function size(A::SymLowRankMat, k::Int64)
    return size(A)[k]
end

function adjoint(A::SymLowRankMat)
    return A
end

function subSketch(A::SymLowRankMat, order::Int64)
    indices = rand(1:size(A, 2), order)
    return LowRankMat(A.L, A.L[indices, :])
end
;
