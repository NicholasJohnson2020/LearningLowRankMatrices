import Base: eltype, size, adjoint, *

struct LowRankMat
    """
    This is a custom matrix class that represents an arbitrary matrix as the
    product of two factors.
    """

    U::Matrix{Float64}
    V::Matrix{Float64}
end

# Define multiplication between a vector and a matrix of type LowRankMat
*(A::LowRankMat, x::Vector{Float64}) = A.U * (A.V' * x)

# Define multiplication between a matrix and a matrix of type LowRankMat
*(A::LowRankMat, X::Matrix{Float64}) = A.U * (A.V' * X)

# Define multiplication between an adjoint matrix and a matrix of type
# LowRankMat
*(X::Adjoint{Float64, Matrix{Float64}}, A::LowRankMat) = (X * A.U) * A.V'

function eltype(A::LowRankMat)
    """
    This function returns the element type of a matrix of type LowRankMat.

    :param A: A n-by-m matrix of type LowRankMat.

    :return: The type of elements in A.
    """

    return eltype(A.U)
end

function size(A::LowRankMat)
    """
    This function returns the size of a matrix of type LowRankMat.

    :param A: A n-by-m matrix of type LowRankMat.

    :return: The size of A (tuple of Int64).
    """

    return (size(A.U)[1], size(A.V)[1])
end

function size(A::LowRankMat, k::Int64)
    """
    This function returns the size of dimension k of a matrix of type
    LowRankMat.

    :param A: A n-by-m matrix of type LowRankMat.
    :param k: The dimension of A for which the size is desired to be known
              (Int64).

    :return: The size of A along dimension k (Int64).
    """

    return size(A)[k]
end

function adjoint(A::LowRankMat)
    """
    This function returns the adjoint a matrix of type LowRankMat.

    :param A: A n-by-m matrix of type LowRankMat.

    :return: The adjoint of A (LowRankMat).
    """

    return LowRankMat(A.V, A.U)
end

function subSketch(A::LowRankMat, order::Int64)
    """
    This function returns a sketch of a matrix of type LowRankMat by
    subsampling order columns.

    :param A: A n-by-m matrix of type LowRankMat.
    :param order: The number of columns of A to subsample in the sketch (Int64).

    :return: A n-by-order sketch of A (LowRankMat).
    """

    indices = rand(1:size(A, 2), order)
    return LowRankMat(A.U, A.V[indices, :])
end

struct SymLowRankMat
    """
    This is a custom matrix class that represents an arbitrary semidefinite
    matrix X by storing a matrix L such that L * L.T = X.
    """
    L::Matrix{Float64}
end

# Define multiplication between a vector and a matrix of type SymLowRankMat
*(A::SymLowRankMat, x::Vector{Float64}) = A.L * (A.L' * x)

# Define multiplication between a matrix and a matrix of type SymLowRankMat
*(A::SymLowRankMat, X::Matrix{Float64}) = A.L * (A.L' * X)

# Define multiplication between an adjoint matrix and a matrix of type
# SymLowRankMat
*(X::Adjoint{Float64, Matrix{Float64}}, A::SymLowRankMat) = (X * A.L) * A.L'

function eltype(A::SymLowRankMat)
    """
    This function returns the element type of a matrix of type SymLowRankMat.

    :param A: A n-by-n matrix of type SymLowRankMat.

    :return: The type of elements in A.
    """

    return eltype(A.L)
end

function size(A::SymLowRankMat)
    """
    This function returns the size of a matrix of type SymLowRankMat.

    :param A: A n-by-n matrix of type SymLowRankMat.

    :return: The size of A (tuple of Int64).
    """

    return (size(A.L)[1], size(A.L)[1])
end

function size(A::SymLowRankMat, k::Int64)
    """
    This function returns the size of dimension k of a matrix of type
    SymLowRankMat.

    :param A: A n-by-n matrix of type SymLowRankMat.
    :param k: The dimension of A for which the size is desired to be known
              (Int64).

    :return: The size of A along dimension k (Int64).
    """

    return size(A)[k]
end

function adjoint(A::SymLowRankMat)
    """
    This function returns the adjoint a matrix of type SymLowRankMat.

    :param A: A n-by-n matrix of type SymLowRankMat.

    :return: The adjoint of A (SymLowRankMat).
    """

    return A
end

function subSketch(A::SymLowRankMat, order::Int64)
    """
    This function returns a sketch of a matrix of type SymLowRankMat by
    subsampling order columns.

    :param A: A n-by-n matrix of type SymLowRankMat.
    :param order: The number of columns of A to subsample in the sketch (Int64).

    :return: A n-by-order sketch of A (LowRankMat).
    """

    indices = rand(1:size(A, 2), order)
    return LowRankMat(A.L, A.L[indices, :])
end
;
