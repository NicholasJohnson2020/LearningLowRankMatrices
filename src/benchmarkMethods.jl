function transformInputMatrix(A; fill_type=NaN)
    """
    This function converts an input sparse matrix intro a dense matrix where
    missing values have type fill_type.

    :param A: A n-by-m sparse (partially observed) matrix. Unobserved values
              should be entered as zero.
    :param fill_type: The type that missing entries should be converted to in
                      the output matrix.

    :return: The converted n-by-m matrix.
    """

    (n, m) = size(A)
    A_missing = convert(Array{Any,2}, Matrix(A))
    for i=1:n, j=1:m
        if A_missing[i, j] == 0
            A_missing[i, j] = fill_type
        end
    end

    return A_missing
end


function fastImpute(A, k; B=nothing)
    """
    This is a wrapper that calls the Fast-Impute function from MatrixImpute.jl
    (algorithm designed by Bertsimas and Li).

    :param A: A n-by-m sparse (partially observed) matrix. Unobserved values
              should be entered as zero.
    :param k: The target rank of the reconstructed matrix.
    :param B: An optional matrix of side information as defined by Bertsimas
              and Li (2020).

    :return: The reconstructed n-by-m matrix.
    """

    # Cast the type of unobserved entries in the input matrix to missing.
    A_missing = transformInputMatrix(A, fill_type=missing)

    # Call the Fast-Impute algorithm
    return Impute(A_missing, k, method=:fastImpute, B=B)
end


function softImpute(A, k)
    """
    This is a wrapper that calls the Soft-Impute function from the python
    package fancyImpute (algorithm designed by Mazumder et al.).

    :param A: A n-by-m sparse (partially observed) matrix. Unobserved values
              should be entered as zero.
    :param k: The target rank of the reconstructed matrix.

    :return: The reconstructed n-by-m matrix.
    """

    # Cast the type of unobserved entries in the input matrix to NaN.
    A_missing = transformInputMatrix(A, fill_type=NaN)

    # Call the Soft-Impute algorithm
    return fancyImpute.SoftImpute(max_rank=k).fit_transform(A_missing)
end


function iterativeSVD(A, k)
    """
    This is a wrapper that calls the Iterative-SVD function from the python
    package fancyImpute (algorithm designed by Troyanskaya et al.).

    :param A: A n-by-m sparse (partially observed) matrix. Unobserved values
              should be entered as zero.
    :param k: The target rank of the reconstructed matrix.

    :return: The reconstructed n-by-m matrix.
    """

    # Cast the type of unobserved entries in the input matrix to NaN.
    A_missing = transformInputMatrix(A, fill_type=NaN)

    # Call the Iterative-SVD algorithm
    return fancyImpute.IterativeSVD(rank=k).fit_transform(A_missing)
end
