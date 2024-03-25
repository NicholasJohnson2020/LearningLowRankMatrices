function transformInputMatrix(A; fill_type=NaN)

    # Convert the inpute sparse matrix into a dense matrix where missing entries
    # have type fill_type
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

    A_missing = transformInputMatrix(A, fill_type=missing)
    return Impute(A_missing, k, method=:fastImpute, B=B)
end


function softImpute(A, k)

    A_missing = transformInputMatrix(A, fill_type=NaN)
    return fancyImpute.SoftImpute(max_rank=k).fit_transform(A_missing)
end


function iterativeSVD(A, k)

    A_missing = transformInputMatrix(A, fill_type=NaN)
    return fancyImpute.IterativeSVD(rank=k).fit_transform(A_missing)
end


function matrixFactorization(A, k)

    A_missing = transformInputMatrix(A, fill_type=NaN)
    return fancyImpute.MatrixFactorization(rank=k).fit_transform(A_missing)
end


function biScalerALS(A, k)

    A_missing = transformInputMatrix(A, fill_type=NaN)
    return fancyImpute.BiScaler().fit_transform(A_missing)
end


function nuclearNormMinimization(A, k)

    A_missing = transformInputMatrix(A, fill_type=NaN)
    return fancyImpute.NuclearNormMinimization().fit_transform(A_missing)
end
