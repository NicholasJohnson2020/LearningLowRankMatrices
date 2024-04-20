function sample_data(m, n, r, d, missing_frac, noise_variance)

    unif = Uniform(0, 1)
    gaus = Normal(0, noise_variance)

    U = rand(unif, (n, r))
    V = rand(unif, (m, r))
    beta = rand(unif, (m, d))
    N = rand(gaus, (n, d))

    A_true = U * V'
    Y = A_true * beta + N

    num_missing = Int(floor(missing_frac * n * m))
    indices = [(i, j) for i=1:n, j=1:m]
    shuffle!(indices)
    missing_indices = indices[1:num_missing]

    A_observed = copy(A_true)
    for (i, j) in missing_indices
        A_observed[i, j] = 0
    end
    A_observed = sparse(A_observed)

    return A_true, A_observed, Y, r

end

function evaluatePerformance(X_fitted, A_observed, A_true, Y, k, lambda, gamma;
                             singular_value_threshold=1e-4)

    (n, m) = size(A_observed)
    S = zeros(n, m)
    for (i, j, value) in zip(findnz(A_observed)...)
        S[i, j] = 1
    end
    S = sparse(S)

    U, _, _ = tsvd(X_fitted, k)
    obj = norm(S .* (X_fitted - A_observed))^2
    obj += lambda * tr(Y' * (Matrix(I, n, n) - U * U') * Y)
    obj += 2 * gamma * sum(abs.(svd(X_fitted).S))

    MSE = norm(X_fitted - A_true)
    return obj, MSE
end
