function sample_data(m, n, r, d, missing_frac, noise_variance)
    """
    This function samples synthetic data for the low rank matrix learning under
    partial observations problem.

    :param m: The column dimension of the partially observed matrix to be
              sampled (Int64).
    :param n: The row dimension of the partially observed matrix to be
              sampled (Int64).
    :param r: The rank of the partially observed matrix to be sampled (Int64).
    :param d: The dimension of the side information matrix to be
              sampled (Int64).
    :param missing_frac: The fraction of missing entries in the partially
                         observed matrix (Float64).
    :param noise_variance: The standard deviation of gaussian noise added to
                           the side information (Float64).

    :return: This function returns four values. (1) The n-by-m fully observed
             matrix, (2) the n-by-m partially observed matrix, (3) the n-by-d
             matrix of side information and (4) the rank of the partially
             observed matrix.
    """

    unif = Uniform(0, 1)
    gaus = Normal(0, noise_variance)

    U = rand(unif, (n, r))
    V = rand(unif, (m, r))
    beta = rand(unif, (m, d))
    N = rand(gaus, (n, d))

    A_true = U * V'
    Y = A_true * beta + N

    # Randomly remove entries of A_true to form the matrix A_observed
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

function evaluatePerformance(X_fitted, A_observed, A_true, Y, k, lambda, gamma)
    """
    This function evaluates the objective value and the reconstruction error of
    an input solution to the low rank matrix learning under partial observations
    problem (problem (8) in the accompanying paper).

    :param X_fitted: A n-by-m matrix that is a candidate solution.
    :param A_observed: A n-by-m partially observed matrix. Unobserved values
                       should be entered as zero.
    :param A_observed: The n-by-m ground truth complete matrix.
    :param Y: A n-by-d side information matrix.
    :param k: A specified target rank (Int64).
    :param lambda: A parameter to weight the emphasis placed on finding a
                   reconstruction that is predictive of the side information
                   in the objective function (Float64).
    :param gamma: A regularization parameter (Float64).

    :return: This function returns two values. (1) The objective value achieved
             by X_fitted and (2) the reconstruction error of X_fitted.
    """

    (n, m) = size(A_observed)
    S = zeros(n, m)
    for (i, j, value) in zip(findnz(A_observed)...)
        S[i, j] = 1
    end
    S = sparse(S)

    U, _, _ = tsvd(X_fitted, k)
    obj = norm(S .* (X_fitted - A_observed))^2
    obj += lambda * norm((Matrix(I, n, n) - U * U') * Y) ^ 2
    obj += 2 * gamma * sum(abs.(svd(X_fitted).S))

    MSE = norm(X_fitted - A_true)
    return obj, MSE
end
