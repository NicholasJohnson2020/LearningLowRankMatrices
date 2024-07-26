function admm(A, k, Y, lambda; gamma=0.01, step_size=10,
              residual_threshold=0.01, max_iteration=100,
              initialization="approximate", P_update="exact")
    """
    This function computes a high quality feasible solution to the low rank
    matrix learning under partial observations (problem (8) in the accompanying
    paper) by executing a mixed-projection alternating direction method of
    multipliers algorithm.

    :param A: A n-by-m partially observed matrix. Unobserved values should be
              entered as zero.
    :param k: A specified target rank (Int64).
    :param Y: A n-by-d side information matrix.
    :param lambda: A parameter to weight the emphasis placed on finding a
                   reconstruction that is predictive of the side information
                   in the objective function (Float64).
    :param gamma: A regularization parameter (Float64).
    :param step_size: The step size parameter (Float64).
    :param residual_threshold: Threshold below which primal residuals are taken
                               to be zero (Float64).
    :param max_iteration: The maximum number of ADMM iterations to execute
                          prior to terminating (Int64).
    :param initialization: If "exact", compute an exact truncated SVD when
                           initializing the algorithm. If "approximate",
                           compute a randomized truncated SVD to initialize the
                           algorithm.
    :param P_update: If "exact", compute an exact truncated SVD when
                     solving the P subproblem. If "approximate",
                     compute a randomized truncated SVD when performing the P
                     update.

    :return: This function returns six values. (1) The final n-by-k matrix U,
             (2) the final m-by-k matrix V, (3) the final n-by-n matrix P,
             (4) the final n-by-k matrix Z, (5) the final dual variable Phi and
             (6) the final dual variable Psi.
    """

    # Verify that initialization and P_update take allowable values
    @assert initialization in ["exact", "approximate"]
    @assert P_update in ["exact", "sub_sketch"]

    opts = LRAOptions()
    opts.sketch = :sub

    # We fix rho_1 and rho_2 to take value step_size
    rho_1 = step_size
    rho_2 = step_size

    (n, m) = size(A)
    d = size(Y)[2]

    # Verify that the side information has the same number of rows as the
    # partially observed matrix.
    @assert size(Y)[1] == n

    # Initialize the sparse binary matrix S
    iIndex, jIndex, _ = findnz(A)
    S = sparse(iIndex, jIndex, ones(Int8, length(iIndex)), n, m)

    # Initialize the primal variables
    if initialization == "exact"
        L, sigma, R = tsvd(A, k)
    else
        L, sigma, R = psvd(A, rank=k, opts)
    end

    U_iterate = L * Diagonal(sqrt.(sigma))
    V_iterate = R * Diagonal(sqrt.(sigma))
    Z_iterate = U_iterate
    P_iterate = SymLowRankMat(L)

    # Initialize the dual variables
    Phi_iterate = ones(Float64, n, k)
    Psi_iterate = ones(Float64, n, k)

    # Main algorithm loop
    for iteration=1:max_iteration

        # Perform the U Update
        Threads.@threads for i=1:n
            inv_mat = 2 * V_iterate' * Diagonal(S[i, :]) * V_iterate
            inv_mat += (2 * gamma + rho_2) * Matrix(I, k, k)
            temp_vec = Psi_iterate[i, :] + rho_2 * Z_iterate[i, :]
            U_iterate[i, :] = inv_mat \ (2 * V_iterate' * A[i, :] + temp_vec)
        end

        # Perform the P Update
        factor_1 = [lambda * Y rho_1 / 2 * Z_iterate Phi_iterate / 2 Z_iterate / 2]
        factor_2 = [Y Z_iterate Z_iterate Phi_iterate]
        temp = LowRankMat(factor_1, factor_2)
        if P_update == "exact"
            L, _, _ = tsvd(temp, k)
        else
            L, _, _ = tsvd(subSketch(temp, k), k)
        end
        P_iterate = SymLowRankMat(L)

        # Perform the V Update
        Threads.@threads for j=1:m
            inv_mat = 2 * U_iterate' * Diagonal(S[:, j]) * U_iterate
            inv_mat += 2 * gamma * Matrix(I, k, k)
            V_iterate[j, :] = inv_mat \ (2 * U_iterate' * A[:, j])
        end

        # Perform the Z Update
        temp = Phi_iterate + rho_1 * U_iterate - (rho_1 / rho_2) * Psi_iterate
        temp = P_iterate * temp
        Z_iterate = (rho_2 * U_iterate - Psi_iterate - Phi_iterate + temp) / (rho_1 + rho_2)

        # Perform the Phi Update
        Phi_residual = Z_iterate - P_iterate * Z_iterate
        Phi_iterate += rho_1 * Phi_residual

        # Perform the Psi Update
        Psi_residual = Z_iterate - U_iterate
        Psi_iterate += rho_2 * Psi_residual

        # Check if the termination criteria has been achieved
        if max(norm(Phi_residual)^2, norm(Psi_residual)^2) < residual_threshold
            break
        end

    end

    return U_iterate, V_iterate, P_iterate, Z_iterate, Phi_iterate,
           Psi_iterate

end
;
