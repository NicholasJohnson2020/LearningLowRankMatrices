function computeObjectiveGradient(U, V, S, A, Y, lambda, gamma;
                                  singular_value_threshold=1e-6)
    """
    This function evaluates the predictive low rank matrix learning under
    partial observations problem objective and evaluates its partial gradients
    when alpha is minimized out and the low-rank matrix is factorized as
    X = U * V^T.

    :param U: An n-by-k matrix corresponding to the U factor of X.
    :param V: An m-by-k matrix corresponding to the V factor of X.
    :param S: An n-by-m binary matrix corresponding to the pattern of revealed
              entries in the input matrix A.
    :param A: A n-by-m partially observed matrix (unobserved values should be
              entered as zero).
    :param Y: A n-by-d side information matrix.
    :param lambda: A parameter to weight the emphasis placed on finding a
                   reconstruction that is predictive of the side information
                   in the objective function (Float64).
    :param gamma: A regularization parameter (Float64).
    :param singular_value_threshold: A threshold parameter below which singular
                                     values are taken to be zero when computing
                                     a matrix inverse (Float64).

    :return: This function outputs two values. (1) The objective value. (2) A
             tuple of length 2 where the first component is the partial gradient
             with respect to U and the second component is the partial with
             respect to V.
    """

    X = U * V'
    (n, m) = size(X)
    k = size(U)[2]

    B, Sigma, C = tsvd(X, k)
    for i=1:k
        if Sigma[i] > singular_value_threshold
            Sigma[i] = 1 / Sigma[i]
        else
            Sigma[i] = 0
        end
    end
    temp_1 = S .* (X - A)

    G = temp_1
    G += lambda *  ((B * B' - 1 * Matrix(I, n, n)) * Y * Y' * B * Diagonal(Sigma) * C')

    U_grad = G * V + 2 * gamma * U
    V_grad = G' * U + 2 * gamma * V

    obj = norm(temp_1) ^ 2
    obj += lambda * tr(Y' * (Matrix(I, n, n) - B * B') * Y)
    obj += gamma * (norm(U) ^ 2 + norm(V) ^ 2)

    return obj, (U_grad, V_grad)

end

function scaledGD(A, k, Y, lambda; gamma=0.01, max_iteration=1000,
                  termination_criteria="rel_improvement", min_improvement=0.001,
                  step_size=0.1)
    """
    This function computes a feasible solution to the predictive low rank
    matrix learning under partial observations problem by employing the
    ScaledGD algorithm as described in "Accelerating Ill-Conditioned Low-Rank
    Matrix Estimation via Scaled Gradient Descent" (Tong et al. 2021).

    :param A: A n-by-m partially observed matrix. Unobserved values should be
              entered as zero.
    :param k: A specified target rank (Int64).
    :param Y: A n-by-d side information matrix.
    :param lambda: A parameter to weight the emphasis placed on finding a
                   reconstruction that is predictive of the side information
                   in the objective function (Float64).
    :param gamma: A regularization parameter (Float64).
    :param max_iteration: The number of iterations of the main optimization
                          loop to execute (Int64).
    :param termination_criteria: String that must take value either
                                 "rel_improvement" or "iteration_count". If
                                 "rel_improvement", the algorithm will terminate
                                 if the fractional decrease in the objective
                                 value after an iteration is less than
                                 min_improvement. If set to "iteration_count"
                                 the algorithm will terminate after
                                 max_iteration steps (String).
    :param min_improvement: The minimal fractional decrease in the objective
                            value required for the procedure to continue
                            iterating when termination_criteria is set to
                            "rel_improvement".
    :param step_size: The step_size parameter to use in the algorithm (Float64).

    :return: This function returns four values. (1) The n-by-k final matrix U,
             (2) the m-by-k final matrix V, (3) the objective value achieved by
             the return solution and (4) the number of iterations executed when
             performing ScaledGD.
    """

    @assert termination_criteria in ["iteration_count", "rel_improvement"]

    (n, m) = size(A)

    S = zeros(n, m)
    for (i, j, value) in zip(findnz(A)...)
        S[i, j] = 1
    end
    S = sparse(S)

    L, sigma, R = tsvd(A, k)
    step_normalization = sigma[1]

    U_iterate = L * Diagonal(sqrt.(sigma))
    V_iterate = R * Diagonal(sqrt.(sigma))

    old_objective = 0
    new_objective = 0
    iter_count = 0

    # Main loop
    for iteration=1:max_iteration

        iter_count += 1
        new_objective, gradients = computeObjectiveGradient(U_iterate,
                                                            V_iterate, S,
                                                            A, Y, lambda, gamma)

        U_update = U_iterate - step_size / step_normalization * gradients[1] * pinv(V_iterate' * V_iterate)
        V_update = V_iterate - step_size / step_normalization * gradients[2] * pinv(U_iterate' * U_iterate)

        # Update the U and V iterates
        U_iterate = U_update
        V_iterate = V_update

        if (termination_criteria == "rel_improvement") & (old_objective != 0)
            if (old_objective - new_objective) / old_objective < min_improvement
                break
            end
        end
        old_objective = new_objective
    end

    return U_iterate, V_iterate, new_objective, iter_count

end

;
