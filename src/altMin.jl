function alternatingMinimization(A, k, Y, lambda; gamma=0.01, max_iteration=1000,
                                 termination_criteria="rel_improvement",
                                 min_improvement=0.001)
    @assert termination_criteria in ["iteration_count", "rel_improvement"]

    (n, m) = size(A)

    S = zeros(n, m)
    for (i, j, value) in zip(findnz(A)...)
        S[i, j] = 1
    end
    S = sparse(S)

    L, sigma, R = tsvd(A, k)

    U_iterate = L * Diagonal(sqrt.(sigma))
    V_iterate = R * Diagonal(sqrt.(sigma))

    # Set LBFGS parameters
    num_steps_hessian_estimate = 2
    grad_norm_termination = 1e-1
    old_objective = 0
    new_objective = 0
    iter_count = 0

    function fgU(x)
        U_mat = reshape(x, n, k)
        obj, gradients = computeObjectiveGradient(U_mat, V_iterate, S, A, Y,
                                                  lambda, gamma)
        return obj, reduce(vcat, gradients[1])
    end

    function fgV(x)
        V_mat = reshape(x, m, k)
        obj, gradients = computeObjectiveGradient(U_iterate, V_mat, S, A, Y,
                                                  lambda, gamma)
        return obj, reduce(vcat, gradients[2])
    end

     # Main loop
    for iteration=1:max_iteration

        iter_count += 1
        # Update U iterate
        Uvec, fU, _, _, _ = OptimKit.optimize(fgU, reduce(vcat, U_iterate),
                                              LBFGS(num_steps_hessian_estimate,
                                              gradtol=grad_norm_termination));
        U_iterate = reshape(Uvec, n, k)

        # Update V iterate
        Vvec, fV, _, _, _ = OptimKit.optimize(fgV, reduce(vcat, V_iterate),
                                              LBFGS(num_steps_hessian_estimate,
                                              gradtol=grad_norm_termination));
        V_iterate = reshape(Vvec, m, k)

        new_objective = fV
        if (termination_criteria == "rel_improvement") & (old_objective != 0)
            if (old_objective - new_objective) / old_objective < min_improvement
                break
            end
        end
        old_objective = new_objective
    end

    return (U_iterate * V_iterate'), new_objective, iter_count

end


function alternatingMinimizationLifted(A, k, Y, lambda; gamma=0.01,
                                      max_iteration=1000,
                                      termination_criteria="rel_improvement",
                                      min_improvement=0.001)
    @assert termination_criteria in ["iteration_count", "rel_improvement"]

    (n, m) = size(A)

    S = zeros(n, m)
    for (i, j, value) in zip(findnz(A)...)
        S[i, j] = 1
    end
    S = sparse(S)

    L, sigma, R = tsvd(A, k)

    U_iterate = L * Diagonal(sqrt.(sigma))
    V_iterate = R * Diagonal(sqrt.(sigma))
    X_temp = U_iterate * V_iterate'
    alpha_iterate = pinv(X_temp' * X_temp) * X_temp' * Y

    # Set LBFGS parameters
    num_steps_hessian_estimate = 2
    grad_norm_termination = 0.1
    old_objective = 0
    new_objective = 0
    iter_count = 0

    function fgV(x)
        V_mat = reshape(x, m, k)

        X_iterate = U_iterate * V_mat'
        term_1 = S .* (X_iterate - A)
        term_2 = Y - X_iterate * alpha_iterate

        obj = norm(term_1)^2 + lambda * norm(term_2)^2
        obj += gamma * (norm(U_iterate)^2 + norm(V_iterate)^2)

        grad = 2 * (term_1 - lambda * term_2 * alpha_iterate')' * U_iterate
        grad += 2 * gamma * V_iterate

        return obj, reduce(vcat, grad)
    end

     # Main loop
    for iteration=1:max_iteration

        iter_count += 1
        # Update U iterate
        for i=1:n
            inv_mat = V_iterate' * (Diagonal(S[i, :]) + lambda * alpha_iterate * alpha_iterate')
            inv_mat = inv(inv_mat * V_iterate + gamma * Matrix(I, k, k))
            U_iterate[i, :] = inv_mat * V_iterate' * (A[i, :] + lambda * alpha_iterate * Y[i, :])
        end

        # Update V iterate
        Vvec, fV, _, _, _ = OptimKit.optimize(fgV, reduce(vcat, V_iterate),
                                              LBFGS(num_steps_hessian_estimate,
                                              gradtol=grad_norm_termination));
        V_iterate = reshape(Vvec, m, k)

        # Update Alpha iterate
        X_temp = U_iterate * V_iterate'
        alpha_iterate = pinv(X_temp' * X_temp) * X_temp' * Y

        new_objective, _ = fgV(reduce(vcat, V_iterate))

        if (termination_criteria == "rel_improvement") & (old_objective != 0)
            if (old_objective - new_objective) / old_objective < min_improvement
                break
            end
        end
        old_objective = new_objective
    end

    return (U_iterate * V_iterate'), new_objective, iter_count

end
