function alternatingMinimization(A, k, Y, lambda; gamma=0.01, max_iteration=1000,
                                 termination_criteria="rel_improvement",
                                 min_improvement=0.001, algorithm="LBFGS")
    @assert termination_criteria in ["iteration_count", "rel_improvement"]
    @assert algorithm in ["LBFGS", "CG", "GD"]

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
        if algorithm == "LBFGS"
            Uvec, fU, _, _, _ = OptimKit.optimize(fgU, reduce(vcat, U_iterate),
                                                  LBFGS(num_steps_hessian_estimate,
                                                  gradtol=grad_norm_termination))
        elseif algorithm == "CG"
            Uvec, fU, _, _, _ = OptimKit.optimize(fgU, reduce(vcat, U_iterate),
                                                  ConjugateGradient(gradtol=grad_norm_termination))
        else
            Uvec, fU, _, _, _ = OptimKit.optimize(fgU, reduce(vcat, U_iterate),
                                                  GradientDescent(gradtol=grad_norm_termination))
        end

        U_iterate = reshape(Uvec, n, k)

        # Update V iterate
        if algorithm == "LBFGS"
            Vvec, fV, _, _, _ = OptimKit.optimize(fgV, reduce(vcat, V_iterate),
                                                  LBFGS(num_steps_hessian_estimate,
                                                  gradtol=grad_norm_termination))
        elseif algorithm == "CG"
            Vvec, fV, _, _, _ = OptimKit.optimize(fgV, reduce(vcat, V_iterate),
                                                  ConjugateGradient(gradtol=grad_norm_termination))
        else
            Vvec, fV, _, _, _ = OptimKit.optimize(fgV, reduce(vcat, V_iterate),
                                                  GradientDescent(gradtol=grad_norm_termination))
        end

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
                                      min_improvement=0.001,
                                      singular_value_threshold=1e-4,
                                      algorithm="NAGD",
                                      step_size=1)
    @assert termination_criteria in ["iteration_count", "rel_improvement"]
    @assert algorithm in ["LBFGS", "NAGD"]

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

    alpha_norm = 0
    for i=1:k
        if sigma[i] > singular_value_threshold
            sigma[i] = 1 / sigma[i]
            if sigma[i] > alpha_norm
                alpha_norm = sigma[i]
            end
        else
            sigma[i] = 0
        end
    end

    alpha_iterate = R * Diagonal(sigma) * L' * Y

    # Set LBFGS parameters
    num_steps_hessian_estimate = 2
    grad_norm_termination = 1e-1
    old_objective = 0
    new_objective = 0
    iter_count = 0

    Y_norm = tsvd(Y, 1)[2][1]

    function fgV_LBFGS(x)
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

    function fgV_NAGD(x)
        V_mat = x
        X_iterate = U_iterate * V_mat'
        term_1 = S .* (X_iterate - A)
        term_2 = Y - X_iterate * alpha_iterate

        obj = norm(term_1)^2 + lambda * norm(term_2)^2
        obj += gamma * (norm(U_iterate)^2 + norm(V_iterate)^2)

        grad = 2 * (term_1 - lambda * term_2 * alpha_iterate')' * U_iterate
        grad += 2 * gamma * V_iterate

        return obj, grad
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
        if algorithm == "LBFGS"
            Vvec, fV, _, _, _ = OptimKit.optimize(fgV_LBFGS, reduce(vcat, V_iterate),
                                                  LBFGS(num_steps_hessian_estimate,
                                                  gradtol=grad_norm_termination))
            V_iterate = reshape(Vvec, m, k)
        else
            L = tsvd(U_iterate' * U_iterate, 1)[2][1]
            L = L * (1 + lambda * alpha_norm ^ 2 * Y_norm ^ 2)
            L = 2 * (L + gamma)
            V_iterate = NesterovGD(V_iterate, L / step_size, fgV_NAGD,
                                   grad_norm_termination=grad_norm_termination)
        end

        # Update Alpha iterate
        X_temp = U_iterate * V_iterate'
        L, sigma, R = tsvd(X_temp, k)
        alpha_norm = 0
        for i=1:k
            if sigma[i] > singular_value_threshold
                sigma[i] = 1 / sigma[i]
                if sigma[i] > alpha_norm
                    alpha_norm = sigma[i]
                end
            else
                sigma[i] = 0
            end
        end
        alpha_iterate = R * Diagonal(sigma) * L' * Y

        new_objective, _ = fgV_LBFGS(reduce(vcat, V_iterate))

        if (termination_criteria == "rel_improvement") & (old_objective != 0)
            if (old_objective - new_objective) / old_objective < min_improvement
                break
            end
        end
        old_objective = new_objective
    end

    return (U_iterate * V_iterate'), new_objective, iter_count

end

function NesterovGD(x0, L, objGradFunction; max_iteration=50,
                    grad_norm_termination=1e-1)

    x = x0
    y = x
    lambda_this = 1
    lambda_next = (1 + sqrt(5)) / 2
    gamma = 0

    for iteration=1:max_iteration

        this_obj, this_grad = objGradFunction(x)

        if norm(this_grad) ^ 2 < grad_norm_termination
            break
        end

        y_next = x - this_grad / L
        x = (1 - gamma) * y_next + gamma * y

        lambda_this = lambda_next
        lambda_next = (1 + sqrt(1 + 4 * lambda_this ^ 2)) / 2
        gamma = (1 - lambda_this) / lambda_next
        y = y_next
    end

    return x

end
