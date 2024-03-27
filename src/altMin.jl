function alternatingMinimization(A, k, Y, lambda; max_iteration=1000,
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
    num_steps_hessian_estimate = 3
    grad_norm_termination = 1e-4
    old_objective = 0
    new_objective = 0
    iter_count = 0

    function fgU(x)
        U_mat = reshape(x, n, k)
        obj, gradients = computeObjectiveGradient(U_mat, V_iterate, S, A, Y,
                                                  lambda)
        return obj, reduce(vcat, gradients[1])
    end

    function fgV(x)
        V_mat = reshape(x, m, k)
        obj, gradients = computeObjectiveGradient(U_iterate, V_mat, S, A, Y,
                                                  lambda)
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
