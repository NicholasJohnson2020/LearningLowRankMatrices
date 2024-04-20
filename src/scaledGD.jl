function computeObjectiveGradient(U, V, S, A, Y, lambda, gamma;
                                  singular_value_threshold=1e-4)

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
                  step_size=2/3)

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

    old_objective = 0
    new_objective = 0
    iter_count = 0

    # Main loop
    for iteration=1:max_iteration

        iter_count += 1
        new_objective, gradients = computeObjectiveGradient(U_iterate,
                                                            V_iterate, S,
                                                            A, Y, lambda, gamma)

        U_update = U_iterate - step_size * gradients[1] * pinv(V_iterate' * V_iterate)
        V_update = V_iterate - step_size * gradients[2] * pinv(U_iterate' * U_iterate)

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

    return (U_iterate * V_iterate'), new_objective, iter_count

end
