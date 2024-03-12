function computeObjectiveGradient(U, V, S, A, Y, c_1, c_2, lambda)

    X = U * V'
    (n, m) = size(X)

    temp_1 = S .* (X - A)
    temp_2 = (X' * X) \ X'

    G = c_1 * temp_1
    G += c_2 * ((X * temp_2 - 1 * Matrix(I, n, n)) * Y * Y' * temp_2')
    G += lambda * X

    U_grad = G * V
    V_grad = G' * U

    obj = c_1 * norm(temp_1) ^ 2
    obj += c_2 * tr(Y' * (1 * Matrix(I, n, n) - X * temp_2) * Y)
    obj += lambda * norm(X) ^ 2

    return obj, (U_grad, V_grad)

end

function scaledGD(A, k, Y, c_1, c_2, lambda; max_iteration=1000,
                  termination_criteria="rel_improvement", min_improvement=0.001)

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

    # Default step size specified in the paper
    eta = 2 / 3
    old_objective = 0
    new_objective = 0

    # Main loop
    for iteration=1:max_iteration

        new_objective, gradients = computeObjectiveGradient(U_iterate,
                                                            V_iterate, S,
                                                            A, Y, c_1, c_2,
                                                            lambda)

        U_update = U_iterate - eta * gradients[1] * pinv(V_iterate' * V_iterate)
        V_update = V_iterate - eta * gradients[2] * pinv(U_iterate' * U_iterate)

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

    return (U_iterate * V_iterate')

end
