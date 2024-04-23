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
                  step_size=1)

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

function vanillaGD(A, k, Y, lambda; gamma=0.01, max_iteration=1000,
                  termination_criteria="rel_improvement", min_improvement=0.001,
                  step_size=1)

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

        # Update the U and V iterates
        U_iterate = U_iterate - step_size / step_normalization * gradients[1]
        V_iterate = V_iterate - step_size / step_normalization * gradients[2]

        if (termination_criteria == "rel_improvement") & (old_objective != 0)
            if (old_objective - new_objective) / old_objective < min_improvement
                break
            end
        end
        old_objective = new_objective
    end

    return U_iterate, V_iterate, new_objective, iter_count

end

function cross_validate(method, A, k, Y, lambda, gamma; num_samples=10,
                        train_frac=0.7, candidate_vals=[10, 1, 0.1, 0.01],
                        singular_value_threshold=1e-4)

    (n, m) = size(A)

    alpha = (1-sqrt(train_frac))
    val_n = Int(floor(n * alpha))
    train_n = n - val_n
    val_m = Int(floor(m * alpha))
    train_m = m - val_m

    param_scores = Dict()
    for step_size in candidate_vals
        param_scores[step_size] = 0
    end

    for trial=1:num_samples

        row_permutation = randperm(n)
        col_permutation = randperm(m)

        val_row_ind = row_permutation[1:val_n]
        train_row_ind = row_permutation[(val_n+1):end]
        val_col_ind = col_permutation[1:val_m]
        train_col_ind = col_permutation[(val_m+1):end]

        A_val = A[val_row_ind, val_col_ind]
        A_train = A[train_row_ind, train_col_ind]
        Y_train = Y[train_row_ind, :]
        LL_block_data = A[train_row_ind, val_col_ind]
        UR_block_data = A[val_row_ind, train_col_ind]

        while norm(A_val) ^ 2 < 1e-4
            row_permutation = randperm(n)
            col_permutation = randperm(m)

            val_row_ind = row_permutation[1:val_n]
            train_row_ind = row_permutation[(val_n+1):end]
            val_col_ind = col_permutation[1:val_m]
            train_col_ind = col_permutation[(val_m+1):end]

            A_val = A[val_row_ind, val_col_ind]
            A_train = A[train_row_ind, train_col_ind]
            Y_train = Y[train_row_ind, :]
            LL_block_data = A[train_row_ind, val_col_ind]
            UR_block_data = A[val_row_ind, train_col_ind]
        end

        for step_size in candidate_vals

            output = method(A_train, k, Y_train, lambda, gamma=gamma,
                            step_size=step_size)
            X_fitted = output[1] * output[2]'
            L, Sigma, R = tsvd(X_fitted, k)
            for i=1:k
                if Sigma[i] > singular_value_threshold
                    Sigma[i] = 1 / Sigma[i]
                else
                    Sigma[i] = 0
                end
            end
            X_inv = L * Diagonal(Sigma) * R'
            val_estimate = UR_block_data * X_inv' * LL_block_data
            val_error = norm(val_estimate - A_val) ^2 / norm(A_val) ^ 2

            param_scores[step_size] += val_error / num_samples
        end
    end

    best_score = 1e9
    best_param = ()
    for (param, score) in param_scores
        if score < best_score
            best_score = score
            best_param = param
        end
    end

    return best_param, param_scores

end;
