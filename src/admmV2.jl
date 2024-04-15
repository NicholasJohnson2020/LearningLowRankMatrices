function admmSmall(A, k, Y, lambda; gamma=0.01, sigma=10,
                   residual_threshold=0.01, max_iteration=100, n_inner=1)

    function computeAugL()

        X = X_iterate
        P = P_iterate
        Phi = Phi_iterate

        obj = norm(S .* (X - A)) ^ 2
        obj += lambda * tr(Y' * (Matrix(I, n, n) - P) * Y)
        obj += gamma * norm(X) ^ 2
        obj += tr(Phi' * (Matrix(I, n, n) - P) * X)
        obj += (sigma / 2) * norm((Matrix(I, n, n) - P) * X) ^ 2

        return obj
    end

    (n, m) = size(A)
    d = size(Y)[2]
    S = zeros(n, m)
    for (i, j, value) in zip(findnz(A)...)
      S[i, j] = 1
    end
    S = sparse(S)

    L, svals, R = tsvd(A, k)

    X_iterate = L * Diagonal(svals) * R'
    M_iterate = L
    P_iterate = L * L'

    Phi_iterate = ones(n, m)

    Phi_residual_hist = []
    augL_hist = [computeAugL()]

    cache_Y = lambda * Y * Y'

    for iteration=1:max_iteration

        for inner_iter=1:n_inner

            # Perform X Update
            temp_mat = (Matrix(I, n, n) - P_iterate) * Phi_iterate
            for j=1:m
                diag_vec = 2 * S[:, j] .+ (2 * gamma + sigma)
                D = Diagonal(diag_vec)
                D_inv = Diagonal( 1 ./ diag_vec)
                inv_mat = inv(M_iterate' * D_inv * M_iterate - Matrix(I, k, k) / sigma)
                inv_mat = D_inv - D_inv * M_iterate * inv_mat * M_iterate' * D_inv
                X_iterate[:, j] = inv_mat * (2 * A[:, j] - temp_mat[:, j])
            end

            append!(augL_hist, computeAugL())

            # Perform P Update
            temp = Phi_iterate * X_iterate' / 2
            temp += temp'
            temp += (cache_Y + (sigma / 2) * X_iterate * X_iterate')
            L, _, _ = tsvd(temp, k)
            M_iterate = L
            P_iterate = L * L'

            append!(augL_hist, computeAugL())

        end

        # Perform Phi Update
        Phi_residual = (Matrix(I, n, n) - P_iterate) * X_iterate
        Phi_iterate += sigma * Phi_residual

        append!(augL_hist, computeAugL())
        append!(Phi_residual_hist, norm(Phi_residual)^2)

        # Check for termination
        if norm(Phi_residual)^2 < residual_threshold
            break
        end

    end

    return X_iterate, P_iterate, Phi_iterate,
           (Phi_residual_hist, augL_hist)

end
;
