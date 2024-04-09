function admm(A, k, Y, lambda; gamma=0.01, rho_1=10, rho_2=10,
              residual_threshold=0.01, max_iteration=100)

    (n, m) = size(A)
    d = size(Y)[2]
    S = zeros(n, m)
    for (i, j, value) in zip(findnz(A)...)
      S[i, j] = 1
    end
    S = sparse(S)

    L, sigma, R = tsvd(A, k)

    U_iterate = L * Diagonal(sqrt.(sigma))
    V_iterate = R * Diagonal(sqrt.(sigma))
    P_iterate = L * L'
    Z_iterate = U_iterate

    sigma1_iterate = ones(n, k)
    sigma2_iterate = ones(n, k)

    sigma1_residual_hist = []
    sigma2_residual_hist = []

    cache_Y = lambda * Y * Y'

    for iteration=1:max_iteration

        # Perform V Update
        for j=1:m
            inv_mat = 2 * U_iterate' * Diagonal(S[:, j]) * U_iterate
            inv_mat += 2 * gamma * Matrix(I, k, k)
            inv_mat = inv(inv_mat)
            V_iterate[j, :] = inv_mat * (2 * U_iterate' * A[:, j])
        end

        # Perform Z Update
        temp = rho_2 * U_iterate - sigma2_iterate
        temp -= (Matrix(I, n, n) - P_iterate) * sigma1_iterate
        Z_iterate = (Matrix(I, n, n) - (rho_1 / rho_2) * P_iterate) * temp
        Z_iterate = Z_iterate / (rho_1 + rho_2)

        # Perform U Update
        for i=1:n
            inv_mat = 2 * V_iterate' * Diagonal(S[i, :]) * V_iterate
            inv_mat += (2 * gamma + rho_2) * Matrix(I, k, k)
            inv_mat = inv(inv_mat)
            temp_vec = sigma2_iterate[i, :] + rho_2 * Z_iterate[i, :]
            U_iterate[i, :] = inv_mat * (2 * V_iterate' * A[i, :] + temp_vec)
        end

        # Perform P Update
        temp = sigma1_iterate * Z_iterate' / 2
        temp += temp'
        temp += cache_Y + (rho_1 / 2) * Z_iterate * Z_iterate'
        L, _, _ = tsvd(A, k)
        P_iterate = L * L'


        # Perform Sigma1 Update
        sigma1_residual = (Matrix(I, n, n) - P_iterate) * Z_iterate
        sigma1_iterate += rho_1 * sigma1_residual

        # Perform Sigma2 Update
        sigma2_residual = Z_iterate - U_iterate
        sigma2_iterate += rho_2 * sigma2_residual

        # Check for termination
        if max(norm(sigma1_residual)^2, norm(sigma2_residual)^2) < residual_threshold
            break
        end

        V_residual = 2 * (S' * (V_iterate * U_iterate' - A)) * U_iterate + 2 * gamma * V_iterate
        Z_residual = rho_1 * (Matrix(I, n, n) - P_iterate) * Z_iterate + sigma2_iterate
        Z_residual += 

        append!(sigma1_residual_hist, norm(sigma1_residual)^2)
        append!(sigma2_residual_hist, norm(sigma2_residual)^2)

    end

    return U_iterate, V_iterate, P_iterate, Z_iterate, sigma1_iterate,
           sigma2_iterate, (sigma1_residual_hist, sigma2_residual_hist)

end
;
