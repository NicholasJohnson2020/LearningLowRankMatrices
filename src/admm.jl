function admmMap(A, k, Y, lambda; gamma=0.01, rho_1=10, rho_2=10,
              residual_threshold=0.01, max_iteration=100)

    function computeAugL()

        U = U_iterate
        V = V_iterate
        P = P_iterate
        Z = Z_iterate
        Phi = Phi_iterate
        Psi = Psi_iterate

        obj = norm(S .* (U * V' - A)) ^ 2
        obj += lambda * tr(Y' * (Matrix(I, n, n) - P) * Y)
        obj += gamma * (norm(U) ^ 2 + norm(V) ^ 2)
        obj += tr(Phi' * (Matrix(I, n, n) - P) * Z)
        obj += tr(Psi' * (Z - U))
        obj += rho_1 / 2 * norm((Matrix(I, n, n) - P) * Z) ^ 2
        obj += rho_2 / 2 * norm(Z - U) ^ 2

        return obj
    end

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

    Phi_iterate = ones(n, k)
    Psi_iterate = ones(n, k)

    Phi_residual_hist = []
    Psi_residual_hist = []
    augL_hist = [computeAugL()]

    cache_Y = lambda * Y * Y'

    update_time = Dict("U_map" => 0, "U_reduce" => 0, "P" => 0, "V_map" => 0, "V_reduce" => 0, "Z" => 0)

    updateU = function(i)
        inv_mat = 2 * V_iterate' * Diagonal(S[i, :]) * V_iterate
        inv_mat += (2 * gamma + rho_2) * Matrix(I, k, k)
        inv_mat = inv(inv_mat)
        temp_vec = Psi_iterate[i, :] + rho_2 * Z_iterate[i, :]
        return inv_mat * (2 * V_iterate' * A[i, :] + temp_vec)
    end

    updateV = function(j)
        inv_mat = 2 * U_iterate' * Diagonal(S[:, j]) * U_iterate
        inv_mat += 2 * gamma * Matrix(I, k, k)
        inv_mat = inv(inv_mat)
        return inv_mat * (2 * U_iterate' * A[:, j])
    end

    for iteration=1:max_iteration

        # Perform U Update
        start = now()
        UParUpdate = pmap(updateU, collect(1:n))
        close = now()
        update_time["U_map"] += Dates.value(close - start)
        start = now()
        for i=1:n
            U_iterate[i, :] = UParUpdate[i]
        end
        close = now()
        update_time["U_reduce"] += Dates.value(close - start)

        append!(augL_hist, computeAugL())

        # Perform P Update
        start = now()
        temp = Phi_iterate * Z_iterate' / 2
        temp += temp'
        temp += cache_Y + (rho_1 / 2) * Z_iterate * Z_iterate'
        L, _, _ = tsvd(temp, k)
        P_iterate = L * L'
        close = now()
        update_time["P"] += Dates.value(close - start)

        append!(augL_hist, computeAugL())

        # Perform V Update
        start = now()
        VParUpdate = pmap(updateV, collect(1:m))
        close = now()
        update_time["V_map"] += Dates.value(close - start)
        start = now()
        for j=1:m
            V_iterate[j, :] = VParUpdate[j]
        end
        close = now()
        update_time["V_reduce"] += Dates.value(close - start)

        append!(augL_hist, computeAugL())

        # Perform Z Update
        start = now()
        temp = rho_2 * U_iterate - Psi_iterate
        temp -= (Matrix(I, n, n) - P_iterate) * Phi_iterate
        Z_iterate = (Matrix(I, n, n) + (rho_1 / rho_2) * P_iterate) * temp
        Z_iterate = Z_iterate / (rho_1 + rho_2)
        close = now()
        update_time["Z"] += Dates.value(close - start)

        append!(augL_hist, computeAugL())

        # Perform Sigma1 Update
        Phi_residual = (Matrix(I, n, n) - P_iterate) * Z_iterate
        Phi_iterate += rho_1 * Phi_residual

        append!(augL_hist, computeAugL())

        # Perform Sigma2 Update
        Psi_residual = Z_iterate - U_iterate
        Psi_iterate += rho_2 * Psi_residual

        append!(augL_hist, computeAugL())
        append!(Phi_residual_hist, norm(Phi_residual)^2)
        append!(Psi_residual_hist, norm(Psi_residual)^2)

        # Check for termination
        if max(norm(Phi_residual)^2, norm(Psi_residual)^2) < residual_threshold
            break
        end

    end

    return U_iterate, V_iterate, P_iterate, Z_iterate, Phi_iterate,
           Psi_iterate, (Phi_residual_hist, Psi_residual_hist, augL_hist, update_time)

end

function admm(A, k, Y, lambda; gamma=0.01, rho_1=10, rho_2=10,
              residual_threshold=0.01, max_iteration=100)

    function computeAugL()

        U = U_iterate
        V = V_iterate
        P = P_iterate
        Z = Z_iterate
        Phi = Phi_iterate
        Psi = Psi_iterate

        obj = norm(S .* (U * V' - A)) ^ 2
        obj += lambda * tr(Y' * (Matrix(I, n, n) - P) * Y)
        obj += gamma * (norm(U) ^ 2 + norm(V) ^ 2)
        obj += tr(Phi' * (Matrix(I, n, n) - P) * Z)
        obj += tr(Psi' * (Z - U))
        obj += rho_1 / 2 * norm((Matrix(I, n, n) - P) * Z) ^ 2
        obj += rho_2 / 2 * norm(Z - U) ^ 2

        return obj
    end

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

    Phi_iterate = ones(n, k)
    Psi_iterate = ones(n, k)

    Phi_residual_hist = []
    Psi_residual_hist = []
    augL_hist = [computeAugL()]

    cache_Y = lambda * Y * Y'

    update_time = Dict("U" => 0, "P" => 0, "V" => 0, "Z" => 0)

    for iteration=1:max_iteration

        # Perform U Update
        start = now()
        for i=1:n
            inv_mat = 2 * V_iterate' * Diagonal(S[i, :]) * V_iterate
            inv_mat += (2 * gamma + rho_2) * Matrix(I, k, k)
            inv_mat = inv(inv_mat)
            temp_vec = Psi_iterate[i, :] + rho_2 * Z_iterate[i, :]
            U_iterate[i, :] = inv_mat * (2 * V_iterate' * A[i, :] + temp_vec)
        end
        close = now()
        update_time["U"] += Dates.value(close - start)

        append!(augL_hist, computeAugL())

        # Perform P Update
        start = now()
        temp = Phi_iterate * Z_iterate' / 2
        temp += temp'
        temp += cache_Y + (rho_1 / 2) * Z_iterate * Z_iterate'
        L, _, _ = tsvd(temp, k)
        P_iterate = L * L'
        close = now()
        update_time["P"] += Dates.value(close - start)

        append!(augL_hist, computeAugL())

        # Perform V Update
        start = now()
        for j=1:m
            inv_mat = 2 * U_iterate' * Diagonal(S[:, j]) * U_iterate
            inv_mat += 2 * gamma * Matrix(I, k, k)
            inv_mat = inv(inv_mat)
            V_iterate[j, :] = inv_mat * (2 * U_iterate' * A[:, j])
        end
        close = now()
        update_time["V"] += Dates.value(close - start)

        append!(augL_hist, computeAugL())

        # Perform Z Update
        start = now()
        temp = rho_2 * U_iterate - Psi_iterate
        temp -= (Matrix(I, n, n) - P_iterate) * Phi_iterate
        Z_iterate = (Matrix(I, n, n) + (rho_1 / rho_2) * P_iterate) * temp
        Z_iterate = Z_iterate / (rho_1 + rho_2)
        close = now()
        update_time["Z"] += Dates.value(close - start)

        append!(augL_hist, computeAugL())

        # Perform Sigma1 Update
        Phi_residual = (Matrix(I, n, n) - P_iterate) * Z_iterate
        Phi_iterate += rho_1 * Phi_residual

        append!(augL_hist, computeAugL())

        # Perform Sigma2 Update
        Psi_residual = Z_iterate - U_iterate
        Psi_iterate += rho_2 * Psi_residual

        append!(augL_hist, computeAugL())
        append!(Phi_residual_hist, norm(Phi_residual)^2)
        append!(Psi_residual_hist, norm(Psi_residual)^2)

        # Check for termination
        if max(norm(Phi_residual)^2, norm(Psi_residual)^2) < residual_threshold
            break
        end

    end

    return U_iterate, V_iterate, P_iterate, Z_iterate, Phi_iterate,
           Psi_iterate, (Phi_residual_hist, Psi_residual_hist, augL_hist, update_time)

end

function admmAdaptive(A, k, Y, lambda; gamma=0.01, adaptive_step_size=false,
              step_size_epsilon_threshold=0.2, rho_1=10, rho_2=10,
              residual_threshold=0.01, max_iteration=100, n_inner=1)

    function computeAugL()

        U = U_iterate
        V = V_iterate
        P = P_iterate
        Z = Z_iterate
        Phi = Phi_iterate
        Psi = Psi_iterate

        obj = norm(S .* (U * V' - A)) ^ 2
        obj += lambda * tr(Y' * (Matrix(I, n, n) - P) * Y)
        obj += gamma * (norm(U) ^ 2 + norm(V) ^ 2)
        obj += tr(Phi' * (Matrix(I, n, n) - P) * Z)
        obj += tr(Psi' * (Z - U))
        obj += rho_1 / 2 * norm((Matrix(I, n, n) - P) * Z) ^ 2
        obj += rho_2 / 2 * norm(Z - U) ^ 2

        return obj
    end

    function evaluateGrad()

        Phi_grad = rho_2 * U_iterate - Psi_iterate - 2 * Phi_iterate
        Phi_grad = (Matrix(I, n, n) - P_iterate) * Phi_grad / (rho_1 + rho_2)

        Psi_grad = rho_2 * U_iterate - 2 * Psi_iterate - (Matrix(I, n, n) - P_iterate) * Phi_iterate
        Psi_grad = (Matrix(I, n, n) - rho_1 * P_iterate / rho_2) * Psi_grad
        Psi_grad = Psi_grad / (rho_1 + rho_2)

        return (Phi_grad, Psi_grad)

    end

    function updateStepSize()

        Phi_delta = new_dual_vars[1] - old_dual_vars[1]
        Psi_delta = new_dual_vars[2] - old_dual_vars[2]
        grad_Phi_delta = new_grad[1] - old_grad[1]
        grad_Psi_delta = new_grad[2] - old_grad[2]

        alpha_SD = norm(Phi_delta) ^ 2 / tr(Phi_delta' * grad_Phi_delta)
        alpha_MG = tr(Phi_delta' * grad_Phi_delta) / norm(grad_Phi_delta) ^ 2

        println(alpha_SD)
        println(alpha_MG)
        println()

        if 2 * alpha_MG > alpha_SD
            alpha = alpha_MG
        else
            alpha = alpha_SD - alpha_MG / 2
        end

        alpha_corr = tr(Phi_delta' * grad_Phi_delta) / (norm(Phi_delta) * norm(grad_Phi_delta))

        beta_SD = norm(Psi_delta) ^ 2 / tr(Psi_delta' * grad_Psi_delta)
        beta_MG = tr(Psi_delta' * grad_Psi_delta) / norm(grad_Psi_delta) ^ 2

        println(beta_SD)
        println(beta_MG)
        println()

        if 2 * beta_MG > beta_SD
            beta = beta_MG
        else
            beta = beta_SD - beta_MG / 2
        end

        beta_corr = tr(Psi_delta' * grad_Psi_delta) / (norm(Psi_delta) * norm(grad_Psi_delta))

        if alpha_corr <= step_size_epsilon_threshold
            alpha = rho_1
        end
        if beta_corr <= step_size_epsilon_threshold
            beta = rho_2
        end

        return alpha, beta

    end

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

    Phi_iterate = ones(n, k)
    Psi_iterate = ones(n, k)

    Phi_residual_hist = []
    Psi_residual_hist = []
    augL_hist = [computeAugL()]

    cache_Y = lambda * Y * Y'

    old_grad = nothing
    new_grad = nothing
    old_dual_vars = nothing
    new_dual_vars = nothing

    for iteration=1:max_iteration

        if adaptive_step_size
            if iteration % 2 == 1
                old_grad = evaluateGrad()
                old_dual_vars = (Phi_iterate, Psi_iterate)
            else
                new_grad = evaluateGrad()
                new_dual_vars = (Phi_iterate, Psi_iterate)
            end
        end

        for inner_iter=1:n_inner

            # Perform U Update
            for i=1:n
                inv_mat = 2 * V_iterate' * Diagonal(S[i, :]) * V_iterate
                inv_mat += (2 * gamma + rho_2) * Matrix(I, k, k)
                inv_mat = inv(inv_mat)
                temp_vec = Psi_iterate[i, :] + rho_2 * Z_iterate[i, :]
                U_iterate[i, :] = inv_mat * (2 * V_iterate' * A[i, :] + temp_vec)
                if adaptive_step_size
                    if iteration % 2 == 1
                        old_grad[2][i, :] -= U_iterate[i, :]
                        old_grad[2][i, :] -= inv_mat * Psi_iterate[i, :]
                    else
                        new_grad[2][i, :] -= U_iterate[i, :]
                        new_grad[2][i, :] -= inv_mat * Psi_iterate[i, :]
                    end
                end
            end

            append!(augL_hist, computeAugL())

            # Perform P Update
            temp = Phi_iterate * Z_iterate' / 2
            temp += temp'
            temp += cache_Y + (rho_1 / 2) * Z_iterate * Z_iterate'
            L, _, _ = tsvd(temp, k)
            P_iterate = L * L'

            append!(augL_hist, computeAugL())

            # Perform V Update
            for j=1:m
                inv_mat = 2 * U_iterate' * Diagonal(S[:, j]) * U_iterate
                inv_mat += 2 * gamma * Matrix(I, k, k)
                inv_mat = inv(inv_mat)
                V_iterate[j, :] = inv_mat * (2 * U_iterate' * A[:, j])
            end

            append!(augL_hist, computeAugL())

            # Perform Z Update
            temp = rho_2 * U_iterate - Psi_iterate
            temp -= (Matrix(I, n, n) - P_iterate) * Phi_iterate
            Z_iterate = (Matrix(I, n, n) + (rho_1 / rho_2) * P_iterate) * temp
            Z_iterate = Z_iterate / (rho_1 + rho_2)

            append!(augL_hist, computeAugL())

        end

        if adaptive_step_size & (iteration % 2 == 0)
            rho_1, rho_2 = updateStepSize()
            println(rho_1)
            println(rho_2)
            println()
        end

        # Perform Sigma1 Update
        Phi_residual = (Matrix(I, n, n) - P_iterate) * Z_iterate
        Phi_iterate += rho_1 * Phi_residual

        append!(augL_hist, computeAugL())

        # Perform Sigma2 Update
        Psi_residual = Z_iterate - U_iterate
        Psi_iterate += rho_2 * Psi_residual

        append!(augL_hist, computeAugL())
        append!(Phi_residual_hist, norm(Phi_residual)^2)
        append!(Psi_residual_hist, norm(Psi_residual)^2)

        # Check for termination
        if max(norm(Phi_residual)^2, norm(Psi_residual)^2) < residual_threshold
            break
        end

    end

    return U_iterate, V_iterate, P_iterate, Z_iterate, Phi_iterate,
           Psi_iterate, (Phi_residual_hist, Psi_residual_hist, augL_hist)

end
;
