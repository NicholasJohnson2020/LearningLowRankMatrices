function admm(A, k, Y, lambda; gamma=0.01, step_size=10,
              residual_threshold=0.01, max_iteration=100,
              initialization="approximate", P_update="exact")

    @assert initialization in ["exact", "approximate"]
    @assert P_update in ["exact", "sub_sketch"]

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
        lag = obj + tr(Phi' * (Matrix(I, n, n) - P) * Z)
        lag += tr(Psi' * (Z - U))
        lag += rho_1 / 2 * norm((Matrix(I, n, n) - P) * Z) ^ 2
        lag += rho_2 / 2 * norm(Z - U) ^ 2

        return obj, lag
    end

    opts = LRAOptions()
    opts.sketch = :sub

    rho_1 = step_size
    rho_2 = step_size

    (n, m) = size(A)
    d = size(Y)[2]

    @assert size(Y)[1] == n

    S = sparse(findnz(A)...)

    if initialization == "exact"
        L, sigma, R = tsvd(A, k)
    else
        L, sigma, R = psvd(A, rank=k, opts)
    end

    U_iterate = L * Diagonal(sqrt.(sigma))
    V_iterate = R * Diagonal(sqrt.(sigma))
    Z_iterate = U_iterate
    P_iterate = SymLowRankMat(L)

    Phi_iterate = ones(Float64, n, k)
    Psi_iterate = ones(Float64, n, k)

    Phi_residual_hist = []
    Psi_residual_hist = []
    obj_hist = []
    augL_hist = []
    obj, augL = computeAugL()
    append!(obj_hist, obj)
    append!(augL_hist, augL)


    update_time = Dict("U" => 0, "P" => 0, "V" => 0, "Z" => 0)

    for iteration=1:max_iteration

        # Perform U Update
        start = now()
        Threads.@threads for i=1:n
            inv_mat = 2 * V_iterate' * Diagonal(S[i, :]) * V_iterate
            inv_mat += (2 * gamma + rho_2) * Matrix(I, k, k)
            temp_vec = Psi_iterate[i, :] + rho_2 * Z_iterate[i, :]
            U_iterate[i, :] = inv_mat \ (2 * V_iterate' * A[i, :] + temp_vec)
        end
        close = now()
        update_time["U"] += Dates.value(close - start)

        obj, augL = computeAugL()
        append!(obj_hist, obj)
        append!(augL_hist, augL)

        # Perform P Update
        start = now()
        factor_1 = [lambda * Y rho_1 / 2 * Z_iterate Phi_iterate / 2 Z_iterate / 2]
        factor_2 = [Y Z_iterate Z_iterate Phi_iterate]
        temp = LowRankMat(factor_1, factor_2)
        if P_update == "exact"
            L, _, _ = tsvd(temp, k)
        else
            L, _, _ = tsvd(subSketch(temp, k), k)
        end
        P_iterate = SymLowRankMat(L)
        close = now()
        update_time["P"] += Dates.value(close - start)

        obj, augL = computeAugL()
        append!(obj_hist, obj)
        append!(augL_hist, augL)

        # Perform V Update
        start = now()
        Threads.@threads for j=1:m
            inv_mat = 2 * U_iterate' * Diagonal(S[:, j]) * U_iterate
            inv_mat += 2 * gamma * Matrix(I, k, k)
            V_iterate[j, :] = inv_mat \ (2 * U_iterate' * A[:, j])
        end
        close = now()
        update_time["V"] += Dates.value(close - start)

        obj, augL = computeAugL()
        append!(obj_hist, obj)
        append!(augL_hist, augL)

        # Perform Z Update
        start = now()
        temp = Phi_iterate + rho_1 * U_iterate - (rho_1 / rho_2) * Psi_iterate
        temp = P_iterate * temp
        Z_iterate = (rho_2 * U_iterate - Psi_iterate - Phi_iterate + temp) / (rho_1 + rho_2)
        close = now()
        update_time["Z"] += Dates.value(close - start)

        obj, augL = computeAugL()
        append!(obj_hist, obj)
        append!(augL_hist, augL)

        # Perform Sigma1 Update
        Phi_residual = Z_iterate - P_iterate * Z_iterate
        Phi_iterate += rho_1 * Phi_residual

        obj, augL = computeAugL()
        append!(obj_hist, obj)
        append!(augL_hist, augL)

        # Perform Sigma2 Update
        Psi_residual = Z_iterate - U_iterate
        Psi_iterate += rho_2 * Psi_residual

        obj, augL = computeAugL()
        append!(obj_hist, obj)
        append!(augL_hist, augL)

        append!(Phi_residual_hist, norm(Phi_residual)^2)
        append!(Psi_residual_hist, norm(Psi_residual)^2)

        # Check for termination
        if max(norm(Phi_residual)^2, norm(Psi_residual)^2) < residual_threshold
            break
        end

    end

    return U_iterate, V_iterate, P_iterate, Z_iterate, Phi_iterate,
           Psi_iterate, (0, 0, update_time),
           (obj_hist, augL_hist, Phi_residual_hist, Psi_residual_hist)

end

function admmV0(A, k, Y, lambda; gamma=0.01, step_size=10,
                residual_threshold=0.01, max_iteration=100)

    """
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
    """

    rho_1 = step_size
    rho_2 = step_size

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
    #augL_hist = [computeAugL()]

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

        #append!(augL_hist, computeAugL())

        # Perform P Update
        start = now()
        temp = Phi_iterate * Z_iterate' / 2
        temp += temp'
        temp += cache_Y + (rho_1 / 2) * Z_iterate * Z_iterate'
        L, _, _ = tsvd(temp, k)
        P_iterate = L * L'
        close = now()
        update_time["P"] += Dates.value(close - start)

        #append!(augL_hist, computeAugL())

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

        #append!(augL_hist, computeAugL())

        # Perform Z Update
        start = now()
        temp = rho_2 * U_iterate - Psi_iterate
        temp -= (Matrix(I, n, n) - P_iterate) * Phi_iterate
        Z_iterate = (Matrix(I, n, n) + (rho_1 / rho_2) * P_iterate) * temp
        Z_iterate = Z_iterate / (rho_1 + rho_2)
        close = now()
        update_time["Z"] += Dates.value(close - start)

        #append!(augL_hist, computeAugL())

        # Perform Sigma1 Update
        Phi_residual = (Matrix(I, n, n) - P_iterate) * Z_iterate
        Phi_iterate += rho_1 * Phi_residual

        #append!(augL_hist, computeAugL())

        # Perform Sigma2 Update
        Psi_residual = Z_iterate - U_iterate
        Psi_iterate += rho_2 * Psi_residual

        #append!(augL_hist, computeAugL())
        append!(Phi_residual_hist, norm(Phi_residual)^2)
        append!(Psi_residual_hist, norm(Psi_residual)^2)

        # Check for termination
        if max(norm(Phi_residual)^2, norm(Psi_residual)^2) < residual_threshold
            break
        end

    end

    return U_iterate, V_iterate, P_iterate, Z_iterate, Phi_iterate,
           Psi_iterate, (Phi_residual_hist,
                         Psi_residual_hist,
                         #augL_hist,
                         update_time)

end

;
