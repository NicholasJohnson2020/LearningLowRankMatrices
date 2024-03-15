function perspectiveRelaxationPrimal(A, k, Y, c_1, c_2, lambda;
                                     solver_output=false)

    (n, m) = size(A)
    d = size(Y)[2]

    nonzero_list = []
    for (i, j, value) in zip(findnz(A)...)
        append!(nonzero_list, [(i, j)])
    end

    # Build optmization problem
    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, MOI.Silent(), !solver_output)

    @variable(model, X[i=1:n, j=1:m])
    @variable(model, P[i=1:n, j=1:n])
    @variable(model, Theta[i=1:m, j=1:m])

    @constraint(model, sum(P[i, i] for i=1:n) == k)

    @constraint(model, P in PSDCone())
    @constraint(model, (1 * Matrix(I, n, n) - P) in PSDCone())
    @constraint(model, [Theta X'; X P] in PSDCone())

    term_1 = sum((X[i, j] - A[i, j]) ^ 2 for (i, j) in nonzero_list)
    temp_mat = Y' * (1 * Matrix(I, n, n) - P) * Y

    @objective(model, Min, c_1 * term_1 + c_2 * sum(temp_mat[i, i] for i=1:d)
               + lambda * sum(Theta[i, i] for i=1:m))

    JuMP.optimize!(model)

    return termination_status(model), objective_value(model),
           (value.(X), value.(P))

end

function perspectiveRelaxationDual(A, k, Y, c_1, c_2, lambda;
                                     solver_output=false)

    (n, m) = size(A)
    d = size(Y)[2]

    nonzero_list = []
    for (i, j, value) in zip(findnz(A)...)
        append!(nonzero_list, [(i, j)])
    end

    # Build optmization problem
    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, MOI.Silent(), !solver_output)

    @variable(model, sigma)
    #@variable(model, Lambda_11[i=1:m, j=1:m])
    @variable(model, Lambda_12[i=1:n, j=1:m])
    @variable(model, Lambda_22[i=1:n, j=1:n])

    for i=1:n, j=1:m
        if (i, j) in nonzero_list
            continue
        end
        @constraint(model, Lambda_12[i, j] == 0)
    end

    for i=1:n, j=1:n
        if i==j
            continue
        end
        @constraint(model, Lambda_22[i, j] == 0)
    end
    #@constraint(model, (lambda * Matrix(I, m, m) - Lambda_11) in PSDCone())
    @constraint(model, (Lambda_22 - sigma * Matrix(I, n, n) + c_2 * Y * Y') in PSDCone())
    @constraint(model, [lambda * Matrix(I, m, m) Lambda_12';
                        Lambda_12 Lambda_22] in PSDCone())

    @objective(model, Max, (n - k) * sigma - sum(Lambda_22[i, i] for i=1:n) -
               sum(2 * Lambda_12[i, j] * A[i, j] + Lambda_12[i, j]^2/c_1 for (i, j) in nonzero_list))

    JuMP.optimize!(model)

    return termination_status(model), objective_value(model),
           (value.(Lambda_12), value.(Lambda_22), value.(sigma))

end
