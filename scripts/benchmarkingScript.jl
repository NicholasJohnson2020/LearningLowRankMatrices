using Random, Distributions, Dates, Statistics, DataFrames, CSV

rho_1 = 10
rho_2 = 10

N = [100, 200, 400, 800, 1000, 2000, 5000]
#N = 1000
#M = [100, 200, 400, 800, 1000, 2000, 5000, 10000]
M = 100
#K = [5, 10, 15, 20, 25, 30, 35, 40]
K = 5

num_trials = 1000

unif = Uniform(0, 1)

output_root = "Z_update"
method_list = ["existing", "simplified", "factored"]

data_dict = Dict()
for method in method_list
    data_dict[method] = Dict()
    for n in N
        data_dict[method][n] = []
    end
end

for n in N
    for i=1:num_trials

        # Generate the data
        U = rand(unif, (n, K))
        Psi = rand(unif, (n, K))
        Phi = rand(unif, (n, K))
        M = rand(unif, (n, K))
        P = M * M'

        # Existing implementation
        start = now()
        temp = rho_2 * U_iterate - Psi_iterate
        temp -= (Matrix(I, n, n) - P_iterate) * Phi_iterate
        Z_iterate = (Matrix(I, n, n) + (rho_1 / rho_2) * P_iterate) * temp
        Z_iterate = Z_iterate / (rho_1 + rho_2)
        close = now()

        elapsed_time = Dates.value(close - start)
        append!(data["existing"][n], elapsed_time)

        # Simplified implementation
        start = now()
        temp = Phi + rho_1 * U - (rho_1 / rho_2) * Psi
        Z_iterate = (P * temp + rho_2 * U - Psi - Phi) / (rho_1 + rho_2)
        close = now()

        elapsed_time = Dates.value(close - start)
        append!(data["simplified"][n], elapsed_time)

        # Factored implementation
        start = now()
        temp = Phi + rho_1 * U - (rho_1 / rho_2) * Psi
        temp = M' * temp
        temp = M * temp
        Z_iterate = (rho_2 * U - Psi - Phi + temp) / (rho_1 + rho_2)
        close = now()

        elapsed_time = Dates.value(close - start)
        append!(data["factored"][n], elapsed_time)

    end
end

for method in method_list

    df = DataFrame(N=Int64[], exec_time=Float64[], exec_time_std=Float64[])

    for n in N
        current_row = [n,
                       Statistics.mean(data[method][n]),
                       Statistics.std(data[method][n]) / (num_trials^0.5)]

        push!(df, current_row)
    end

    CSV.write(output_root * "_" * method * ".csv", df)

end
