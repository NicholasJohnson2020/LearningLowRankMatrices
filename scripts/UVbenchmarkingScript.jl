using Random, Distributions, Dates, Statistics, DataFrames, CSV, LinearAlgebra
using Distributed

include("../lowRankMatrixLearning.jl")

#N = [100, 200, 400, 800, 1000, 2000, 5000]
N = 1000
M = [100, 200, 400, 800, 1000, 2000, 4000]
#M = 100
#K = [5, 10, 15, 20, 25, 30, 35, 40]
K = 5

d = 50
missing_frac = 0.9
noise_variance = 2

gamma = 1 / 10

num_trials = 100

unif = Uniform(0, 1)

output_root = "V_update"
method_list = ["loop", "pmap", "map"]

data_dict = Dict()
for method in method_list
    data_dict[method] = Dict()
    for m in M
        data_dict[method][m] = []
    end
end

for m in M

    for i=1:num_trials

        # Generate the data
        U = rand(unif, (N, K))
        _, A, _, _ = sample_data(m, N, K, d, missing_frac, noise_variance)
        S = zeros(N, m)
        for (i, j, value) in zip(findnz(A)...)
          S[i, j] = 1
        end
        S = sparse(S)

        updateV = function(j)
            inv_mat = 2 * U' * Diagonal(S[:, j]) * U
            inv_mat += 2 * gamma * Matrix(I, K, K)
            inv_mat = inv(inv_mat)
            return inv_mat * (2 * U' * A[:, j])
        end


        # For loop implementation
        V_iterate = zeros(m, K)
        start = now()
        for j=1:m
            inv_mat = 2 * U' * Diagonal(S[:, j]) * U
            inv_mat += 2 * gamma * Matrix(I, K, K)
            inv_mat = inv(inv_mat)
            V_iterate[j, :] = inv_mat * (2 * U' * A[:, j])
        end
        close = now()

        elapsed_time = Dates.value(close - start)
        append!(data_dict["loop"][m], elapsed_time)

        # pmap implementation
        V_iterate = zeros(m, K)
        start = now()
        VParUpdate = pmap(updateV, collect(1:m))
        for j=1:m
            V_iterate[j, :] = VParUpdate[j]
        end
        close = now()

        elapsed_time = Dates.value(close - start)
        append!(data_dict["pmap"][m], elapsed_time)

        # map implementation
        V_iterate = zeros(m, K)
        start = now()
        VParUpdate = map(updateV, collect(1:m))
        for j=1:m
            V_iterate[j, :] = VParUpdate[j]
        end
        close = now()

        elapsed_time = Dates.value(close - start)
        append!(data_dict["map"][m], elapsed_time)

    end
end

for method in method_list

    df = DataFrame(M=Int64[], exec_time=Float64[], exec_time_std=Float64[])

    for m in M
        current_row = [m,
                       Statistics.mean(data_dict[method][m]),
                       Statistics.std(data_dict[method][m]) / (num_trials^0.5)]

        push!(df, current_row)
    end

    CSV.write(output_root * "_" * method * ".csv", df)

end
