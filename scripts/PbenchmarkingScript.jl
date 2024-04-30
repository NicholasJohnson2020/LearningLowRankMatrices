using Random, Distributions, Dates, Statistics, DataFrames, CSV, LinearAlgebra
using RandomizedLingAlg, LowRankApprox

include("../lowRankMatrixLearning.jl")

N = [100, 200, 400, 800, 1000, 2000, 5000]
#N = 1000
#M = [100, 200, 400, 800, 1000, 2000]
M = 100
#K = [5, 10, 15, 20, 25, 30, 35, 40]
K = 5

d = 50


lambda = 1 / 100
rho_1 = 10
rho_2 = 10

num_trials = 100

unif = Uniform(0, 1)

output_root = "P_update"
method_list = ["tsvd", "rsvd", "rsvd_fnkz", "pqr_randn", "svd_randn", "eig_randn",
               "pqr_sub", "svd_sub", "eig_sub"]

data_dict = Dict()
for method in method_list
    data_dict[method] = Dict()
    for m in M
        data_dict[method][m]["time"] = []
        data_dict[method][m]["optimality_loss"] = []
        data_dict[method][m]["error_loss"] = []
    end
end

for n in N

    for i=1:num_trials

        # Generate the data
        Y = rand(unif, (n, d))
        Z = rand(unif, (n, K))
        Phi = rand((unif, (n, K)))

        temp = Phi * Z' / 2
        temp += temp'
        temp += Y * Y' + (rho_1 / 2) * Z * Z'

        # For tsvd implementation
        start = now()
        L, _, _ = tsvd(temp, K)
        close = now()
        elapsed_time = Dates.value(close - start)

        opt_val = L' * temp
        opt_val = tr(L * opt_val)
        opt_sol = L * L'

        append!(data_dict["tsvd"][n]["time"], elapsed_time)
        append!(data_dict["tsvd"][n]["optimality_loss"], 0)
        append!(data_dict["tsvd"][n]["error_loss"], 0)

        # For rsvd implementation
        start = now()
        L, _, _ = rsvd(temp, K)
        close = now()
        elapsed_time = Dates.value(close - start)

        this_val = L' * temp
        this_val = tr(L * this_val)
        this_sol = L * L'
        append!(data_dict["rsvd"][n]["time"], elapsed_time)
        append!(data_dict["rsvd"][n]["optimality_loss"], (opt_val - this_val) / opt_val)
        append!(data_dict["rsvd"][n]["error_loss"], norm(opt_sol - this_sol)^2 / norm(opt_sol)^2)

        # For rsvd_fnkz implementation
        start = now()
        L, _, _ = rsvd_fnkz(temp, K)
        close = now()
        elapsed_time = Dates.value(close - start)

        this_val = L' * temp
        this_val = tr(L * this_val)
        this_sol = L * L'
        append!(data_dict["rsvd_fnkz"][n]["time"], elapsed_time)
        append!(data_dict["rsvd_fnkz"][n]["optimality_loss"], (opt_val - this_val) / opt_val)
        append!(data_dict["rsvd_fnkz"][n]["error_loss"], norm(opt_sol - this_sol)^2 / norm(opt_sol)^2)


        sketch_types = [:randn, :sub]

        for sketch_type in sketch_types
            opts = LRAOptions()
            opts.sketch = sketch_type

            # For pqrfact gaussian sketch implementation
            start = now()
            L, _, _ = pqrfact(temp, rank=K, opts)
            close = now()
            elapsed_time = Dates.value(close - start)

            this_val = L' * temp
            this_val = tr(L * this_val)
            this_sol = L * L'
            append!(data_dict["pqr_" * string(sketch_type)][n]["time"], elapsed_time)
            append!(data_dict["pqr_" * string(sketch_type)][n]["optimality_loss"], (opt_val - this_val) / opt_val)
            append!(data_dict["pqr_" * string(sketch_type)][n]["error_loss"], norm(opt_sol - this_sol)^2 / norm(opt_sol)^2)

            # For svd gaussian sketch implementation
            start = now()
            L, _, _ = psvd(temp, rank=K, opts)
            close = now()
            elapsed_time = Dates.value(close - start)

            this_val = L' * temp
            this_val = tr(L * this_val)
            this_sol = L * L'
            append!(data_dict["svd_" * string(sketch_type)][n]["time"], elapsed_time)
            append!(data_dict["svd_" * string(sketch_type)][n]["optimality_loss"], (opt_val - this_val) / opt_val)
            append!(data_dict["svd_" * string(sketch_type)][n]["error_loss"], norm(opt_sol - this_sol)^2 / norm(opt_sol)^2)

            # For eig gaussian sketch implementation
            start = now()
            _, L = psvd(temp, rank=K, opts)
            close = now()
            elapsed_time = Dates.value(close - start)

            this_val = L' * temp
            this_val = tr(L * this_val)
            this_sol = L * L'
            append!(data_dict["eig_" * string(sketch_type)][n]["time"], elapsed_time)
            append!(data_dict["eig_" * string(sketch_type)][n]["optimality_loss"], (opt_val - this_val) / opt_val)
            append!(data_dict["eig_" * string(sketch_type)][n]["error_loss"], norm(opt_sol - this_sol)^2 / norm(opt_sol)^2)
        end
    end
end

for method in method_list

    df = DataFrame(N=Int64[], exec_time=Float64[], exec_time_std=Float64[],
                   opt_loss=Float64[], opt_loss_std=Float64[],
                   error_loss=Float64[], error_loss_std=Float64[])

    for n in N
        current_row = [n,
                       Statistics.mean(data_dict[method][m]["time"]),
                       Statistics.std(data_dict[method][m]["time"]) / (num_trials^0.5),
                       Statistics.mean(data_dict[method][m]["optimality_loss"]),
                       Statistics.std(data_dict[method][m]["optimality_loss"]) / (num_trials^0.5),
                       Statistics.mean(data_dict[method][m]["error_loss"]),
                       Statistics.std(data_dict[method][m]["error_loss"]) / (num_trials^0.5)]

        push!(df, current_row)
    end

    CSV.write(output_root * "_" * method * ".csv", df)

end
