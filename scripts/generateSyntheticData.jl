include("../lowRankMatrixLearning.jl")

file_path = ARGS[1]

NUM_TRIALS_PER_CONFIG = 10

N = [100, 200, 400, 800, 1000, 2000, 5000, 10000]
#N = 1000
#M = [100, 1000, 10000, 100000]
M = 100
#k = [5, 10, 15, 20, 25, 30, 35, 40]
K = 5
#d = [10, 50, 100, 150, 200, 500]
D = 50
missing_frac = 0.9
noise_variance = 2

config_count = 0

data_dict = Dict()
param_dict = Dict()

data_dict["Trials"] = NUM_TRIALS_PER_CONFIG
data_dict["n"] = N
data_dict["m"] = M
data_dict["k"] = K
data_dict["d"] = D
data_dict["missing"] = missing_frac
data_dict["noise"] = noise_variance

start = now()
# Main loop to sample data
for n in N, m in M, k in K, d in D, frac in missing_frac, noise in noise_variance

    global config_count += 1
    param_dict[config_count] = Dict("N"=>n, "M"=>m, "K"=>k, "D"=>d,
                                    "frac"=>missing_frac, "noise"=>noise)

    current_data_dict = Dict()
    for trial_num = 1:NUM_TRIALS_PER_CONFIG
        A_true, A_observed, Y, _ = sample_data(m, n, k, d, frac, noise)
        current_data_dict[trial_num] = Dict("A_true"=>A_true,
                                            "A_observed"=>A_observed,
                                            "Y"=>Y,
                                            "k"=>k)
    end

    data_dict[string(param_dict[config_count])] = current_data_dict

end

elapsed_time = now() - start

f = open(file_path * "data.json","w")
JSON.print(f, JSON.json(data_dict))
close(f)

f = open(file_path * "params.json","w")
JSON.print(f, JSON.json(param_dict))
close(f)

println("The total number of configs is $config_count.")

println("Execution time is $elapsed_time")
