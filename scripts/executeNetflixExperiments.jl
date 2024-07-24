include("../lowRankMatrixLearning.jl")

function unserialize_matrix(mat)
    """
    This function unserializes an array of arrays into a 2-dimensional matrix.

    :param mat: An array of arrays

    :return: This function returns the matrix version of the input array of
             arrays.
    """
    n = size(mat[1])[1]
    m = size(mat)[1]
    output = zeros(n, m)
    for i=1:n, j=1:m
        output[i, j] = mat[j][i]
    end
    return output
end;


method_name = ARGS[1]
input_path = ARGS[2]
data_type = ARGS[3]
task_ID_input = parse(Int64, ARGS[4])
num_tasks_input = parse(Int64, ARGS[5])

valid_methods = ["admm_exact", "admm_sub", "fastImpute", "softImpute"]
valid_types = ["4Y", "6Y"]

@assert method_name in valid_methods
@assert data_type in valid_types

output_path = input_path * data_type * '/' * method_name * '/'

# Load the data
netflix_data = Dict()
open(input_path * data_type * '/' * data_type * "_data.json", "r") do f
    global netflix_data
    dicttxt = JSON.read(f, String)  # file information to string
    netflix_data = JSON.parse(dicttxt)  # parse and transform data
    netflix_data = JSON.parse(netflix_data)
end

# Load the experiment parameters
param_dict = Dict()
open(input_path * data_type * '/' * data_type * "_params.json", "r") do f
    global param_dict
    dicttxt = JSON.read(f, String)  # file information to string
    param_dict = JSON.parse(dicttxt)  # parse and transform data
    param_dict = JSON.parse(param_dict)
end

NUM_TRIALS = netflix_data["Trials"]

start_time_global = now()

# Main loop to execute experiments
K = [3, 4, 5, 6, 7, 8, 9, 10]

task_ID_list = collect((task_ID_input+1):num_tasks_input:length(K))

Y = unserialize_matrix(netflix_data["Y"])
for (index, k) in enumerate(K)

    global netflix_data
    global param_dict

    # Load experiment specific data
    n = param_dict[string(index)]["N"]
    m = param_dict[string(index)]["M"]
    k_target = param_dict[string(index)]["K"]
    d = param_dict[string(index)]["D"]
    missing_frac = param_dict[string(index)]["frac"]

    # Create dictionary to store experiment results
    experiment_results = Dict()
    experiment_results["Method"] = method_name
    experiment_results["N"] = n
    experiment_results["M"] = m
    experiment_results["K"] = k_target
    experiment_results["D"] = d
    experiment_results["frac"] = missing_frac

    experiment_results["Trials"] = NUM_TRIALS

    experiment_results["train_size"] = []
    experiment_results["test_size"] = []
    experiment_results["in_reconstruction_error"] = []
    experiment_results["in_error"] = []
    experiment_results["out_reconstruction_error"] = []
    experiment_results["out_error"] = []
    experiment_results["gamma"] = []
    experiment_results["lambda"] = []
    experiment_results["execution_time"] = []

    if method_name in ["admm_sub", "admm_exact"]
        experiment_results["update_times"] = []
        experiment_results["step_size"] = []
        experiment_results["Phi_residual"] = []
        experiment_results["Psi_residual"] = []
        experiment_results["Phi_residual_hist"] = []
        experiment_results["Psi_residual_hist"] = []
        experiment_results["U_sol"] = []
        experiment_results["V_sol"] = []
        experiment_results["P_sol"] = []
        experiment_results["Z_sol"] = []
        experiment_results["Phi_sol"] = []
        experiment_results["Psi_sol"] = []
    end

    start_time = now()

    # Loop to execute specified experiment for each trial
    for trial_num=1:NUM_TRIALS

        println("Starting trial " * string(trial_num) * " of " * string(NUM_TRIALS))

        train_i = netflix_data[string(trial_num)]["train_i"]
        train_j = netflix_data[string(trial_num)]["train_j"]
        train_val = netflix_data[string(trial_num)]["train_val"]

        train_i = convert(Vector{Int64}, train_i)
        train_j = convert(Vector{Int64}, train_j)
        train_val = convert(Vector{Int64}, train_val)

        A_observed = sparse(train_i, train_j, train_val, n, m)

        test_i = netflix_data[string(trial_num)]["test_i"]
        test_j = netflix_data[string(trial_num)]["test_j"]
        test_val = netflix_data[string(trial_num)]["test_val"]

        test_i = convert(Vector{Int64}, test_i)
        test_j = convert(Vector{Int64}, test_j)
        test_val = convert(Vector{Int64}, test_val)

        gamma = 1 / length(train_val)
        #gamma = 1 / (m * n)
        lambda = 1 / length(train_val)
        #lambda = 0

        # Switch to execute the specified method
        if method_name == "admm_exact"
            step_size = 10
            trial_start = now()
            output = admm(A_observed, k_target, Y, lambda, gamma=gamma,
                          step_size=step_size, max_iteration=20,
                          residual_threshold=1e-4, initialization="exact",
                          P_update="exact")
            trial_end_time = now()
            U_fitted = output[1]
            V_fitted = output[2]
            P_fitted = output[3]
            Z_fitted = output[4]
            Phi_iterate = output[5]
            Psi_iterate = output[6]

            Phi_residual = Z_fitted - P_fitted * Z_fitted
            Psi_residual = Z_fitted - U_fitted

            append!(experiment_results["Phi_residual"], norm(Phi_residual)^2)
            append!(experiment_results["Psi_residual"], norm(Psi_residual)^2)

            append!(experiment_results["Phi_residual_hist"], [output[8][1]])
            append!(experiment_results["Psi_residual_hist"], [output[8][2]])

            append!(experiment_results["update_times"], [output[7][3]])
            append!(experiment_results["step_size"], step_size)
            append!(experiment_results["U_sol"], [U_fitted])
            append!(experiment_results["V_sol"], [V_fitted])
            append!(experiment_results["P_sol"], [P_fitted])
            append!(experiment_results["Z_sol"], [Z_fitted])
            append!(experiment_results["Phi_sol"], [Phi_iterate])
            append!(experiment_results["Psi_sol"], [Psi_iterate])
        elseif method_name == "admm_sub"
            step_size = 10
            trial_start = now()
            output = admm(A_observed, k_target, Y, lambda, gamma=gamma,
                          step_size=step_size, max_iteration=20,
                          residual_threshold=1e-4, initialization="exact",
                          P_update="sub_sketch")
            trial_end_time = now()
            U_fitted = output[1]
            V_fitted = output[2]
            append!(experiment_results["update_times"], [output[7][3]])
            append!(experiment_results["step_size"], step_size)
        elseif method_name == "fastImpute"
            trial_start = now()
            X_fitted = fastImpute(A_observed, k_target)
            trial_end_time = now()
        elseif method_name == "softImpute"
            trial_start = now()
            X_fitted = softImpute(A_observed, k_target)
            trial_end_time = now()
        end

        # Compute the performance measures of the returned solution
        elapsed_time = Dates.value(trial_end_time - trial_start)

        # Do in sample and out of sample evaluation using a threaded loop
        n_in = length(train_val)
        n_out = length(test_val)

        in_sample_error = zeros(n_in)
        in_TSS = zeros(n_in)
        out_sample_error = zeros(n_out)
        out_TSS = zeros(n_out)

        if method_name in ["admm_exact", "admm_sub"]
            Threads.@threads for i=1:n_in
                fitted_val = U_fitted[train_i[i], :]' * V_fitted[train_j[i], :]
                in_sample_error[i] = (train_val[i] - fitted_val) ^ 2
                in_TSS[i] = train_val[i] ^ 2
            end
            Threads.@threads for i=1:n_out
                fitted_val = U_fitted[test_i[i], :]' * V_fitted[test_j[i], :]
                out_sample_error[i] = (test_val[i] - fitted_val) ^ 2
                out_TSS[i] = test_val[i] ^ 2
            end
        else
            Threads.@threads for i=1:n_in
                fitted_val = X_fitted[train_i[i], train_j[i]]
                in_sample_error[i] = (train_val[i] - fitted_val) ^ 2
                in_TSS[i] = train_val[i] ^ 2
            end
            Threads.@threads for i=1:n_out
                fitted_val = X_fitted[test_i[i], test_j[i]]
                out_sample_error[i] = (test_val[i] - fitted_val) ^ 2
                out_TSS[i] = test_val[i] ^ 2
            end
        end

        in_error = sum(in_sample_error) / sum(in_TSS)
        out_error = sum(out_sample_error) / sum(out_TSS)

        # Store the performance measures of the returned solution
        append!(experiment_results["train_size"], n_in)
        append!(experiment_results["test_size"], n_out)
        append!(experiment_results["in_reconstruction_error"], in_error)
        append!(experiment_results["in_error"], sum(in_sample_error))
        append!(experiment_results["out_reconstruction_error"], out_error)
        append!(experiment_results["out_error"], sum(out_sample_error))
        append!(experiment_results["gamma"], gamma)
        append!(experiment_results["lambda"], lambda)
        append!(experiment_results["execution_time"], elapsed_time)

        print("Completed trial $trial_num of $NUM_TRIALS total trials.")

    end

    # Save the results to file
    f = open(output_path * "_" * string(index) * ".json","w")
    JSON.print(f, JSON.json(experiment_results))
    close(f)

    total_time = now() - start_time
    print("Total execution time: ")
    println(total_time)
end
