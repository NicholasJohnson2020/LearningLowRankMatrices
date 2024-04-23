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

cross_validate_step_size = true
cv_samples = 5

method_name = ARGS[1]
input_path = ARGS[2]
output_path = ARGS[3] * method_name * "/"
task_ID_input = parse(Int64, ARGS[4])
num_tasks_input = parse(Int64, ARGS[5])

valid_methods = ["ScaledGD", "VanillaGD", "admm", "admmMap",
                 "fastImpute", "softImpute", "SVD", "MF"]

@assert method_name in valid_methods

# Load the syntehtic data
synthetic_data = Dict()
open(input_path * "data.json", "r") do f
    global synthetic_data
    dicttxt = JSON.read(f, String)  # file information to string
    synthetic_data = JSON.parse(dicttxt)  # parse and transform data
    synthetic_data = JSON.parse(synthetic_data)
end

# Load the experiment parameters
param_dict = Dict()
open(input_path * "params.json", "r") do f
    global param_dict
    dicttxt = JSON.read(f, String)  # file information to string
    param_dict = JSON.parse(dicttxt)  # parse and transform data
    param_dict = JSON.parse(param_dict)
end

NUM_TRIALS = synthetic_data["Trials"]

task_ID_list = collect((task_ID_input+1):num_tasks_input:length(param_dict))

start_time_global = now()

# Main loop to execute experiments
for task_ID in task_ID_list

    global synthetic_data
    global param_dict

    # Load experiment specific data
    n = param_dict[string(task_ID)]["N"]
    m = param_dict[string(task_ID)]["M"]
    k_target = param_dict[string(task_ID)]["K"]
    d = param_dict[string(task_ID)]["D"]
    missing_frac = param_dict[string(task_ID)]["frac"]
    noise = param_dict[string(task_ID)]["noise"]

    key = Dict{String, Real}(param_dict[string(task_ID)])
    experiment_data = synthetic_data[string(key)]

    param_dict = nothing
    synthetic_data = nothing
    GC.gc()

    # Create dictionary to store experiment results
    experiment_results = Dict()
    experiment_results["Method"] = method_name
    experiment_results["N"] = n
    experiment_results["M"] = m
    experiment_results["K"] = k_target
    experiment_results["D"] = d
    experiment_results["frac"] = missing_frac
    experiment_results["noise"] = noise
    experiment_results["Trials"] = NUM_TRIALS

    experiment_results["solution"] = []
    experiment_results["reconstruction_error"] = []
    experiment_results["objective"] = []
    experiment_results["rank"] = []
    experiment_results["gamma"] = []
    experiment_results["lambda"] = []
    experiment_results["execution_time"] = []

    if method_name in ["admm", "admmMap"]
        experiment_results["update_times"] = []
        experiment_results["step_size"] = []
    end

    if method_name in ["ScaledGD", "VanillaGD"]
        experiment_results["iterations"] = []
        experiment_results["step_size"] = []
    end

    start_time = now()

    # Loop to execute specified experiment for each trial
    for trial_num=1:NUM_TRIALS

        println("Starting trial " * string(trial_num) * " of " * string(NUM_TRIALS))

        A_true = experiment_data[string(trial_num)]["A_true"]
        A_true = unserialize_matrix(A_true)
        A_observed = experiment_data[string(trial_num)]["A_observed"]
        A_observed = unserialize_matrix(A_observed)
        Y = experiment_data[string(trial_num)]["Y"]
        Y = unserialize_matrix(Y)

        gamma = 1 / 10
        lambda = 1 / 100

        # Switch to execute the specified method
        if method_name == "ScaledGD"
            step_size = 1
            if cross_validate_step_size
                cv_output = cross_validate(scaledGD, A_observed, k_target, Y,
                                           lambda, gamma,
                                           num_samples=cv_samples)
                step_size = cv_output[1]
            end
            trial_start = now()
            U_fitted, V_fitted, _, iterations = scaledGD(A_observed, k_target,
                                                         Y, lambda, gamma=gamma,
                                                         min_improvement=1e-3,
                                                         step_size=step_size)
            trial_end_time = now()
            X_fitted = U_fitted * V_fitted'
            append!(experiment_results["iterations"], iterations)
            append!(experiment_results["step_size"], step_size)
        elseif method_name == "VanillaGD"
            step_size = 1
            if cross_validate_step_size
                cv_output = cross_validate(vanillaGD, A_observed, k_target, Y,
                                           lambda, gamma,
                                           num_samples=cv_samples)
                step_size = cv_output[1]
            end
            trial_start = now()
            start = now()
            U_fitted, V_fitted, _, iterations = vanillaGD(A_observed, k_target,
                                                          Y, lambda, gamma=gamma,
                                                          min_improvement=1e-3,
                                                          step_size=step_size)
            trial_end_time = now()
            X_fitted = U_fitted * V_fitted'
            append!(experiment_results["iterations"], iterations)
            append!(experiment_results["step_size"], step_size)
        elseif method_name == "admm"
            step_size = 10
            if cross_validate_step_size
                cv_output = cross_validate(admm, A_observed, k_target, Y,
                                           lambda, gamma)
                step_size = cv_output[1]
            end
            trial_start = now()
            output = admm(A_observed, k_target, Y, lambda, gamma=gamma,
                          step_size=step_size, max_iteration=20,
                          residual_threshold=1e-4)
            trial_end_time = now()
            X_fitted = output[1] * output[2]'
            append!(experiment_results["updates_times"], [output[7][3]])
            append!(experiment_results["step_size"], step_size)
        elseif method_name == "admmMap"
            step_size = 10
            if cross_validate_step_size
                cv_output = cross_validate(admmMap, A_observed, k_target, Y,
                                           lambda, gamma)
                step_size = cv_output[1]
            end
            trial_start = now()
            output = admmMap(A_observed, k_target, Y, lambda, gamma=gamma,
                             step_size=step_size, max_iteration=20,
                             residual_threshold=1e-4)
            trial_end_time = now()
            X_fitted = output[1] * output[2]'
            append!(experiment_results["updates_times"], [output[7][3]])
            append!(experiment_results["step_size"], step_size)
        elseif method_name == "fastImpute"
            trial_start = now()
            X_fitted = fastImpute(A_observed, k_target)
            trial_end_time = now()
        elseif method_name == "softImpute"
            trial_start = now()
            X_fitted = softImpute(A_observed, k_target)
            trial_end_time = now()
        elseif method_name == "SVD"
            trial_start = now()
            X_fitted = iterativeSVD(A_observed, k_target)
            trial_end_time = now()
        elseif method_name == "MF"
            trial_start = now()
            X_fitted = matrixFactorization(A_observed, k_target)
            trial_end_time = now()
        end

        # Compute the performance measures of the returned solution
        objective, MSE = evaluatePerformance(X_fitted, A_observed, A_true, Y,
                                             k_target, lambda, gamma)
        reconstruction_error = MSE ^ 2 / norm(A_true) ^ 2
        fitted_rank = rank(matrixF_X)
        elapsed_time = Dates.value(trial_end_time - trial_start)

        # Store the performance measures of the returned solution
        append!(experiment_results["solution"], [X_fitted])
        append!(experiment_results["reconstruction_error"], reconstruction_error)
        append!(experiment_results["objective"], objective)
        append!(experiment_results["rank"], fitted_rank)
        append!(experiment_results["gamma"], gamma)
        append!(experiment_results["lambda"], lambda)
        append!(experiment_results["execution_time"], elapsed_time)

        print("Completed trial $trial_num of $NUM_TRIALS total trials.")

    end

    # Save the results to file
    f = open(output_path * "_" * string(task_ID) * ".json","w")
    JSON.print(f, JSON.json(experiment_results))
    close(f)

    total_time = now() - start_time
    print("Total execution time: ")
    println(total_time)
end
