using JSON, LinearAlgebra, Statistics, DataFrames, CSV

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

function processData(input_path, method_name, version)
    """
    This function loads raw experiment output data as a JSON and processes it
    into a dataframe.

    :param input_path: File path to raw experiment output data.
    :param method_name: Method name of raw experiment data to be processed.

    :return: A dataframe of the processed experiment data.
    """

    # Define the dataframe columns
    df = DataFrame(N=Int64[], M=Int64[], K=Int64[], D=Int64[],
                   missing_frac=Float64[], gamma=Float64[], lambda=Float64[],
                   in_reconstruction_error=Float64[], in_error_std=Float64[],
                   out_reconstruction_error=Float64[], out_error_std=Float64[],
                   exec_time=Float64[], exec_time_std=Float64[], r2=Float64[],
                   r2_std=Float64[], feat1=Float64[], feat1_std=Float64[],
                   feat2=Float64[], feat2_std=Float64[], feat3=Float64[],
                   feat3_std=Float64[], feat4=Float64[], feat4_std=Float64[],
                   feat5=Float64[], feat5_std=Float64[], feat6=Float64[],
                   feat6_std=Float64[])

    if version != "6Y"
        select!(df, Not([:feat5, :feat5_std, :feat6, :feat6_std]))
    end

    successful_entries = 0

    root_path = input_path * version * "/" * method_name * "/"
    file_paths = readdir(root_path, join=true)

    # Iterate over all files in the input directory
    for file_name in file_paths

        if file_name[end-4:end] != ".json"
            continue
        end

        # Load the data from the filesystem
        exp_data = Dict()
        open(file_name, "r") do f
            dicttxt = JSON.read(f, String)
            exp_data = JSON.parse(dicttxt)
            exp_data = JSON.parse(exp_data)
        end

        task_ID = file_name[end-6:end-5]
        if task_ID[1] == '_'
            task_ID = task_ID[2:end]
        end

        num_samples = length(exp_data["execution_time"])
        if num_samples == 0
            continue
        end

        # Extract and store the relevant data
        current_row = [exp_data["N"],
                       exp_data["M"],
                       exp_data["K"],
                       exp_data["D"],
                       exp_data["frac"],
                       exp_data["gamma"][1],
                       exp_data["lambda"][1]]

        for label in ["in_reconstruction_error", "out_reconstruction_error",
                      "execution_time", "r2"]
            new_data = exp_data[label]
            append!(current_row, Statistics.mean(new_data))
            append!(current_row, Statistics.std(new_data) / (num_samples^0.5))
        end

        Y = unserialize_matrix(exp_data["Y"])
        mean_vec = mean(Y, dims=1)
        println(mean_vec)
        println(Statistics.std(Y, dims = 1))
        println()
        r2_errors = zeros(num_samples, size(Y)[2])
        for i=1:num_samples
            predictions = unserialize_matrix(exp_data["predictions"][1])
            for j=1:size(Y)[2]
                y_vals = Y[:, j]
                pred_vals = predictions[:, j]
                r2 = 1 - (sum((y_vals - pred_vals) .^ 2) / sum((y_vals .- mean_vec[j]) .^ 2))
                r2_errors[i, j] = r2
            end
        end

        for j=1:size(Y)[2]
            append!(current_row, Statistics.mean(r2_errors[:, j]))
            append!(current_row, Statistics.std(r2_errors[:, j]) / (num_samples^0.5))
        end

        push!(df, current_row)
        successful_entries += 1

    end

    println("$successful_entries entries have been entered into the dataframe.")
    return df
end;

METHOD_NAME = ARGS[1]
INPUT_PATH = ARGS[2]
VERSION = ARGS[3]

OUTPUT_ROOT = INPUT_PATH * VERSION * "/" * METHOD_NAME * "/" * METHOD_NAME

# Process and save the data
df1 = processData(INPUT_PATH, METHOD_NAME, VERSION)
CSV.write(OUTPUT_ROOT * "_aggrData.csv", df1)
