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

function processData(input_path, method_name)
   """
   This function loads raw experiment output data as a JSON and processes it
   into a dataframe.

   :param input_path: File path to raw experiment output data.
   :param method_name: Method name of raw experiment data to be processed.

   :return: A dataframe of the processed experiment data.
   """

   # Define the dataframe columns
   df = DataFrame(N=Int64[], M=Int64[], K=Int64[], D=Int64[],
                  missing_frac=Float64[], noise_param=Float64[],
                  gamma=Float64[], lambda=Float64[],
                  reconstruction_error=Float64[],
                  reconstruction_error_std=Float64[], objective=Float64[],
                  objective_std=Float64[], fitted_rank=Float64[],
                  fitted_rank_std=Float64[], exec_time=Float64[],
                  exec_time_std=Float64[], r2=Float64[], r2_std=Float64[])

   successful_entries = 0

   root_path = input_path * method_name * "/"
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

      key = Dict{String, Real}(param_dict[string(task_ID)])
      experiment_data = synthetic_data[string(key)]

      # Extract and store the relevant data
      current_row = [exp_data["N"],
                     exp_data["M"],
                     exp_data["K"],
                     exp_data["D"],
                     exp_data["frac"],
                     exp_data["noise"],
                     exp_data["gamma"][1],
                     exp_data["lambda"][1],
                     Statistics.mean(exp_data["reconstruction_error"]),
                     Statistics.std(exp_data["reconstruction_error"]) / (num_samples^0.5),
                     Statistics.mean(exp_data["objective"]),
                     Statistics.std(exp_data["objective"]) / (num_samples^0.5),
                     Statistics.mean(exp_data["rank"]),
                     Statistics.std(exp_data["rank"]) / (num_samples^0.5),
                     Statistics.mean(exp_data["execution_time"][2:end]),
                     Statistics.std(exp_data["execution_time"][2:end]) / (num_samples^0.5)]

      r2_vals = []
      for i=1:num_samples
         X = unserialize_matrix(exp_data["solution"][i])
         Y = unserialize_matrix(experiment_data[string(i)]["Y"])

         beta = pinv(X'*X)*X'*Y
         preds = X * beta
         RSS = norm(Y - preds) ^ 2
         mean_vec = mean(Y, dims=1)
         TSS = 0
         for i=1:size(Y)[1]
             TSS += norm(Y[i, :] - mean_vec') ^ 2
         end
         r2 = 1 - (RSS / TSS)
         append!(r2_vals, r2)
      end

      append!(current_row, Statistics.mean(r2_vals))
      append!(current_row, Statistics.std(r2_vals) / (num_samples^0.5))

      push!(df, current_row)
      successful_entries += 1

   end

   println("$successful_entries entries have been entered into the dataframe.")
   return df
end;

METHOD_NAME = ARGS[1]
INPUT_PATH = ARGS[2]

OUTPUT_ROOT = INPUT_PATH * METHOD_NAME * "/" * METHOD_NAME

# Load the syntehtic data
synthetic_data = Dict()
open(INPUT_PATH * "data.json", "r") do f
    global synthetic_data
    dicttxt = JSON.read(f, String)  # file information to string
    synthetic_data = JSON.parse(dicttxt)  # parse and transform data
    synthetic_data = JSON.parse(synthetic_data)
end

# Load the experiment parameters
param_dict = Dict()
open(INPUT_PATH * "params.json", "r") do f
    global param_dict
    dicttxt = JSON.read(f, String)  # file information to string
    param_dict = JSON.parse(dicttxt)  # parse and transform data
    param_dict = JSON.parse(param_dict)
end

# Process and save the data
df1 = processData(INPUT_PATH, METHOD_NAME)
CSV.write(OUTPUT_ROOT * "_aggrData.csv", df1)
