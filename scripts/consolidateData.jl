using JSON, LinearAlgebra, Statistics, DataFrames, CSV

function processData(input_path, method_name)
   """
   This function loads raw experiment output data and processes it into a
   dataframe.
   """
   df = DataFrame(N=Int64[], M=Int64[], K=Int64[], D=Int64[],
                  missing_frac=Float64[], noise_param=Float64[],
                  gamma=Float64[], lambda=Float64[],
                  reconstruction_error=Float64[],
                  reconstruction_error_std=Float64[], objective=Float64[],
                  objective_std=Float64[], fitted_rank=Float64[],
                  fitted_rank_std=Float64[], exec_time=Float64[],
                  exec_time_std=Float64[])

   successful_entries = 0

   root_path = input_path * method_name * "/"

   file_paths = readdir(root_path, join=true)
   # Iterate over all files in the input directory
   for file_name in file_paths

      if file_name[end-4:end] != ".json"
         continue
      end

      exp_data = Dict()
      open(file_name, "r") do f
         dicttxt = JSON.read(f, String)
         exp_data = JSON.parse(dicttxt)
         exp_data = JSON.parse(exp_data)
      end

      task_ID = file_name[end-6:end-5]
      if task_ID[1] == '_'
          task_ID = task_ID[2:end]
      end;

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

      push!(df, current_row)
      successful_entries += 1

   end

   println("$successful_entries entries have been entered into the dataframe.")
   return df
end;

METHOD_NAME = ARGS[1]
INPUT_PATH = ARGS[2]

OUTPUT_ROOT = INPUT_PATH * METHOD_NAME * "/" * METHOD_NAME

# Process and save the data
df1 = processData(INPUT_PATH, METHOD_NAME)
CSV.write(OUTPUT_ROOT * "_aggrData.csv", df1)
