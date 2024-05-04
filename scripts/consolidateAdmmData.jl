using JSON, LinearAlgebra, Statistics, DataFrames, CSV

function processData(input_path, method_name)
   """
   This function loads raw experiment output data and processes it into a
   dataframe.
   """
   df = DataFrame(N=Int64[], M=Int64[], K=Int64[], D=Int64[],
                   missing_frac=Float64[], noise_param=Float64[],
                   gamma=Float64[], lambda=Float64[],
                   Z_exec_time=Float64[], Z_exec_time_std=Float64[],
                   U_exec_time=Float64[], U_exec_time_std=Float64[],
                   U_map_exec_time=Float64[], U_map_exec_time_std=Float64[],
                   U_reduce_exec_time=Float64[], U_reduce_exec_time_std=Float64[],
                   P_exec_time=Float64[], P_exec_time_std=Float64[],
                   V_exec_time=Float64[], V_exec_time_std=Float64[],
                   V_map_exec_time=Float64[], V_map_exec_time_std=Float64[],
                   V_reduce_exec_time=Float64[], V_reduce_exec_time_std=Float64[])


   if method_name in ["admmV0", "admm_exact", "admm_pqr"]
       select!(df, Not([:U_map_exec_time, :U_map_exec_time_std,
               :U_reduce_exec_time, :U_reduce_exec_time_std,
               :V_map_exec_time, :V_map_exec_time_std, :V_reduce_exec_time,
               :V_reduce_exec_time_std]))
   else
      select!(df, Not([:U_exec_time, :U_exec_time_std, :V_exec_time,
                       :V_exec_time_std]))
   end

   keySet = Dict()
   for method_name in ["admmV0", "admm_exact", "admm_pqr"]
      keySet[method_name] = ["Z", "U", "P", "V"]
   end
   #for method_name in ["admm_exact", "admm_pqr", "admm_pheig"]
   #   keySet[method_name] = ["Z", "U_map", "U_reduce",
   #                          "P", "V_map", "V_reduce"]
   #end

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
                     exp_data["noise"],
                     exp_data["gamma"][1],
                     exp_data["lambda"][1]]

      for key in keySet[method_name]
            update_data = [raw_data[key] for raw_data in exp_data["update_times"]]
            append!(current_row, Statistics.mean(update_data))
            append!(current_row, Statistics.std(update_data) / (num_samples^0.5))
      end

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
CSV.write(OUTPUT_ROOT * "_times_aggrData.csv", df1)
