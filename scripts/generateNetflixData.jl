using SparseArrays, NPZ, JSON, StatsBase, Random, Dates

function filterData(Y);
    # filter Y to only keep indices that have values across all columns
    good_indexes = [x[1] for x in findall(==(0), sum(Y .== 0, dims=2))]
    Y = Y[good_indexes, :]

    netflixID_map = netflixID_map_raw[good_indexes]

    raw_i = processed_data["i"]
    raw_j = processed_data["j"]
    raw_val = processed_data["val"]

    partial_filtered_i = []
    partial_filtered_j = []
    partial_filtered_val = []

    # Filter out movies for which we do not have features
    num_elements_raw = length(raw_i)
    for index=1:num_elements_raw
        if raw_i[index] in netflixID_map
            append!(partial_filtered_i, raw_i[index])
            append!(partial_filtered_j, raw_j[index])
            append!(partial_filtered_val, raw_val[index])
        end
    end

    filtered_i = []
    filtered_j = []
    filtered_val = []

    # Filter out users who have rated less than 5 movies
    counts = countmap(partial_filtered_j)
    num_elements_partial = length(partial_filtered_i)
    for index=1:num_elements_partial
        if counts[partial_filtered_j[index]] >= 5
            append!(filtered_i, partial_filtered_i[index])
            append!(filtered_j, partial_filtered_j[index])
            append!(filtered_val, partial_filtered_val[index])
        end
    end

    return filtered_i, filtered_j, filtered_val

end;


function performMasking(i_data, j_data, val_data, test_frac);

    num_examples = length(i_data)
    test_indices = sort(sample(1:num_examples, Int(floor(test_frac * num_examples)), replace = false))
    train_indices = [i for i=1:num_examples if !(i in test_indices)]

    return (i_data[train_indices], j_data[train_indices], val_data[train_indices]),
           (i_data[test_indices], j_data[test_indices], val_data[test_indices])
end;


version_mode = parse(Int64, ARGS[1])
@assert version_mode in [1, 2]

# version_mode 1 corresponds to 4 Y features
# version_mode 2 corresponds to 6 Y features

NUM_TRIALS_PER_CONFIG = 5

ROOT_PATH = "../../../data/low-rank-learning/netflix/netflix_data/"

K = [3, 4, 5, 6, 7, 8, 9, 10]

test_frac = 0.2

data_dict = Dict()
param_dict = Dict()

data_dict["Trials"] = NUM_TRIALS_PER_CONFIG
data_dict["Test Frac"] = test_frac

processed_data = Dict()
open(ROOT_PATH * "processed_lists.json", "r") do f
    global processed_data
    dicttxt = JSON.read(f, String)
    processed_data = JSON.parse(dicttxt)
    processed_data = JSON.parse(processed_data)
end

Y_raw = npzread(ROOT_PATH * "Y.npy")
netflixID_map_raw = npzread(ROOT_PATH * "netflixID_map.npy")
Y_colnames_raw = ["budget", "revenue", "popularity",
                  "vote_average", "vote_count", "runtime"]

if version_mode == 1
    Y = Y_raw[:, 3:end]
    Y_colnames = Y_colnames_raw[3:end]
    output_path = ROOT_PATH * "4Y_"
else
    Y = Y_raw
    Y_col_names = Y_colnames_raw
    output_path = ROOT_PATH * "6Y_"
end

start = now()

i_data, j_data, val_data = filterData(Y)

# Extract N, M, D, missing_frac
n = length(unique(i_data))
m = length(unique(j_data))
missing_frac = 1 - length(val_data) / (n * m)
data_dict["n"] = n
data_dict["m"] = m
data_dict["d"] = size(Y)[2]
data_dict["Y_cols"] = Y_col_names
data_dict["missing"] = missing_frac

for (index, k) in enumerate(K)
    param_dict[index] = Dict("N"=>n, "M"=>m, "K"=>k, "D"=>size(Y)[2],
                                    "frac"=>missing_frac)
    data_dict[string(param_dict[index])] = Dict()
end

for trial_num=1:NUM_TRIALS_PER_CONFIG
    # sample the data
    train_data, test_data = performMasking(i_data, j_data, val_data, test_frac)
    A_observed = sparse(train_data[1], train_data[2], train_data[3])
    test_i, test_j, test_val = test_data
    for (index, k) in enumerate(K)
        # store in dataframe at correct location
        data_dict[string(param_dict[index])][trial_num] = Dict("A_observed"=>A_observed,
                                                               "Y"=>Y,
                                                               "k"=>k,
                                                               "test_i"=>test_i,
                                                               "test_j"=>test_j,
                                                               "test_val"=>test_val)
    end
end

elapsed_time = now() - start

f = open(output_path * "data.json","w")
JSON.print(f, JSON.json(data_dict))
close(f)

f = open(output_path * "params.json","w")
JSON.print(f, JSON.json(param_dict))
close(f)

println("Execution time is $elapsed_time")
println()
println("Column names:")
println(Y_col_names)
println("n: $n")
println("m: $n")
println("missing frac: $missing_frac")
