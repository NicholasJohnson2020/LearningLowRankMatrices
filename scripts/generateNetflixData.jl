using SparseArrays, NPZ, JSON, StatsBase, Random, Dates, Distributions

function filterData(Y);
    """
    This function filters out movies that do not have all side information
    present and filters out users that do not have at least 5 ratings across
    the retained movies.

    :param Y: A n-by-d matrix of side information.

    :return: This function returns four values. (1) A list of row indices of the
             retained matrix entries, (2) a list of column indices of the
             retained matrix entries, (3) a list of values of the retained
             matrix entries and (4) the filtered n-by-d matrix of side
             information.
    """

    # Filter Y to only keep indices that have values across all columns
    good_indexes = [x[1] for x in findall(==(0), sum(Y .== 0, dims=2))]
    Y = Y[good_indexes, :]

    netflixID_map = netflixID_map_raw[good_indexes]

    raw_i = processed_data["i"]
    raw_j = processed_data["j"]
    raw_val = processed_data["val"]

    num_elements_raw = length(raw_i)
    partial_filtered_i = zeros(Int16, num_elements_raw)
    partial_filtered_j = zeros(Int32, num_elements_raw)
    partial_filtered_val = zeros(Int8, num_elements_raw)

    # Filter out movies for which we do not have features
    filtered_index = 1
    for index=1:num_elements_raw
        if raw_i[index] in netflixID_map
            partial_filtered_i[filtered_index] = raw_i[index]
            partial_filtered_j[filtered_index] = raw_j[index]
            partial_filtered_val[filtered_index] = raw_val[index]
            filtered_index += 1
        end
    end
    partial_filtered_i = partial_filtered_i[1:(filtered_index-1)]
    partial_filtered_j = partial_filtered_j[1:(filtered_index-1)]
    partial_filtered_val = partial_filtered_val[1:(filtered_index-1)]

    num_elements_partial = length(partial_filtered_i)
    filtered_i = zeros(Int16, num_elements_partial)
    filtered_j = zeros(Int32, num_elements_partial)
    filtered_val = zeros(Int8, num_elements_partial)

    # Filter out users who have rated less than 5 movies
    counts = countmap(partial_filtered_j)
    filtered_index = 1
    for index=1:num_elements_partial
        if counts[partial_filtered_j[index]] >= 5
            filtered_i[filtered_index] = partial_filtered_i[index]
            filtered_j[filtered_index] = partial_filtered_j[index]
            filtered_val[filtered_index] = partial_filtered_val[index]
            filtered_index += 1
        end
    end
    filtered_i = filtered_i[1:(filtered_index-1)]
    filtered_j = filtered_j[1:(filtered_index-1)]
    filtered_val = filtered_val[1:(filtered_index-1)]

    reverseNetflixIDMap = Dict()
    for (index, val) in enumerate(netflixID_map)
        reverseNetflixIDMap[val] = index
    end
    i_data = zeros(Int16, length(filtered_i))
    for (index, val) in enumerate(filtered_i)
        i_data[index] = Int16(reverseNetflixIDMap[val])
    end

    userIDs = unique(filtered_j)
    reverseUserIDMap = Dict()
    for (index, val) in enumerate(userIDs)
        reverseUserIDMap[val] = index
    end
    j_data = zeros(Int32, length(filtered_j))
    for (index, val) in enumerate(filtered_j)
        j_data[index] = Int32(reverseUserIDMap[val])
    end

    return i_data, j_data, filtered_val, Y

end;


function performMasking(i_data, j_data, val_data, test_frac);
    """
    This function creates a random partition of training and test data.

    :param i_data: A list of row indices of the preprocessed matrix entries.
    :param j_data: A list of column indices of the preprocessed matrix entries.
    :param val_data: A list of values of the preprocessed matrix entries.
    :param test_frac: The fraction of data to be used as test data (Float64).

    :return: This function returns two values. (1) A tuple of length 3 where
             the first entry is a list of row indices for the training data,
             the second entry is a list of column indices for the training data
             and the third is a list of values for the training data. (2) A
             tuple of length 3 where the first entry is a list of row indices
             for the test data, the second entry is a list of column indices
             for the test data and the third is a list of values for the test
             data.
    """

    num_examples = length(i_data)

    train_i = zeros(Int16, num_examples)
    train_j = zeros(Int32, num_examples)
    train_val = zeros(Int8, num_examples)

    test_i = zeros(Int16, num_examples)
    test_j = zeros(Int32, num_examples)
    test_val = zeros(Int8, num_examples)

    train_index = 1
    test_index = 1
    for data_index=1:num_examples
        if rand(Uniform(0, 1)) < test_frac
            # Add to test data
            test_i[test_index] = i_data[data_index]
            test_j[test_index] = j_data[data_index]
            test_val[test_index] = val_data[data_index]
            test_index += 1
        else
            # Add to train data
            train_i[train_index] = i_data[data_index]
            train_j[train_index] = j_data[data_index]
            train_val[train_index] = val_data[data_index]
            train_index += 1
        end
    end

    return (train_i[1:(train_index-1)], train_j[1:(train_index-1)], train_val[1:(train_index-1)]),
           (test_i[1:(test_index-1)], test_j[1:(test_index-1)], test_val[1:(test_index-1)])
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

data_dict["Configs"] = length(K)
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
    Y_col_names = Y_colnames_raw[3:end]
    output_path = ROOT_PATH * "4Y_"
else
    Y = Y_raw
    Y_col_names = Y_colnames_raw
    output_path = ROOT_PATH * "6Y_"
end

start = now()

i_data, j_data, val_data, Y_data = filterData(Y)


processed_data = nothing
GC.gc()

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
end

data_dict["Y"] = Y_data
for trial_num=1:NUM_TRIALS_PER_CONFIG
    # sample the data
    train_data, test_data = performMasking(i_data, j_data, val_data, test_frac)
    train_i, train_j, train_val = train_data
    test_i, test_j, test_val = test_data
    # store in dataframe at correct location
    data_dict[trial_num] = Dict("train_i"=>train_i,
                                "train_j"=>train_j,
                                "train_val"=>train_val,
                                "test_i"=>test_i,
                                "test_j"=>test_j,
                                "test_val"=>test_val)
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
