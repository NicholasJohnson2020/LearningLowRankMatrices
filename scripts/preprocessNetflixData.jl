### Run this on cluster! Takes up too much memory on local machine
using JSON

ROOT_PATH = "~/research/data/low-rank-learning/netflix/netflix_data/"

files = ["combined_data_1.txt",
         "combined_data_2.txt",
         "combined_data_3.txt",
         "combined_data_4.txt"]

num_movies = 17770
num_users = 480189

userIndexIdMap = zeros(num_users)
userIdIndexMap = Dict()
i_index = []
j_index = []
val_index = []

for this_file in files
    open(ROOT_PATH * this_file) do f

        # line_number
        line = 0
        movie_index = 0

        # read till end of file (edit this line out when running on the cluster)
        while (! eof(f))

            # read a new / next line for every iteration
            s = readline(f)

            if s[end] == ':'
                movie_index = parse(Int, s[1:end-1])
                if movie_index % 500 == 1
                    println("Processing movie $movie_index")
                end
            else
                raw_data = split(s, ",")
                user_id = parse(Int, raw_data[1])
                rating = parse(Int, raw_data[2])

                if !haskey(userIdIndexMap, user_id)
                    new_user_index = length(userIdIndexMap) + 1
                    userIndexIdMap[new_user_index] = user_id
                    userIdIndexMap[user_id] = new_user_index
                end
                append!(i_index, Int16(movie_index))
                append!(j_index, Int32(userIdIndexMap[user_id]))
                append!(val_index, Int8(rating))
            end
        end
    end
end

println()
println("Processed $movie_index total movies.")

netflix_data = Dict()
netflix_data["userIndexIdMap"] = userIndexIdMap
netflix_data["userIdIndexMap"] = userIdIndexMap
netflix_data["i"] = i_index
netflix_data["j"] = j_index
netflix_data["val"] = val_index

f = open(ROOT_PATH * "processed_lists.json", "w")
JSON.print(f, JSON.json(netflix_data))
close(f)
