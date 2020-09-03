using DataFrames
using CSV

function read_csv_file(filename)
    return DataFrame(CSV.File(filename))
end