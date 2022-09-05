# Implementation of Flashfill algorithm
# Paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/popl11-synthesis.pdf
# Only impelemented a small subset of the whole algorithm
# Just enough to produce substring programs.

# Used following implementations as reference
# Scala: https://github.com/MikaelMayer/StringSolver
# JavaScript: https://github.com/bijection/robosheets

module AutoFill

include("token.jl")
include("lang.jl")
include("dag.jl")
include("generate.jl")

############################################
# Use `generate_substrings` with DataFrame #
############################################

# TODO: find good way to get best program
choose_top_program(p::Vector{<:Op}) = p[1]

using Tables
using DataFrames

"Fills rows of specified column that contain `missing` with suitable values"
function autofill(x::DataFrame, col)
    filled = filter(col => !ismissing, x)
    to_fill = filter(col => ismissing, x)

    # specs = collect(zip(
    #     collect(zip(
    #         Tables.getcolumn(filled, i) 
    #         for i in Tables.columnnames(filled) if i != col
    #     )),
    #     Tables.getcolumn(filled, col)
    # ))    
    specs = [i for i in zip(
        Vector.(eachrow(select(filled, Not(col)))),
        Tables.matrix(select(x, col))
    )]
    # inputs = collect(zip(
    #     Tables.getcolumn(to_fill, i) 
    #     for i in Tables.columnnames(to_fill) if i != col
    # ))
    inputs = Vector.(eachrow(select(to_fill, Not(col))))

    programs = generate_programs(specs)
    if isempty(programs)
        return nothing, nothing
    end
    p = choose_top_program(programs)
    outputs = [invoke(p, i) for i in inputs]

    filled = copy(x)
    filled[ismissing.(x[!, col]), col] = outputs
    return filled, p
end

end
