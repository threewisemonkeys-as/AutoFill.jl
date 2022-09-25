####################################
# DAG Representation of Substrings #
####################################


const Node = Vector{Int}
const Path = Vector{Node}
# TODO: Use NamedTuple or UnitRange here -
const Edge = Pair{Node, Node}
const EdgeExprMap = Dict{Edge, Set{<:Op}}

struct DAG
    η::Set{Node}
    ηˢ::Node
    ηᵗ::Node
    ξ::Set{Edge}
    W::EdgeExprMap
end

import Base.isempty
import Base.intersect

isempty(dag::DAG) = isempty(dag.η)

function intersect(d1::DAG, d2::DAG)
    nodes = Set{Node}()
    source = [d1.ηˢ; d2.ηˢ]
    target = [d1.ηᵗ; d2.ηᵗ]
    edges = Set{Edge}()
    edge_expr_map = EdgeExprMap()
    for ((d1_ηₛ, d1_ηₜ), (d2_ηₛ, d2_ηₜ)) in Iterators.product(
        keys(d1.W), keys(d2.W)
    )
        result = intersect(
            d1.W[d1_ηₛ => d1_ηₜ], d2.W[d2_ηₛ => d2_ηₜ]
        )
        if !isempty(result)
            n1, n2 = [d1_ηₛ; d2_ηₛ], [d1_ηₜ; d2_ηₜ]
            push!(nodes, n1)
            push!(nodes, n2)
            push!(edges, n1 => n2)
            edge_expr_map[n1 => n2] = result
        end
    end

    return DAG(nodes, source, target, edges, edge_expr_map)
end


function unify(d1::DAG, d2::DAG, w_id::String, unification_cache::Dict{Edge, Set{<:Op}})
    nodes = Set{Node}()
    source = [d1.ηˢ; d2.ηˢ]
    target = [d1.ηᵗ; d2.ηᵗ]
    edges = Set{Edge}()
    edge_expr_map = EdgeExprMap()
    t1 = length(d1.W) * length(d2.W)
    c1 = 1

    # TODO: pre-compute hashes
    # TODO: Only unify for equal length substrings

    for ((d1_ηₛ, d1_ηₜ), (d2_ηₛ, d2_ηₜ)) in Iterators.product(
        keys(d1.W), keys(d2.W)
    )
        # println(
        #     c1 += 1, "/", t1, " - ",
        #     d1_ηₛ => d1_ηₜ, " ", d2_ηₛ => d2_ηₜ, " ",
        #     length(d1.W[d1_ηₛ => d1_ηₜ]), " ", length(d2.W[d2_ηₛ => d2_ηₜ]), " ",
        #     length(d1.W[d1_ηₛ => d1_ηₜ]) * length(d2.W[d2_ηₛ => d2_ηₜ])
        # )

        n1, n2 = [d1_ηₛ; d2_ηₛ], [d1_ηₜ; d2_ηₜ]
        if haskey(unification_cache,  n1 => n2)
            push!(nodes, n1)
            push!(nodes, n2)
            push!(edges, n1 => n2)
            edge_expr_map[n1 => n2] = unification_cache[n1 => n2]
        else
            result = Set{Op}()
            p_dict = Dict{UInt64, Set{Op}}()

            for p1 in d1.W[d1_ηₛ => d1_ηₜ]
                p_hash = loop_invariant_hash(p1)
                if !haskey(p_dict, p_hash)
                    p_dict[p_hash]  = Set{Op}()
                end
                push!(p_dict[p_hash], p1)
            end

            for p2 in d2.W[d2_ηₛ => d2_ηₜ]
                p_hash = loop_invariant_hash(p2)
                if haskey(p_dict, p_hash)
                    for p1 in p_dict[p_hash]
                        u = unify(p1, p2, w_id)
                        if !isnothing(u) push!(result, u) end
                    end
                end
            end 

            # for (p1, p2) in Iterators.product(
            #     d1.W[d1_ηₛ => d1_ηₜ], d2.W[d2_ηₛ => d2_ηₜ]
            # )
            #     u = unify(p1, p2, w_id)
            #     if !isnothing(u) push!(result, u) end
            # end

            if !isempty(result)
                push!(nodes, n1)
                push!(nodes, n2)
                push!(edges, n1 => n2)
                edge_expr_map[n1 => n2] = result
            end
        end
    end
    
    return DAG(nodes, source, target, edges, edge_expr_map)
end

outgoing_edges(dag::DAG, node::Node) = [
    n1 => n2 for (n1, n2) in dag.ξ if n1 == node
]

incoming_edges(dag::DAG, node::Node) = [
    n1 => n2 for (n1, n2) in dag.ξ if n2 == node
]

function dfs_forward_from(dag::DAG, node::Node) 
    if node == dag.ηᵗ return [node] end
    for edge in outgoing_edges(dag, node)
        suf = sample_forward_from(dag, edge[2])
        if !isnothing(suf) return vcat([node], suf) end
    end
    return nothing
end

function dfs_backward_from(dag::DAG, node::Node) 
    if node == dag.ηˢ return [node] end
    for edge in incoming_edges(dag, node)
        pre = dfs_backward_from(dag, edge[1])
        if !isnothing(pre) return vcat(pre, [node]) end
    end
    return nothing
end

function package(path::Union{Path, Nothing}, dag::DAG)
    return if isnothing(path) nothing
    elseif length(path) == 2 rand(dag.W[path[1] => path[2]])
    else Concatenate([
        rand(dag.W[path[i] => path[i+1]]) for i in 1:length(path) - 1
    ])
    end
end

sample(dag::DAG) = package(dfs_backward_from(dag, dag.ηᵗ), dag)


# function Base.iterate(dag::DAG) 
#     q = [[dag.ηᵗ]]
#     while true
#         path = pop!(q)
#         node = path[length(traj)]
#         if node == dag.ηˢ return package(path) end
#         for edge in incoming_edges(dag, node)
#             push!(q, vcat)  
#     end
# end

# TODO: maybe use ResumableFunctions.jl or proper state definition
function dfs_backward_itr_from(dag::DAG, node::Node)
    if node == dag.ηˢ return [[node]] end
    paths = Vector{Path}()
    for edge in incoming_edges(dag, node)
        pres = dfs_backward_itr_from(dag, edge[1])
        if !isnothing(pres) append!(paths, pres) end
    end
    return !isempty(paths) ? vcat.(paths, [[node]]) : nothing
end

# function sample_from_path(path, dag)
#     program_path = [
#         dag.W[path[i] => path[i+1]] for i in 1:length(path) - 1
#     ]

# end

# function sample_programs(dag)

# end

function unroll_path(path, dag)
    program_path = [
        dag.W[path[i] => path[i+1]] for i in 1:length(path) - 1
    ]
    return [
        length(p) == 1 ? first(p) : Concatenate([p...])
        for p in collect(Iterators.product(program_path...))
    ]
end

function extract_programs(dag::DAG)
    paths = dfs_backward_itr_from(dag, dag.ηᵗ)
    if isnothing(paths) return nothing end
    return reduce(vcat, [unroll_path(p, dag) for p in paths])    
end

function Base.iterate(dag::DAG)
    programs = extract_programs(dag)
    return (programs[1], (programs=programs, count=2))
end

Base.iterate(
    dag::DAG,
    state
) = return state.count > length(state.programs) ? nothing : (
    state.programs[state.count],
    (programs=state.programs, count=state.count + 1)
)
