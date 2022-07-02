# Implementation of Flashfill algorithm
# Paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/popl11-synthesis.pdf
# Only impelemented a small subset of the whole algorithm
# Just enough to produce substring programs.

# Used following implementations as reference
# Scala: https://github.com/MikaelMayer/StringSolver
# JavaScript: https://github.com/bijection/robosheets


module FlashFill

###################
# Lagugage Tokens #
###################


# We use the following collection of character classes 𝐶: 
# - Numeric Digits (0-9), 
# - Alphabets (a-zA-Z), 
# - Lowercase alphabets (a-z), 
# - Uppercase alphabets (A-Z), 
# - Accented alphabets, 
# - Alphanumeric characters, 
# - Whitespace characters, 
# - All characters. 
# We use the following SpecialTokens:
# - StartTok: Matches the beginning of a string. 
# - EndTok: Matches the end of a string.
# - A token for each special character, 
#   such as hyphen, dot, semicolon, colon, 
#   comma, backslash, forwardslash, 
#   left/right parenthesis/bracket etc.


# For better readability, we reference tokens by representative names. 
# For example, AlphTok refers to a sequence of alphabetic characters,
# NumTok refers to a sequence of numeric digits, NonDigitTok refers to 
# a sequence of characters that are not numeric digits, HyphenTok matches 
# with the hyphen character.

@enum Token begin
    AlphNumTok
    AlphNumWsTok
    AlphTok
    BackSlashTok
    ColonTok
    CommaTok
    DashTok
    DotTok
    EndTok
    HyphenTok
    LeftAngleTok
    LeftParenTok
    LeftSquareTok
    UnderscoreTok
    LowerTok
    NonAlphNumTok
    NonAlphNumWsTok
    NonAlphTok
    NonLowerTok
    NonNumTok
    NonUpperTok
    NumTok
    RightAngleTok
    RightParenTok
    RightSquareTok
    SemicolonTok
    SlashTok
    StartTok
    UpperTok
    WsTok
end

const TokenSet = Set{Token}
const TokenSeq = Vector{Token}

const token_to_r_string = Dict(
    AlphNumTok => "[a-zA-Z0-9]+",
    AlphNumWsTok => "[a-zA-Z0-9 ]+",
    AlphTok => "[a-zA-Z]+",
    BackSlashTok => "\\\\",
    ColonTok => ":",
    CommaTok => ",",
    DashTok => "-",
    DotTok => "\\.",
    EndTok => "\$",
    HyphenTok => "-",
    LeftAngleTok => "<",
    LeftParenTok => "\\(",
    LeftSquareTok => "<",
    UnderscoreTok => "_",
    LowerTok => "[a-z]+",
    NonAlphNumTok => "[^a-zA-Z0-9]+",
    NonAlphNumWsTok => "[^a-zA-Z0-9 ]+",
    NonAlphTok => "[^a-zA-Z]+",
    NonLowerTok => "[^a-z]+",
    NonNumTok => "[^\\d]+",
    NonUpperTok => "[^A-Z]+",
    NumTok => "\\d+",
    RightAngleTok => ">",
    RightParenTok => "\\)",
    RightSquareTok => ">",
    SemicolonTok => ";",
    SlashTok => "\\/",
    StartTok => "^",
    UpperTok => "[A-Z]+",
    WsTok => " ",
)

tokseq_to_r_string(tokseq) = join([
    token_to_r_string[tok] for tok in tokseq
])

const r_string_to_token = Dict(
    r_str => tok for (tok, r_str) in pairs(token_to_r_string) 
)

const token_to_r = Dict(
    tok => Regex(r_str) for (tok, r_str) in pairs(token_to_r_string) 
)


# # Uncomment below to pretty-print regex
# Base.show(io::IO, tok::Token) = print(io, token_to_r_string[tok])
# Base.show(io::IO, tokseq::TokenSeq) = print(
#     io,
#     join([token_to_r_string[tok] for tok in tokseq])
# )



######################
# Language Semantics #
######################

# Input state to program is list of strings
const ProgramState = Vector{<:AbstractString}

# TODO: NamedTuple
# Specifications are pairs of ProgramStates and output strings
const ProgramSpec = Vector{<:Tuple{ProgramState, AbstractString}}


# Syntax (Limited subset of original) -
# Trace expr e  := Concatenate(f₁, …, fₙ)
# Atomic expr f := SubStr(vᵢ, p₁, p₂)
#                | ConstStr(s)
#                | Loop(λw: e)
# Position p    := CPos(k) | Pos(r₁, r₂, c)
# Int expr c    := k | k₁w + k₂
# Regexp  r     := TokenSeq(T₁, …,Tₘ)

abstract type Op end
abstract type TraceOp <: Op end
abstract type AtomicOp <: Op end 
abstract type PositionOp <: Op end


# Define recursive equality check for Op types

import Base.==
import Base.hash

function ==(a::T, b::T) where T <: Op
    f = fieldnames(T)
    getfield.(Ref(a), f) == getfield.(Ref(b), f)
end

function hash(x::T, h::UInt) where T <: Op
    h = hash(:T, h)
    # for f in fieldnames(T) h = hash(f, h) end
    # return h
    return hash(getfield.(Ref(x), fieldnames(T)), h)
end

# Concatenate(exprs) concatenate the result of all elements of exprs

struct Concatenate <: TraceOp
    exprs::Vector{AtomicOp}
end

invoke(op::Concatenate, input::ProgramState) = join(
    [invoke(e, input) for e in op.exprs]
)

# Fallback unify method
unify(op1, op2, w_id) = nothing


# The SubStr(vᵢ, p₁, p₂) constructor makes use of two position expressions 
# p₁ and p₂, each of which evaluates to an index within the string vᵢ.
# SubStr(𝑣𝑖,p1,p2) denotes the substring of string 𝑣𝑖 that starts at index 
# specified by p1 and ends at index specified by p2-1. 
# If either of p1 or p2 refer to an index that is outside the range of 
# string 𝑣𝑖, then the SubStr constructor returns ⊥.

struct SubStr <: AtomicOp
    vᵢ::Int
    p₁::PositionOp
    p₂::PositionOp
end

function invoke(op::SubStr, input::ProgramState)
    lpos = invoke(op.p₁, input[op.vᵢ])
    rpos = invoke(op.p₂, input[op.vᵢ])

    if isnothing(lpos) || isnothing(rpos) 
        return nothing
    else
        return SubString(input[op.vᵢ], lpos, rpos)
    end
end

function unify(op1::SubStr, op2::SubStr, w_id::String)
    if op1 == op2 return op1 end
    if op1.vᵢ == op2.vᵢ
        u1 = unify(op1.p₁, op2.p₁, w_id)
        u2 = unify(op1.p₂, op2.p₂, w_id)
        if !isnothing(u1) && !isnothing(u2)
            return SubStr(op1.vᵢ, u1, u2)
        end
    end
    return nothing
end


# ConstStr denotes a constant string

struct ConstStr <: AtomicOp
    s::String
end

invoke(op::ConstStr, input::ProgramState) = op.s

unify(op1::ConstStr, op2::ConstStr, w_id::String) = op1 == op2 ? op1 : nothing


# The string expression Loop(𝜆𝑤 : e) refers to concatenation of e1, e2, . . . , e𝑛, 
# where e𝑖 is obtained from e by replacing all occurrences of 𝑤 by 𝑖. 𝑛 is the 
# smallest integer such that evaluation of e𝑛+1 yields ⊥. It is also possible to 
# define more interesting termination conditions (based on position expression, 
# or predicates), but we leave out details for lack of space.

struct Loop <: AtomicOp
    w_id::String
    repeat_op::Union{TraceOp, AtomicOp}
end

invoke(op::Loop, input::ProgramState) = loop_recurisve(
    op.w_id,
    op.repeat_op,
    1,
    input,
)

function loop_recurisve(
    w_id::String, op::Union{TraceOp, AtomicOp}, k::Int, input::ProgramState
)
    replace_vars(op, Dict(w_id => k))
    result = invoke(op, input)
    if isnothing(result) || k > maximum(length.(input))
        return ""
    else
        return result * loop_recurisve(w_id, op, k+1, input)
    end
end

function replace_vars(op::Op, mapping::Dict{String, Int})
    if op isa Concatenate
        for child_op in op.exprs replace_vars(child_op, mapping) end
    elseif op isa SubStr
        replace_vars(op.p₁, mapping)
        replace_vars(op.p₂, mapping)
    elseif op isa Loop
        replace_vars(op.repeat_op, mapping)
    elseif (op isa Pos || op isa PosSet) &&
         op.c isa IntVar && haskey(mapping, op.c.w_id)
        op.c.w = mapping[op.c.w_id]
    end
end

# TODO: Think about handling Loop unification
function unify(op1::Loop, op2::Loop, w_id::String) 
    if op1 == op2 return op1 end
    if op1.w == op2.w 
        u = unify(op1.repeat_op, op2.repeat_op, w_id)
        if !isnothing(u) return Loop(op1.w, u) end
    end
    return nothing
end

# The position expression CPos(𝑘) refers to the 𝑘𝑡h index in a given string 
# from the left side (or right side), if the integer constant 𝑘 is 
# non-negative (or negative).

struct CPos <: PositionOp
    k::Int
end

function invoke(op::CPos, input::AbstractString)
    if op.k == 0 || abs(op.k) > length(input)
        return nothing
    elseif op.k < 0
        return length(input) + 1 + op.k
    else
        return op.k
    end
end

unify(op1::CPos, op2::CPos, w_id::String) = op1.k == op2.k ? op1 : nothing

# Pos(r1 , r2 , c) is another position constructor, where r1 and r2 are some 
# regular expressions and integer expression c evaluates to a non-zero integer. 
# The Pos constructor evaluates to an index 𝑡 in a given string 𝑠 such that r1 
# matches some suffix of 𝑠[0 : 𝑡-1] and r2 matches some prefix of 𝑠[𝑡 : l-1], 
# where l = Length(𝑠). Furthermore, 𝑡 is the c𝑡h such match starting from the 
# left side (or the right side) if c is positive (or negative). 
# If not enough matches exist, then ⊥ is returned.

# TODO: Alternative to mutable?
mutable struct IntVar
    k1::Int
    k2::Int
    w_id::String
    w::Union{Int, Nothing}
end

struct Pos <: PositionOp
    r₁::TokenSeq
    r₂::TokenSeq
    c::Union{Int, IntVar}
end

function invoke(op::Pos, input::AbstractString)
    if op.c isa IntVar && isnothing(op.c.w) error(
        "named variable should be bound before operator invokation"
    ) end

    c = op.c isa IntVar ? op.c.k1 * op.c.w + op.c.k2 : op.c

    if c == 0 error("index to Pos being 0 should not be possible") end

    lrs = join([token_to_r_string[tok] for tok in op.r₁])
    rrs = join([token_to_r_string[tok] for tok in op.r₂])
    r = Regex("(" * lrs * ")(" * rrs * ")")

    matches = collect(eachmatch(r, input))
    if abs(c) > length(matches) return nothing end
    match = wrapped_index(matches, c)
    return match.offset + length(match[1])
end

function wrapped_index(vec::Vector, index::Int)
    if index < 0
        return vec[length(vec) + 1 + index]
    else
        return vec[index]
    end
end

# function unify(op1::Pos, op2::Pos, w_id::String) 
#     if op1 == op2 return op1 end
#     if op1.r₁ == op2.r₁ && op1.r₂ == op2.r₂ && op1.c != op2.c
#         if isa(op1.c, Int) && isa(op2.c, Int)
#             # if op2.c < op1.c error("how did his happen?") end
#             diff = op2.c - op1.c
#             return Pos(
#                 op1.r₁,
#                 op2.r₂,
#                 IntVar(op1.c - diff, diff, w_id, nothing)
#             )
#         end
#     end
#     return nothing
# end


struct PosSet <: PositionOp
    r₁::Vector{TokenSet}
    r₂::Vector{TokenSet}
    c::Union{Int, IntVar}
end


function unify(op1::PosSet, op2::PosSet, w_id::String) 
    if op1 == op2 return op1 end
    if op1.r₁ == op2.r₁ && op1.r₂ == op2.r₂ && op1.c != op2.c
        if isa(op1.c, Int) && isa(op2.c, Int)
            # if op2.c < op1.c error("how did his happen?") end
            diff = op2.c - op1.c
            return PosSet(
                op1.r₁,
                op2.r₂,
                IntVar(op1.c - diff, diff, w_id, nothing)
            )
        end
    end
    return nothing
end

invoke(op::PosSet, input::AbstractString) = invoke(
    Pos(
        [first(tok_set) for tok_set in op.r₁],
        [first(tok_set) for tok_set in op.r₂],
        op.c
    ),
    input
)



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


function unify(d1::DAG, d2::DAG, w_id::String)
    println("Unifying DAGs")
    nodes = Set{Node}()
    source = [d1.ηˢ; d2.ηˢ]
    target = [d1.ηᵗ; d2.ηᵗ]
    edges = Set{Edge}()
    edge_expr_map = EdgeExprMap()
    t1 = length(d1.W) * length(d2.W)
    c1 = 1
    for ((d1_ηₛ, d1_ηₜ), (d2_ηₛ, d2_ηₜ)) in Iterators.product(
        keys(d1.W), keys(d2.W)
    )
        # println(
        #     c1 += 1, "/", t1, " - ",
        #     d1_ηₛ => d1_ηₜ, " ", d2_ηₛ => d2_ηₜ, " ",
        #     length(d1.W[d1_ηₛ => d1_ηₜ]), " ", length(d2.W[d2_ηₛ => d2_ηₜ]), " ",
        #     length(d1.W[d1_ηₛ => d1_ηₜ]) * length(d2.W[d2_ηₛ => d2_ηₜ])
        # )
        result = Set{Op}()
        for (p1, p2) in Iterators.product(
            d1.W[d1_ηₛ => d1_ηₜ], d2.W[d2_ηₛ => d2_ηₜ]
        )
            u = unify(p1, p2, w_id)
            if !isnothing(u) push!(result, u) end
        end

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

function package(path::Path, dag::DAG)
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

function Base.iterate(dag::DAG)
    paths = dfs_backward_itr_from(dag, dag.ηᵗ)
    return if isnothing(paths) nothing 
    else (package(paths[1], dag), (paths=paths, count=2))
    end
end

Base.iterate(
    dag::DAG,
    state::NamedTuple{(:paths, :count), Tuple{Vector{Path}, Int}}
) = return state.count > length(state.paths) ? nothing : (
    package(state.paths[state.count], dag),
    (paths=state.paths, count=state.count + 1)
)



######################
# Program generation #
######################

struct Cache
    token_eqv_sets::Dict{
        AbstractString, Dict{Vector{SubString{String}}, TokenSet}
    }
    token_matches::Dict{
        AbstractString, Dict{Token, Vector{RegexMatch}}
    }
    lr_matches::Dict{
        AbstractString,
        NamedTuple{
            (:left, :right),
            Tuple{
                Dict{
                    Int, Vector{NamedTuple{(:tok, :start), Tuple{Token, Int}}}
                },
                Dict{
                    Int, Vector{NamedTuple{(:tok, :stop), Tuple{Token, Int}}}
                }
            }
        }
    }
    pos::Dict{Tuple{AbstractString, Int}, Set{PositionOp}}
end

Cache() = Cache(Dict(), Dict(), Dict(), Dict())


"Generates program that satisfies given I/O specs"
function generate_programs(specs::ProgramSpec)
    cache = Cache()
    dags = [generate_dag(s..., cache) for s in specs]
    println("Done generating DAGs")
    isection = dags[1]
    for d in dags[2:length(dags)]
        isection = intersect(isection, d)
    end
    if isempty(isection) return [] end
    p = sample(isection)
    if isnothing(p) return [] end
    works = all([invoke(p, s[1]) == s[2] for s in specs])
    return [p]
end

"Generate DAG representation of all possible traces"
function generate_dag(
    input_state::ProgramState, output_part::AbstractString, cache::Cache
)
    println("Generating DAG")
    η = Set([[i] for i in 1:length(output_part)])
    ηˢ = [0]
    ηᵗ = [length(output_part)]
    ξ = Set([([i] => [j]) for j in 1:length(output_part) for i in 0:j-1])
    W = EdgeExprMap(
        edge => union(
            generate_substrings(
                input_state, 
                SubString(output_part, edge[1][1]+1, edge[2][1]),
                cache
            ),
            [ConstStr(SubString(output_part, edge[1][1]+1, edge[2][1]))]
        )
        for edge in ξ
    )
    W = generate_loop(input_state, output_part, W, cache)
    W = unroll_possets(W)
    return DAG(η, ηˢ, ηᵗ, ξ, W)
end

function unroll_possets(W::EdgeExprMap)
    new_W = EdgeExprMap()
    for (edge, exprs) in W
        new_W[edge] = Set{Op}()
        for e in exprs
            if e isa PosSet 
                union!(new_W[edge], [Pos(r1, r2, e.c)
                    for r1 in Iterators.product(e.r₁...)
                    for r2 in Iterators.product(e.r₂...)
                ])
            else
                push!(new_W[edge], e)
            end
        end
    end
    return new_W
end

function generate_loop(
    input_state::ProgramState,
    output_part::AbstractString,
    W::EdgeExprMap,
    cache::Cache
)
    new_W = W
    w_counter = 0
    for k3 in 1:length(output_part)
        for k1 in 0:k3-1
            for k2 in k1+1:k3-1
                println(k1, " ", k2, " ", k3)
                Wl = filter(((k, v),) -> k1 <= k[1][1] && k[2][1] <= k2, W)
                Wr = filter(((k, v),) -> k2 <= k[1][1] && k[2][1] <= k3, W)
                dl = DAG(Set([[i] for i in k1:k2]), [k1], [k2], keys(Wl), Wl)
                dr = DAG(Set([[i] for i in k2:k3]), [k2], [k3], keys(Wr), Wr)
                w_id = string("w", w_counter += 1)
                unified_dag = unify(dl, dr, w_id)
                for p in unified_dag
                    t_loop = Loop(w_id, p)
                    result = invoke(t_loop, input_state)
                    for r in findall(result, output_part)
                        push!(new_W[[r.start - 1] => [r.stop]], t_loop)
                    end
                end
            end
        end
    end
    return new_W
end

"Generates a `SubStr` expression given input and output"
generate_substrings(
    input_state::ProgramState,
    output_part::AbstractString,
    cache::Cache
) = Set{SubStr}([
    SubStr(input_part_idx, lp, rp)        
    for (input_part_idx, input_part) in enumerate(input_state)
    for frange in findall(output_part, input_part)    
    for lp in generate_positions(input_part, frange.start, cache)
    for rp in generate_positions(input_part, frange.stop, cache)
])


"Generates `CPos` and `Pos` expressions given input string and index"
function generate_positions(
    input_part::AbstractString, index::Int, cache::Cache
)
    if haskey(cache.pos, (input_part, index)) 
        return cache.pos[(input_part, index)]
    end
    result = Set{PositionOp}()
    union!(result, [CPos(index), CPos(-(length(input_part) + 1 - index))])

    tok_eqv_sets, tok_matches = findall_token_matches(input_part, cache)
    lrm = generate_lr_matches(input_part, tok_eqv_sets, tok_matches, cache)
    left_rs, right_rs = find_matching_tokseqs(lrm, index)

    for ((kl, lr), (kr, rr)) in Iterators.product(left_rs, right_rs)
        lrs = join([token_to_r_string[tok] for tok in lr])
        rrs = join([token_to_r_string[tok] for tok in rr])
        r = Regex("(" * lrs * ")(" * rrs * ")")

        # TODO: change to logic that chooses match according to index
        # rather than using `findfirst`
        matches = collect(eachmatch(r, input_part))
        current_substring = SubString(input_part, kl, kr-1)
        c = findfirst([m.match == current_substring for m in matches])
        if isnothing(c) continue end

        to_toksetseq = ts -> [
            tok_eqv_sets[[m.match for m in tok_matches[t]]]
            for t in ts
        ]
        lr, rr = to_toksetseq(lr), to_toksetseq(rr)
        push!(result, PosSet(lr, rr, c,))
        push!(result, PosSet(lr, rr, -(length(matches) + 1 - c)))
    end

    cache.pos[(input_part, index)] = result
    return result
end

"""
Finds matches for all tokens within given  
string and generates equivalence sets
"""
function findall_token_matches(s::AbstractString, cache::Cache)
    if haskey(cache.token_eqv_sets, s) && haskey(cache.token_matches, s) 
        return cache.token_eqv_sets[s], cache.token_matches[s]
    end
    tok_eqv_sets = Dict{Vector{SubString{String}}, TokenSet}()
    tok_matches = Dict{Token, Vector{RegexMatch}}()

    for tok in instances(Token)
        matches = collect(eachmatch(token_to_r[tok], s))
        match_strings = [m.match for m in matches]
        tok_matches[tok] = matches
        if haskey(tok_eqv_sets, match_strings) 
            push!(tok_eqv_sets[match_strings], tok)
        else
            tok_eqv_sets[match_strings] = TokenSet([tok])
        end
    end

    cache.token_eqv_sets[s] = tok_eqv_sets
    cache.token_matches[s] = tok_matches

    return tok_eqv_sets, tok_matches
end

"""
Generates left side and right side matching token sequences
for given set of matches
"""
function generate_lr_matches(
    s::AbstractString,
    tok_eqv_sets::Dict{Vector{SubString{String}}, TokenSet},
    tok_matches::Dict{Token, Vector{RegexMatch}},
    cache::Cache
)
    if haskey(cache.lr_matches, s) return cache.lr_matches[s] end

    right_matches = Dict()
    left_matches = Dict()
    reps = [first(eqv_set) for eqv_set in values(tok_eqv_sets)]
    
    for tok in reps
        for m in tok_matches[tok]
            start, stop = m.offset, m.offset + length(m.match)

            if !haskey(right_matches, start) right_matches[start] = [] end
            push!(right_matches[start], (tok=tok, stop=stop))

            if !haskey(left_matches, stop) left_matches[stop] = [] end
            push!(left_matches[stop], (tok=tok, start=start))
        end
    end

    lr_matches = (left=left_matches, right=right_matches)
    cache.lr_matches[s] = lr_matches
    return lr_matches
end


"Finds matching regex token sequence for specified index of a given string"
function find_matching_tokseqs(lr_matches, k::Int) 
    right_seqs, left_seqs = [], []
        
    if haskey(lr_matches.left, k)
        for m1 in lr_matches.left[k]
            push!(left_seqs, (m1.start, [m1.tok]))

            if haskey(lr_matches.left, m1.start)
                for m2 in lr_matches.left[m1.start]
                    push!(left_seqs, (m1.start, [m2.tok, m1.tok]))
                end
            end            
        end
    end

    if haskey(lr_matches.right, k)
        for m1 in lr_matches.right[k]
            push!(right_seqs, (m1.stop, [m1.tok]))

            if haskey(lr_matches.right, m1.stop)
                for m2 in lr_matches.right[m1.stop]
                    push!(right_seqs, (m1.stop, [m1.tok, m2.tok]))
                end
            end            
        end
    end

    return left_seqs, right_seqs
end




# TODO: find good way to get best program
choose_top_program(p::Vector{<:Op}) = p[1]


############################################
# Use `generate_substrings` with DataFrame #
############################################

using Tables
using DataFrames

"Fills rows of specified column that contain `missing` with suitable values"
function flashfill(x::DataFrame, col)
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
    if isempty(programs) return nothing, nothing end    
    p = choose_top_program(programs)
    outputs = [invoke(p, i) for i in inputs]

    filled = copy(x)
    filled[ismissing.(x[!, col]), col] = outputs
    return filled, p
end


end