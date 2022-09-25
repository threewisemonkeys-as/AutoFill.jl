######################
# Language Semantics #
######################

using Memoization

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


# Fallback unify method
unify(op1, op2, w_id) = nothing

# Concatenate(exprs) concatenate the result of all elements of exprs

struct Concatenate <: TraceOp
    exprs::Vector{AtomicOp}
end

invoke(op::Concatenate, input::ProgramState) = join(
    [
        str for str in [invoke(e, input) for e in op.exprs]
        if !isnothing(str)
    ]    

)

function replace_vars(op::Concatenate, mapping::Dict{String, Int})
    for child_op in op.exprs replace_vars(child_op, mapping) end
end

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

function replace_vars(op::SubStr, mapping::Dict{String, Int})
    replace_vars(op.p₁, mapping)
    replace_vars(op.p₂, mapping)
end

# ConstStr denotes a constant string

struct ConstStr <: AtomicOp
    s::String
end

invoke(op::ConstStr, input::ProgramState) = op.s

unify(op1::ConstStr, op2::ConstStr, w_id::String) = op1 == op2 ? op1 : nothing

replace_vars(op::ConstStr, mapping::Dict{String, Int}) = nothing

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

function replace_vars(op::Loop, mapping::Dict{String, Int})
    replace_vars(op.repeat_op, mapping)
end


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

replace_vars(op::CPos, mapping::Dict{String, Int}) = op

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

    c = op.c isa IntVar ? op.c.k1 + op.c.w * op.c.k2 : op.c

    # error("index to Pos being 0 should not be possible") 
    if c == 0 return nothing end

    lrs = join([token_to_r_string[tok] for tok in op.r₁])
    rrs = join([token_to_r_string[tok] for tok in op.r₂])
    r = Regex("(" * lrs * ")(" * rrs * ")")

    matches = collect(eachmatch(r, input))
    if abs(c) > length(matches) return nothing end
    match = wrapped_index(matches, c)

    result = match.offset + length(match[1]) 
    if result == length(input) + 1
        result -= 1
    end

    return result
end

[0, 1] => [1, 2]
[1, 2] => [2, 3]
[2, 3] => [3, 4]

[0, 2]
[1, 3]
[2, 4]

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

function replace_vars(
    op::Union{Pos, PosSet},
    mapping::Dict{String, Int}
)
    if op.c isa IntVar && haskey(mapping, op.c.w_id)
        op.c.w = mapping[op.c.w_id]
    end
end



# Define recursive equality check for Op types

import Base.==
import Base.hash

function ==(a::T, b::T) where T <: Op
    f = fieldnames(T)
    getfield.(Ref(a), f) == getfield.(Ref(b), f)
end

function ==(a::IntVar, b::IntVar)
    return (
        a.k1 == b.k1
        && a.k2 == b.k2
        && a.w_id == b.w_id
    )
end


function hash(x::T, h::UInt) where T <: Op
    h = hash(:T, h)
    # for f in fieldnames(T) h = hash(f, h) end
    # return h
    return hash(getfield.(Ref(x), fieldnames(T)), h)
end

function hash(x::IntVar, h::UInt)
    h = hash(:IntVar, h)
    return hash(
        (x.k1, x.k2, x.w_id),
        h
    )
end

loop_invariant_hash(x, h::UInt) = hash(x, h)

loop_invariant_hash(x::Union{Tuple, Vector}, h::UInt) = hash(
        loop_invariant_hash.(x, h)
    )

loop_invariant_hash(x::T) where T <: Op = loop_invariant_hash(
    x, zero(UInt)
)

function loop_invariant_hash(x::T, h::UInt) where T <: Op
    h = hash(:T, h)
    # for f in fieldnames(T) h = hash(f, h) end
    # return h
    return loop_invariant_hash(
        getfield.(Ref(x), fieldnames(T)), h
    )
end

loop_invariant_hash(x::Union{Pos, PosSet}, h::UInt) = hash(
    (x.r₁, x.r₂), h
)

