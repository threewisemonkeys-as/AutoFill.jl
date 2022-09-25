######################s
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
function generate_programs(
    specs::ProgramSpec,
    max_samples_per_dag::Int = 100,
    max_num::Int = 1
)
    dags = [generate_dag(s...) for s in specs]
    # println([length([p for p in d]) for d in dags])
    # println("Done generating DAGs")
    i = 0
    programs = Vector{Op}()
    for d in dags
        for sample_idx in 1:max_samples_per_dag
            p = sample(d)
            if all([invoke(p, s[1]) == s[2] for s in specs])
                push!(programs, p)
                i += 1
                if i >= max_num break end
            end
        end
        if i >= max_num break end
    end

    return programs

    # isection = dags[1]
    # for d in dags[2:length(dags)]
    #     isection = intersect(isection, d)
    # end
    # if isempty(isection) return [] end
    # p = sample(isection)
    # if isnothing(p) return [] end
    # works = all([invoke(p, s[1]) == s[2] for s in specs])
    # return [p]
end

"Generate DAG representation of all possible traces"
function generate_dag(
    input_state::ProgramState, output_part::AbstractString
)
    cache = Cache()

    # println("Generating DAG")
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

unroll_possets(W::EdgeExprMap) = EdgeExprMap(
    edge => Set{Op}(reduce(vcat, vec.(unroll_possets.(exprs))))
    for (edge, exprs) in W
)

unroll_possets(op::Concatenate) = [
    Concatenate([c...])
    for c in Iterators.product(
        [unroll_possets(e) for e in op.exprs]...
    )
]

unroll_possets(op::Loop) = [
    Loop(op.w_id, e) for e in unroll_possets(op.repeat_op)
]

unroll_possets(op::SubStr) = [
    SubStr(op.vᵢ, p1, p2)
    for (p1, p2) in Iterators.product(
        unroll_possets(op.p₁),
        unroll_possets(op.p₂)
    )
]

unroll_possets(op::PosSet) = [
    Pos(collect(r1), collect(r2), op.c) for 
    (r1, r2) in Iterators.product(
        collect(Iterators.product(op.r₁...)),
        collect(Iterators.product(op.r₂...))
    )
]

unroll_possets(op) = [op]

function generate_loop(
    input_state::ProgramState,
    output_part::AbstractString,
    W::EdgeExprMap,
    cache::Cache
)
    new_W = W
    w_id = string("w", 0)
    # println("Creating unification cache")
    unification_cache = cache_unification(W, 0, length(output_part), w_id)

    # println("Unifying DAGs")
    for k3 in 1:length(output_part)
        for k1 in 0:k3-1
            for k2 in k1+1:k3-1
                if k2 - k1 == k3 - k2 == 1
                    Wl = filter(((k, v),) -> k1 <= k[1][1] && k[2][1] <= k2, W)
                    Wr = filter(((k, v),) -> k2 <= k[1][1] && k[2][1] <= k3, W)
                    dl = DAG(Set([[i] for i in k1:k2]), [k1], [k2], keys(Wl), Wl)
                    dr = DAG(Set([[i] for i in k2:k3]), [k2], [k3], keys(Wr), Wr)
                    unified_dag = unify(dl, dr, w_id, unification_cache)
                    subprograms = extract_programs(unified_dag)
                    if isnothing(subprograms) continue end
                    # println(k1, " ", k2, " ", k3, " ", length(subprograms))
                    for (idx, p) in enumerate(subprograms)
                        # println(idx, " / ", length(subprograms))
                        t_loop = Loop(w_id, p)
                        result = invoke(t_loop, input_state)
                        if !isnothing(result) && length(result) > 1
                            for r in findall(result, output_part)
                                push!(new_W[[r.start - 1] => [r.stop]], t_loop)
                            end
                        end
                    end
                end
            end
        end
    end
    return new_W
end

function cache_unification(
    W::EdgeExprMap, start_idx::Int, stop_idx::Int, w_id::String
)
    cache = Dict{Edge, Set{<:Op}}()
    for k3 in start_idx+1:stop_idx
        for k1 in start_idx:k3-1
            for k2 in k1+1:k3-1
                s1 = W[[k1] => [k2]]

                if k2 - k1 == k3 - k2
                    s2 = W[[k2] => [k3]]
                    t = length(s1) * length(s2)
                    counter = 0
                    # println(k1, " ", k2, " ", k3, " ", length(s1), " ", length(s2), " ", t) 
                    result = Set{Op}()
                    p_dict = Dict{UInt, Set{Op}}()
                    
                    for p1 in s1
                        p_hash = loop_invariant_hash(p1)
                        if !haskey(p_dict, p_hash)
                            p_dict[p_hash] = Set{Op}()
                        end
                        push!(p_dict[p_hash], p1)
                    end

                    for p2 in s2
                        p_hash = loop_invariant_hash(p2)
                        if haskey(p_dict, p_hash)
                            for p1 in p_dict[p_hash]
                                u = unify(p1, p2, w_id)
                                if !isnothing(u)
                                    push!(result, u)
                                    # println(AutoFill.invoke(Loop("w0", u), "International Conference on Machine Learning"))
                                end
                            end
                        end
                    end

                    # for (p1, p2) in Iterators.product(s1, s2)
                    #     u = unify(p1, p2, w_id)
                    #     counter += 1
                    #     if counter % Int(floor(t // 100)) == 0 println(counter, " ", t, " ", counter / t) end
                    #     if !isnothing(u) push!(result, u) end
                    # end    
                    # collected = Vector{Union{Nothing, Op}}(
                    #     undef, length(s1) * length(s2)
                    # )
                    # for (idx, (p1, p2)) in collect(
                    #     enumerate(Iterators.product(s1, s2))
                    # )
                    #     collected[idx] = unify(p1, p2, w_id)
                    # end

                    # for item in collected
                    #     if !isnothing(item) push!(result, item) end
                    # end
                    if !isempty(result) cache[[k1, k2] => [k2, k3]] = result end
                end
            end
        end
    end
    return cache
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
    left_rs, right_rs = construct_matching_tokseqs(lrm, index)

    for ((kl, lr), (kr, rr)) in Iterators.product(left_rs, right_rs)
        lrs = join([token_to_r_string[tok] for tok in lr])
        rrs = join([token_to_r_string[tok] for tok in rr])
        r = Regex("(" * lrs * ")(" * rrs * ")")

        matches = collect(eachmatch(r, input_part))
        current_substring = SubString(input_part, kl, kr-1)
        c = nothing
        for (idx, match) in enumerate(matches)
            if match.offset == index 
                c = idx
                break
            end
        end
        if isnothing(c) continue end

        to_toksetseq = ts -> [
            haskey(tok_matches, t) ? tok_eqv_sets[[m.match for m in tok_matches[t]]] : TokenSet([t])
            for t in ts
        ]
        lr, rr = to_toksetseq(lr), to_toksetseq(rr)
        push!(result, PosSet(lr, rr, c))
        push!(result, PosSet(lr, rr, -(length(matches) + 1 - c)))
        # println(r, " ", current_substring, " ", c, " ", index, " ", [m.match for m in matches], " ", PosSet(lr, rr, c), " ", loop_invariant_hash(PosSet(lr, rr, c)))
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
        if tok == StartTok
            # handled in generate_lr_matches
        elseif tok == EndTok
            # handled in generate_lr_matches
        else
            matches = collect(eachmatch(token_to_r[tok], s))
            match_strings = [m.match for m in matches]
            tok_matches[tok] = matches
            if haskey(tok_eqv_sets, match_strings) 
                push!(tok_eqv_sets[match_strings], tok)
            else
                tok_eqv_sets[match_strings] = TokenSet([tok])
            end
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

    if !haskey(right_matches, 1) right_matches[1] = [] end
    push!(right_matches[1], (tok=StartTok, stop=1))

    len = length(s)
    if !haskey(left_matches, len) left_matches[len] = [] end
    push!(left_matches[len], (tok=EndTok, start=len))

    lr_matches = (left=left_matches, right=right_matches)
    cache.lr_matches[s] = lr_matches
    return lr_matches
end


"Finds matching regex token sequence for specified index of a given string"
function construct_matching_tokseqs(lr_matches, k::Int) 
    left_seqs, right_seqs = [(k, [])], [(k, [])]

    if haskey(lr_matches.left, k)
        for m1 in lr_matches.left[k]
            push!(left_seqs, (m1.start, [m1.tok]))

            if haskey(lr_matches.left, m1.start)
                for m2 in lr_matches.left[m1.start]
                    push!(left_seqs, (m2.start, [m2.tok, m1.tok]))
                end
            end            
        end
    end

    if haskey(lr_matches.right, k)
        for m1 in lr_matches.right[k]
            push!(right_seqs, (m1.stop, [m1.tok]))

            if haskey(lr_matches.right, m1.stop)
                for m2 in lr_matches.right[m1.stop]
                    push!(right_seqs, (m2.stop, [m1.tok, m2.tok]))
                end
            end            
        end
    end

    return left_seqs, right_seqs
end
