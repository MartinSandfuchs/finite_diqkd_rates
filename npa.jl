# """
# Solve polynomial optimization problems using the NPA hierarchy
# """
module NPA

import Base:+,-,*,/,real,imag,size,getindex,append!,vec
using DataStructures
using LinearAlgebra
using JuMP
using Mosek
using MosekTools


#############################################################################
############################## Polynome struct ##############################
#############################################################################
"""
A Polynome with coefficients of type `T`.
Use the `Variables` data type to construct degree one polynomes.
More complex polynomes can then be constructed by adding, subtracting and multiplying smaller polynomes.
"""
struct Polynome{T<:Number}
    monomes::Vector{Vector{Int}}
    coeffs::Vector{T}
    deg::Int

    function Polynome(monomes::Vector{Vector{Int}}, coeffs::Vector{T}) where T<:Number
        deg = 0
        for monome in monomes
            deg = max(deg, length(monome))
        end
        new{T}(monomes, coeffs, deg)
    end
end

function Polynome{T}() where T<:Number
    Polynome(Vector{Vector{Int}}(), T[])
end

from_char(t::Type{T}, char) where T<:Number = Polynome([[char]], [one(t)])


# Addition
function +(p1::Polynome{<:Number}, p2::Polynome{<:Number})
    coeffs1, coeffs2 = promote(p1.coeffs, p2.coeffs)
    Polynome([p1.monomes; p2.monomes], [coeffs1; coeffs2])
end

function +(p::Polynome{S}, x::T) where S<:Number where T<:Number
    t = promote_type(S, T)
    Polynome([p.monomes; [Vector{Int}()]], [convert(Vector{t}, p.coeffs); convert(t, x)])
end

+(x::S, p::Polynome{T}) where {S<:Number, T<:Number} = p + x


# Subtraction
function -(p1::Polynome{<:Number}, p2::Polynome{<:Number})
    coeffs1, coeffs2 = promote(p1.coeffs, p2.coeffs)
    Polynome([p1.monomes; p2.monomes], [coeffs1; .-coeffs2])
end

function -(p::Polynome{S}, x::T) where {S<:Number, T<:Number}
    t = promote_type(S, T)
    Polynome([p.monomes; [Vector{Int}()]], [convert(Vector{t}, p.coeffs); convert(t, -x)])
end

function -(x::S, p::Polynome{T}) where {S<:Number, T<:Number}
    t = promote_type(S, T)
    Polynome([[Vector{Int}()]; p.monomes], [convert(t, x); convert(Vector{t}, .-p.coeffs)])
end

function -(p::Polynome{T}) where T<:Number
    Polynome(copy(p.monomes), .-p.coeffs)
end


# Multiplication
function *(p1::Polynome{S}, p2::Polynome{T}) where {S<:Number, T<:Number}
    t = promote_type(S, T)
    coeffs = Vector{t}()
    sizehint!(coeffs, length(p1.coeffs)*length(p2.coeffs))
    monomes = Vector{Vector{Int}}()
    sizehint!(monomes, length(p1.monomes)*length(p2.monomes))

    for i in 1:length(p1.coeffs)
        for j in 1:length(p2.coeffs)
            push!(coeffs, p1.coeffs[i]*p2.coeffs[j])
            push!(monomes, [p1.monomes[i]; p2.monomes[j]])
        end
    end
    Polynome(monomes, coeffs)
end

function *(p::Polynome{S}, x::T) where {S<:Number, T<:Number}
    t = promote_type(S, T)
    Polynome(copy(p.monomes), convert(t, x)*convert(Vector{t}, p.coeffs))
end

*(x::S, p::Polynome{T}) where {S<:Number, T<:Number} = p*x


# Division
function /(p::Polynome{S}, x::T) where {S<:Number, T<:Number}
    t = promote_type(S, T)
    Polynome(copy(p.monomes), convert(Vector{t}, p.coeffs)/convert(t, x))
end


##############################################################################
############################## Helper functions ##############################
##############################################################################
"""
Substitute the first occurance of `old_mono` with `new_mono`.
"""
function substitute_monomial!(mono, old_mono, new_mono)
    for i in 1:length(mono) - length(old_mono) + 1
        j = i + length(old_mono) - 1
        is_equal = true
        for k in i:j
            if mono[k] != old_mono[k - i + 1]
                is_equal = false
                break
            end
        end
        if is_equal && new_mono == nothing
            return true
        end
        if is_equal
            splice!(mono, i:j, new_mono)
            return true
        end
    end
    false
end

"""
Repeatedly apply all the `substitutions` in subs to `mono`.
If one of the substitutions is empty (corresponding to the zero monomial), return nothing.
"""
function apply_substitutions(mono, subs)
    mono = copy(mono)
    updated = true
    while updated
        updated = false
        for (old, new) in subs
            updated = substitute_monomial!(mono, old, new)
            if updated
                if new == nothing
                    return nothing
                end
                break
            end
        end
    end
    mono
end


#################################################################################
############################## Variable Management ##############################
#################################################################################
"""
Create variables from which polynomials can be built
"""
struct Variables
    hermitian_idx::Vector{Int}
end

function Variables()
    Variables([])
end

"""
Create a pair of non hermitian variables. Returns the new variable and its conjugate
"""
function create_non_hermitian_var(vars::Variables)
    idx = length(vars.hermitian_idx) + 1
    push!(vars.hermitian_idx, idx + 1)
    push!(vars.hermitian_idx, idx)
    from_char(Float64, idx), from_char(Float64, idx + 1)
end

"""
Create a hermitian variable and return it
"""
function create_hermitian_var(vars::Variables)
    idx = length(vars.hermitian_idx) + 1
    push!(vars.hermitian_idx, idx)
    from_char(Float64, idx)
end


# Relaxation parameters
"""
Parameters for building the NPA hierarchy
"""
struct RelaxationParameters
    deg::Int
    substitutions::Vector{Tuple{Vector{Int}, Union{Nothing, Vector{Int}}}}
    hermitian_idx::Vector{Int}
    basis::Vector{Vector{Int}}
    monomial_index::Dict{Vector{Int}, Int}
    monomes::Vector{Vector{Int}}
end

"""
Compare two monomes
returns `true` if `m1 <= m2`
"""
function leq(m1, m2)
    if length(m1) < length(m2) 
        return true
    elseif length(m1) > length(m2)
        return false
    end

    for i in 1:length(m1)
        if m1[i] < m2[i]
            return true
        elseif m1[i] > m2[i]
            return false
        end
    end
    true
end

function RelaxationParameters(;
        k,
        vars::Variables,
        substitutions=Vector{Tuple{Polynome{Float64}, Union{Nothing, Polynome{Float64}}}}(),
        extra_monomials=Vector{Polynome{Float64}}()
    )
    n_chars = length(vars.hermitian_idx)

    deg = k
    for mono in extra_monomials
        deg = max(deg, mono.deg)
    end

    # Sort the substitutions
    new_subs = Vector{Tuple{Vector{Int}, Union{Nothing, Vector{Int}}}}()
    for (sub_l, sub_r) in substitutions
        if sub_r == nothing
            push!(new_subs, (sub_l.monomes[1], sub_r))
            continue
        elseif sub_l == nothing
            push!(new_subs, (sub_r.monomes[1], sub_l))
            continue
        end

        monome_l, monome_r = sub_l.monomes[1], sub_r.monomes[1]
        if monome_l == monome_r 
            continue
        end
        if leq(monome_l, monome_r)
            push!(new_subs, (monome_r, monome_l))
        else
            push!(new_subs, (monome_l, monome_r))
        end
    end

    # Build all monomials of degree <= k
    monos_k::Vector{Vector{Int}} = [[]]
    for deg in 1:k
        indices = CartesianIndices(tuple(n_chars * ones(Int, deg)...))
        append!(monos_k, [[Tuple(index)...] for index in indices])
    end

    monos_extra = [mono.monomes[1] for mono in extra_monomials]
    append!(monos_extra, [adjoint_mono(vars.hermitian_idx, mono.monomes[1]) for mono in extra_monomials])

    basis::Vector{Vector{Int}} = []
    monomial_index = Dict{Vector{Int}, Int}()
    monomes::Vector{Vector{Int}} = []

    # Build the basis
    for mono in [monos_k; monos_extra]
        simplified_mono = apply_substitutions(mono, new_subs)
        if simplified_mono == nothing
            continue
        end
        @assert leq(simplified_mono, mono)

        # Skip duplicate items (after simplification)
        if !haskey(monomial_index, simplified_mono)
            push!(basis, simplified_mono)
            push!(monomes, simplified_mono)
            monomial_index[simplified_mono] = length(monomes)
        end
        monomial_index[mono] = monomial_index[simplified_mono]
    end

    RelaxationParameters(deg, new_subs, vars.hermitian_idx, basis, monomial_index, monomes)
end

function adjoint_mono(hermitian_idx::Vector{Int}, mono)
    adj = reverse(mono)
    for idx in 1:length(adj)
        adj[idx] = hermitian_idx[adj[idx]]
    end
    adj
end

adjoint_mono(params::RelaxationParameters, mono) = adjoint_mono(params.hermitian_idx, mono)

function monomial_idx(params::RelaxationParameters, mono)
    # Check if the unsimplified monomial is in the index
    if haskey(params.monomial_index, mono)
        return params.monomial_index[mono]
    end

    # Check if the simplified monomial is in the index
    simplified_mono = apply_substitutions(mono, params.substitutions)
    if simplified_mono == nothing
        return nothing
    end

    if haskey(params.monomial_index, simplified_mono)
        res = params.monomial_index[simplified_mono]
        params.monomial_index[mono] = res
        return res
    end

    # Otherwise we create a new entry in the index
    push!(params.monomes, simplified_mono)
    monome_id = length(params.monomes)
    params.monomial_index[mono] = monome_id
    params.monomial_index[simplified_mono] = monome_id
    monome_id
end


###############################################################################
############################## COO Sparse matrix ##############################
###############################################################################
"""
Sparse matrix format that stores it's entries in COO (triplet) format.
We need this because even `SparseMatrixCSC` can use too much memory.
"""
struct SparseMatrixCOO{Tv, Ti<:Integer} <: AbstractMatrix{Tv}
    m::Int
    n::Int
    rows::Vector{Ti}
    cols::Vector{Ti}
    nzval::Vector{Tv}
end

function SparseMatrixCOO{Tv, Ti}(m::Integer, n::Integer) where {Tv, Ti<:Integer}
    SparseMatrixCOO{Tv,Ti}(m, n, Ti[], Ti[], Tv[])
end

function SparseMatrixCOO{Tv, Ti}(mat::Matrix{Tv}) where {Tv, Ti<:Integer}
    m, n = size(mat)
    rows::Vector{Ti} = []
    cols::Vector{Ti} = []
    nzval::Vector{Tv} = []
    for i in 1:m
        for j in 1:n
            if mat[i, j] != 0.0
                push!(rows, i)
                push!(cols, j)
                push!(nzval, mat[i, j])
            end
        end
    end

    SparseMatrixCOO{Tv,Ti}(m, n, rows, cols, nzval)
end

function +(mat1::SparseMatrixCOO{Tv, Ti}, mat2::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti<:Integer}
    @assert (mat1.m == mat2.m) && (mat1.n == mat2.n)
    SparseMatrixCOO{Tv,Ti}(mat1.m, mat1.n, [mat1.rows; mat2.rows], [mat1.cols; mat2.cols], [mat1.nzval; mat2.nzval])
end

function -(mat::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti<:Integer}
    SparseMatrixCOO{Tv,Ti}(mat.m, mat.n, copy(mat.rows), copy(mat.cols), .-mat.nzval)
end

function append!(mat::SparseMatrixCOO, i, j, x)
    push!(mat.rows, i)
    push!(mat.cols, j)
    push!(mat.nzval, x)
    nothing
end

function transpose(mat::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti<:Integer}
    SparseMatrixCOO{Tv, Ti}(mat.n, mat.m, copy(mat.cols), copy(mat.rows), copy(mat.nzval))
end

function real(mat::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti<:Integer}
    SparseMatrixCOO{Tv, Ti}(mat.m, mat.n, copy(mat.cols), copy(mat.rows), real.(mat.nzval))
end

function imag(mat::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti<:Integer}
    SparseMatrixCOO{Tv, Ti}(mat.m, mat.n, copy(mat.cols), copy(mat.rows), imag.(mat.nzval))
end

function size(mat::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti<:Integer}
    mat.m, mat.n
end

# TODO: Remove this function
function getindex(mat::SparseMatrixCOO{Tv, Ti}, i, j) where {Tv, Ti<:Integer}
    for k in 1:length(mat.nzval)
        if mat.rows[k] == i && mat.cols[k] == j
            return mat.nzval[k]
        end
    end
    0.0
end

function vec(mat::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti<:Integer}
    vec = zeros(Tv, mat.m*mat.n)
    for k in 1:length(mat.nzval)
        row = mat.rows[k]
        col = mat.cols[k]
        vec[row + (col - 1)*mat.m] = mat.nzval[k]
    end

    vec
end

###########################################################################
############################## NPA Hierarchy ##############################
###########################################################################
function moment_matrices_(params::RelaxationParameters)
    localizing_matrices_(params, Polynome([Int[]], [1.0]))
end

function localizing_matrices_(params::RelaxationParameters, qi::Polynome{T}) where {T<:Number}
    di = qi.deg/2 |> ceil |> Int

    # Find the number of monomes in the basis with the desired degree
    mat_size::Int = 0
    for monome in params.basis
        if length(monome) > params.deg - di
            break
        end
        mat_size += 1
    end

    # Build the matrix from its components
    matrices::Vector{SparseMatrixCOO{T, Int}} = []
    for i in 1:mat_size
        for j in 1:mat_size
            for s in 1:length(qi.monomes)
                v = params.monomes[i]
                u = qi.monomes[s]
                w = params.monomes[j]
                y_idx = monomial_idx(params, [adjoint_mono(params, v); u; w])
                if y_idx == nothing
                    continue
                end
                
                append!(matrices, [SparseMatrixCOO{T, Int}(mat_size, mat_size) for _ in 1:(y_idx - length(matrices))]) 
                append!(matrices[y_idx], i, j, qi.coeffs[s])
            end
        end
    end

    matrices
end

function objective_vector_(params::RelaxationParameters, p::Polynome{T}) where {T<:Number}
    obj_vec = T[]
    for k in 1:length(p.monomes)
        idx = monomial_idx(params, p.monomes[k])
        if idx == nothing
            continue
        end

        append!(obj_vec, [0.0 for _ in 1:(idx - length(obj_vec))])
        obj_vec[idx] += p.coeffs[k]
    end

    obj_vec
end

"""
Move from conjugate variables to real and imaginary part
"""
function convert_matrices(params::RelaxationParameters, matrices::Vector{<:AbstractMatrix{<:Real}})
    matrices_new = typeof(matrices)()
    for idx in 1:length(matrices)
        mon = params.monomes[idx]
        adj_idx = monomial_idx(params, adjoint_mono(params, mon))

        M = real(matrices[idx])

        MT = transpose(M)
        if adj_idx > idx
            push!(matrices_new, M + MT)
        elseif adj_idx == idx
            push!(matrices_new, M)
        end
    end
 
    matrices_new
end


function convert_matrices(params::RelaxationParameters, matrices::Vector{<:AbstractMatrix{Tv}}) where {Tv <: Number}
    matrices_a = typeof(matrices)()
    matrices_b = typeof(matrices)()
    for idx in 1:length(matrices)
        mon = params.monomes[idx]
        adj_idx = monomial_idx(params, adjoint_mono(params, mon))

        m = real(matrices[idx])
        n = imag(matrices[idx])

        mt = transpose(m)
        nt = transpose(n)
        if adj_idx > idx
            push!(matrices_a, SparseMatrixCOO{Tv, Int}([(m + mt) (nt - n); (n - nt) (m + mt)]))
            push!(matrices_b, SparseMatrixCOO{Tv, Int}([-(n + nt) (mt - m); (m - mt) -(n + nt)]))
        elseif adj_idx == idx
            push!(matrices_a, SparseMatrixCOO{Tv, Int}([m (nt - n); (n - nt) m]))
        end
    end
 
    @assert length(matrices_a) + length(matrices_b) == length(matrices)
    [matrices_a; matrices_b]
end


"""
Move from conjugate variables to real and imaginary part
"""
function convert_vector(params::RelaxationParameters, obj_vec::Vector{<:Real})
    new_vec = typeof(obj_vec)()
    for idx in 1:length(obj_vec)
        mon = params.monomes[idx]
        adj_idx = monomial_idx(params, adjoint_mono(params, mon))

        M = real(obj_vec[idx])
        if adj_idx > idx
            push!(new_vec, 2*M)
        elseif adj_idx == idx
            push!(new_vec, M)
        end
    end

    new_vec
end

function convert_vector(params::RelaxationParameters, obj_vec)
    new_vec_a = typeof(obj_vec)()
    new_vec_b = typeof(obj_vec)()
    for idx in 1:length(obj_vec)
        mon = params.monomes[idx]
        adj_idx = monomial_idx(params, adjoint_mono(params, mon))

        M = real(obj_vec[idx])
        N = imag(obj_vec[idx])
        if adj_idx > idx
            push!(new_vec_a, 2*M)
            push!(new_vec_b, -2*N)
        elseif adj_idx == idx
            push!(new_vec_a, M)
        end
    end

    [new_vec_a; new_vec_b]
end

"""
Compute the objective vector for the polynomial p
"""
function objective_vector(params::RelaxationParameters, p::Polynome)
    obj_vec = objective_vector_(params, p)
    convert_vector(params, obj_vec)
end

"""
Compute the moment matrices
"""
function moment_matrices(params::RelaxationParameters)
    mk = moment_matrices_(params)
    convert_matrices(params, mk)
end

"""
Compute the localizing matrices for the constraint polynomials `qs`
"""
function localizing_matrices(params::RelaxationParameters, qs::Vector{Polynome{T}} where {T <: Number})
    [convert_matrices(params, localizing_matrices_(params, qi)) for qi in qs]
end

"""
Compute the moment vectors for the constraint polynomials `rs`
"""
function moment_vectors(params::RelaxationParameters, rs::Vector{Polynome{T}} where {T <: Number})
    [convert_vector(params, objective_vector_(params, ri)) for ri in rs]
end


#########################################################################
############################## Solving SDP ##############################
#########################################################################
function solve_sdp(obj_vec, ms, rs, model::AbstractModel; verbose=false)
    n_vars = length(obj_vec)
    @variable(model, X[1:n_vars])

    constraints = []
    for mats in ms
        m, n = size(mats[1])
        dicts = [OrderedDict{VariableRef, Float64}() for i in 1:m, j in 1:n]

        for i in 2:length(mats)
            mat = mats[i]
            for j in 1:length(mat.nzval)
                row = mat.rows[j]
                col = mat.cols[j]
                val = mat.nzval[j]
                if haskey(dicts[row, col], X[i - 1])
                    dicts[row, col][X[i - 1]] += val
                else
                    dicts[row, col][X[i - 1]] = val
                end
            end
        end

        constraint = [AffExpr(mats[1][i, j], dicts[i, j]) for i in 1:m, j in 1:n]
        push!(constraints, constraint)
    end

    @objective(model, Min, dot(obj_vec, X))
    for constraint in constraints
        @constraint(model, Symmetric(constraint) in PSDCone())
    end

    @constraint(model, con[i = 1:length(rs)], rs[i][1] + dot(rs[i][2:end], X) >= 0)

    JuMP.optimize!(model)
    model, X, shadow_price.(con)::Vector{Float64}
    # model, X, nothing
end

function solve_relaxation(obj, mk, mkqs, rs_vecs, model; verbose=false)
    n_vars = max(length(obj) - 1, length(mk) - 1)
    for mkq in mkqs
        n_vars = max(n_vars, length(mkq) - 1)
    end

    padded_obj = [obj[2:end]; zeros(1 + n_vars - length(obj))]

    rs_vecs_padded = Vector{Vector{Float64}}()
    for i in 1:length(rs_vecs)
        push!(rs_vecs_padded, [rs_vecs[i]; zeros(1 + n_vars - length(rs_vecs[i]))])
    end

    model, X, grad = solve_sdp(padded_obj, [mk, mkqs...], rs_vecs_padded, model; verbose=verbose)
    status = JuMP.termination_status(model)
    # obj_val = objective_value(model)
    obj_val = dual_objective_value(model)

    if verbose
        println(status)
        flush(stdout)
    end
    obj[1] + obj_val, grad
end


function solve_dual_sdp(obj_vec, mk, mkqs, rs, model::AbstractModel; verbose=false)
    # Build the required variables
    m, n = size(mk[1])
    @assert(m == n)
    @variable(model, V[1:m, 1:m], PSD)
    @variable(model, Z[1:length(rs)] >= 0)

    Ws = typeof(V)[]
    for i in 1:length(mkqs)
        l = size(mkqs[i][1])[1]
        W = @variable(model, base_name="W$i", [1:l, 1:l], PSD)
        push!(Ws, W)
    end

    # Helper function to add trace terms to the objective/constraint
    function add_trace!(dict, var, mat)
        for i in 1:length(mat.nzval)
            row = mat.rows[i]
            col = mat.cols[i]
            val = mat.nzval[i]
            if haskey(dict, var[row, col])
                dict[var[row, col]] += val
            else
                dict[var[row, col]] = val
            end
        end
    end

    # Build the objective
    dict = OrderedDict{VariableRef, Float64}()
    add_trace!(dict, V, -mk[1])
    for i in 1:length(mkqs)
        add_trace!(dict, Ws[i], -mkqs[i][1])
    end
    for j in 1:length(rs)
        dict[Z[j]] = -rs[j][1]
    end
    obj = AffExpr(-obj_vec[1], dict)
    @objective(model, Max, obj)

    # Add the constraints
    for i in 2:length(mk)
        dict = OrderedDict{VariableRef, Float64}()

        add_trace!(dict, V, mk[i])
        for j in 1:length(mkqs)
            add_trace!(dict, Ws[j], mkqs[j][i])
        end
        for j in 1:length(rs)
            dict[Z[j]] = rs[j][i]
        end
        
        constraint = AffExpr(-obj_vec[i], dict)
        @constraint(model, constraint == 0)
    end

    JuMP.optimize!(model)
    model, Z
end

function solve_relaxation_dual(obj, mk, mkqs, rs_vecs, model::AbstractModel; verbose=false)
    n_vars = max(length(obj), length(mk))   
    for mkq in mkqs
        n_vars = max(n_vars, length(mkq) - 1)
    end
    padded_obj = [obj; zeros(n_vars - length(obj))]

    rs_vecs_padded = Vector{Float64}[]
    for i in 1:length(rs_vecs)
        push!(rs_vecs_padded, [rs_vecs[i]; zeros(n_vars - length(rs_vecs[i]))])
    end

    model, Z = solve_dual_sdp(padded_obj, mk, mkqs, rs_vecs_padded, model; verbose=verbose)
    # obj_val = dual_objective_value(model)
    obj_val = objective_value(model)
    status = JuMP.termination_status(model)
    
    if verbose
        println(status)
        flush(stdout)
    end
    
    obj_val, value.(Z)
end


###################################################################
############################## Tests ##############################
###################################################################
function test()
    vars = Variables()
    x0 = create_hermitian_var(vars)
    x1 = create_hermitian_var(vars)
    params = RelaxationParameters(
        k=2,
        vars=vars,
        substitutions=[(x0*x0, x0)],
    )

    p = x0*x1 + x1*x0
    q1 = -x1*x1 + x1 + 0.5
    q2 = x0*x0 - x0

    obj_vec = objective_vector(params, p)
    mk = moment_matrices(params)
    mkqs = localizing_matrices(params, [q1]) #, q2, -q2])
    rs_vecs = Vector{Vector{Float64}}()
    
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true))
    # model = Model(with_optimizer(SCS.Optimizer))
    # model = Model(with_optimizer(ProxSDP.Optimizer, log_verbose=verbose, tol_gap=1e-3, tol_feasibility=1e-3))
    # model = Model(with_optimizer(COSMO.Optimizer, verbose=verbose, decompose=false, max_iter=20_000, eps_abs=1e-5))

    @show solve_relaxation(obj_vec, mk, mkqs, rs_vecs, model)
    @show solve_relaxation_dual(obj_vec, mk, mkqs, rs_vecs, model)
end

function test_big()
    vars = Variables()
    _, _ = create_hermitian_var(vars), create_hermitian_var(vars)
    _, _ = create_hermitian_var(vars), create_hermitian_var(vars)
    for _ in 1:3
        _, _ = create_non_hermitian_var(vars)
    end
    params = RelaxationParameters(
        k=3,
        vars=vars,
    )
    mk = moment_matrices(params)
    println(length(mk))
end

end # Module NPA

