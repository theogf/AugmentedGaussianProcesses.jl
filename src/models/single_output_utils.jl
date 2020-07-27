@traitfn function ∇E_μ(m::TGP) where {T, TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
    ∇E_μ(m.likelihood, opt_type(m.inference), get_y(m))
end

## return the linear sum of the expectation gradient given diag(Σ) ##
@traitfn function ∇E_Σ(m::TGP) where {T, TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
    ∇E_Σ(m.likelihood, opt_type(m.inference), get_y(m))
end

function wrap_X(X::AbstractMatrix{T}, obsdim = 2) where {T<:Real}
    return KernelFunctions.vec_of_vecs(X, obsdim = obsdim), T
end

function wrap_X(X::AbstractVector{T}) where {T<:Real}
    return X, T
end

function wrap_X(X::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
    return X, T
end
@traitfn get_y(m::TGP) where {T,TGP<:AbstractGP{T};!IsMultiOutput{TGP}} = yview(m.inference)
