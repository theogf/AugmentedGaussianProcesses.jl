@traitfn function ∇E_μ(m::TGP) where {T, TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
    ∇E_μ(m.likelihood, opt_type(m.inference), get_y(m))
end

## return the linear sum of the expectation gradient given diag(Σ) ##
@traitfn function ∇E_Σ(m::TGP) where {T, TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
    ∇E_Σ(m.likelihood, opt_type(m.inference), get_y(m))
end

function wrap_X(X::AbstractArray{<:Real})
    return RowVecs(X isa AbstractVector ? reshape(X, :, 1) : X)
end

function wrap_X(X::AbstractVector)
    return X
end

@traitfn get_y(m::TGP) where {T,TGP<:AbstractGP{T};!IsMultiOutput{TGP}} = yview(m.inference)
