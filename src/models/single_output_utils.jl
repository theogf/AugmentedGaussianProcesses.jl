@traitfn function ∇E_μ(m::TGP) where {T,TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
    ∇E_μ(m.likelihood, m.inference.vi_opt[1], get_y(m))
end

## return the linear sum of the expectation gradient given diag(Σ) ##
@traitfn function ∇E_Σ(m::TGP) where {T,TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
    ∇E_Σ(m.likelihood, m.inference.vi_opt[1], get_y(m))
end

function wrap_X(X)
    return X = if X isa AbstractVector
        reshape(X, :, 1)
    else
        X
    end
end
