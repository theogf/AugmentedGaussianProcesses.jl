## return the expectation gradient given μ ##
@traitfn function ∇E_μ(m::TGP) where {T,TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
    return ∇E_μ(likelihood(m), opt_type(inference(m)), yview(m))
end

## return the expectation gradient given diag(Σ) ##
@traitfn function ∇E_Σ(m::TGP) where {T,TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
    return ∇E_Σ(likelihood(m), opt_type(inference(m)), yview(m))
end
