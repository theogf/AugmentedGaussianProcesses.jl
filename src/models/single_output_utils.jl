## return the expectation gradient given μ ##
@traitfn function ∇E_μ(
    m::TGP, y, local_vars
) where {T,TGP<:AbstractGPModel{T};!IsMultiOutput{TGP}}
    return ∇E_μ(likelihood(m), opt_type(inference(m)), y, local_vars)
end

## return the expectation gradient given diag(Σ) ##
@traitfn function ∇E_Σ(
    m::TGP, y, local_vars
) where {T,TGP<:AbstractGPModel{T};!IsMultiOutput{TGP}}
    return ∇E_Σ(likelihood(m), opt_type(inference(m)), y, local_vars)
end
