@traitfn function ∇E_μ(m::TGP) where {T,TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
    ∇E_μ(m.likelihood, m.inference.vi_opt[1], get_y(m))
end

## return the linear sum of the expectation gradient given diag(Σ) ##
@traitfn function ∇E_Σ(m::TGP) where {T,TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
    ∇E_Σ(m.likelihood, m.inference.vi_opt[1], get_y(m))
end

# @traitfn function get_input(m::TGP) where {T,TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
#     m.X
# end
#
# @traitfn function get_inputs(m::TGP) where {T,TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
#     Ref(m.X)
# end
