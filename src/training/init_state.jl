function init_state(model::AbstractGPModel)
    state = init_local_vars((;), model)
    state = init_vi_opt_state(state, model)
    return state
end

@traitfn function init_local_vars(
    state, model::TGP
) where {TGP <: AbstractGPModel; !IsMultiOutput{TGP}}
    return state = init_local_vars(state, likelihood(model), batchsize(model))
end

function init_vi_opt_state(state, model::AbstractGPModel)
    if inference(model) isa VariationalInference
        vi_opt_state = map(model.f) do gp
            init_vi_opt_state(gp, inference(model))
        end
        return merge(state, (; vi_opt_state))
    else
        return state
    end
end

function init_vi_opt_state(::VarLatent, ::VariationalInference)
    return nothing
end

function init_vi_opt_state(gp::SparseVarLatent, vi::VariationalInference)
    state = (; ∇η₁=zero(mean(gp)), ∇η₂=zero(cov(gp)))
    if is_stochastic(vi)
        state_η₁ = init(opt(vi), nat1(gp))
        state_η₂ = init(opt(vi), nat2(gp).data)
        merge(state, (; state_η₁, state_η₂))
    end
    return state
end
