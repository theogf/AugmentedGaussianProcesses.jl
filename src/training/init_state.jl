function init_state(model::AbstractGPModel)
    state = (;)
    state = init_local_vars(state, model)
    merge(state, (; local_vars))
    if inference(model) isa VariationalInference
        vi_opt_state = init_vi_opt_state(model)
        state = merge(state, (; vi_opt_state))
    end
    return state
end

@traitfn function init_local_vars(state, model::TGP) where {TGP<:AbstractGPModel;!IsMultiOutput{TGP}}
    return state = init_local_vars(state, likelihood(model), batchsize(model))
end