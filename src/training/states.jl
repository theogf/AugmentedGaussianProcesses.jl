function init_state(model::AbstractGPModel)
    state = init_local_vars((;), model)
    state = init_opt_state(state, model)
    state = init_hyperopt_state(state, model)
    if n_output(model) > 1
        state = init_state_A(state, model)
    end
    return state
end

@traitfn function init_local_vars(
    state, model::TGP
) where {TGP <: AbstractGPModel; !IsMultiOutput{TGP}}
    if inference(model) isa Union{Analytic,AnalyticVI,GibbsSampling}
        local_vars = init_local_vars(likelihood(model), batchsize(model))
        return merge(state, (; local_vars))
    else
        return state
    end
end

@traitfn function init_local_vars(
    state, model::TGP
) where {TGP <: AbstractGPModel; IsMultiOutput{TGP}}
    if inference(model) isa Union{Analytic,AnalyticVI,GibbsSampling}
        local_vars = init_local_vars.(likelihood(model), batchsize(model))
        return merge(state, (; local_vars))
    else
        return state
    end
end

function init_opt_state(state, model::AbstractGPModel)
    if inference(model) isa VariationalInference
        opt_state = map(model.f) do gp
            init_opt_state(gp, inference(model))
        end
        return merge(state, (; opt_state))
    else
        return state
    end
end

function init_opt_state(
    ::Union{VarLatent{T},TVarLatent{T}}, vi::VariationalInference
) where {T}
    return (;)
end

function init_opt_state(gp::Union{VarLatent{T},TVarLatent{T}}, vi::NumericalVI) where {T}
    return (;
        ν=zeros(T, batchsize(vi)), # Derivative -<dv/dx>_qn
        λ=zeros(T, batchsize(vi)), # Derivative  <d²V/dx²>_qm
        state_μ=Optimisers.state(opt(vi).optimiser, mean(gp)),
        state_Σ=Optimisers.state(opt(vi), cov(gp).data),
        ∇η₁=zero(mean(gp)),
        ∇η₂=zero(cov(gp).data),
    )
end

function init_opt_state(gp::SparseVarLatent{T}, vi::VariationalInference) where {T}
    state = (; ∇η₁=zero(mean(gp)), ∇η₂=zero(cov(gp).data))
    if vi isa AnalyticVI && is_stochastic(vi)
        state = merge(
            state,
            (;
                state_η₁=Optimisers.state(opt(vi).optimiser, nat1(gp)),
                state_η₂=Optimisers.state(opt(vi), nat2(gp).data),
            ),
        )
    end
    if vi isa NumericalVI
        state = merge(
            state,
            (;
                ν=zeros(T, batchsize(vi)), # Derivative -<dv/dx>_qn
                λ=zeros(T, batchsize(vi)), # Derivative  <d²V/dx²>_qm
                state_μ=Optimisers.state(opt(vi).optimiser, mean(gp)),
                state_Σ=Optimisers.state(opt(vi), cov(gp).data),
            ),
        )
    end
    return state
end

function init_opt_state(gp::OnlineVarLatent{T}, vi::VariationalInference) where {T}
    state = (; ∇η₁=zero(mean(gp)), ∇η₂=zero(cov(gp).data))
    if is_stochastic(vi)
        state_η₁ = state(opt(vi), nat1(gp))
        state_η₂ = state(opt(vi), nat2(gp).data)
        state = merge(state, (; state_η₁, state_η₂))
    end
    k = dim(gp)
    prev𝓛ₐ = zero(T)
    invDₐ = Symmetric(Matrix{T}(I(k)))
    prevη₁ = zeros(T, k)
    return merge(state, (; previous_gp=(; prev𝓛ₐ, invDₐ, prevη₁)))
end

function init_state_A(state, model::AbstractGPModel)
    A_state = [
        [Optimisers.init(model.A_opt, model.A[i][j]) for j in 1:model.nf_per_task[i]] for
        i in 1:n_output(model)
    ]
    return merge(state, (; A_state))
end

function init_hyperopt_state(state, model::GP)
    hyperopt_state = init_hyperopt_state(model.f)
    return merge(state, (; hyperopt_state))
end

function init_hyperopt_state(state, model::AbstractGPModel)
    hyperopt_state = map(model.f) do gp
        init_hyperopt_state(gp)
    end
    return merge(state, (; hyperopt_state))
end

@traitfn function init_hyperopt_state(gp::TGP) where {TGP <: AbstractLatent; IsFull{TGP}}
    hyperopt_state = (;)
    if !isnothing(opt(gp))
        k = kernel(gp)
        state_k = Optimisers.state(opt(gp), k)
        hyperopt_state = merge(hyperopt_state, (; state_k))
    end
    hyperopt_state = init_priormean_state(hyperopt_state, pr_mean(gp))
    return hyperopt_state
end

@traitfn function init_hyperopt_state(gp::TGP) where {TGP <: AbstractLatent; !IsFull{TGP}}
    hyperopt_state = (;)
    if !isnothing(opt(gp))
        k = kernel(gp)
        state_k = Optimisers.state(opt(gp), k)
        hyperopt_state = merge(hyperopt_state, (; state_k))
    end
    if !isnothing(Zopt(gp))
        Z = Zview(gp)
        state_Z = Optimisers.state(opt(gp), Z)
        hyperopt_state = merge(hyperopt_state, (; state_Z))
    end
    hyperopt_state = init_priormean_state(hyperopt_state, pr_mean(gp))
    return hyperopt_state
end

function Optimisers.state(opt, Z::Union{ColVecs,RowVecs})
    return Optimisers.state(opt, Z.X)
end
