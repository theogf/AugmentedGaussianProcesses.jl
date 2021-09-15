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
        opt_state = map(model.f) do gp
            init_vi_opt_state(gp, inference(model))
        end
        return merge(state, (; opt_state))
    else
        return state
    end
end

function init_vi_opt_state(::VarLatent, ::VariationalInference)
    return nothing
end

function init_vi_opt_state(gp::SparseVarLatent, vi::VariationalInference)
    state = (; ∇η₁=zero(mean(gp)), ∇η₂=zero(cov(gp).data))
    if is_stochastic(vi)
        state_η₁ = init(opt(vi), nat1(gp))
        state_η₂ = init(opt(vi), nat2(gp).data)
        merge(state, (; state_η₁, state_η₂))
    end
    return state
end

function init_vi_opt_state(gp::OnlineVarLatent, vi::VariationalInference)
    state = (; ∇η₁=zero(mean(gp)), ∇η₂=zero(cov(gp).data))
    if is_stochastic(vi)
        state_η₁ = init(opt(vi), nat1(gp))
        state_η₂ = init(opt(vi), nat2(gp).data)
        merge(state, (; state_η₁, state_η₂))
    end
    k = dim(gp)
    Kab = zeros(T, k, k)
    κₐ = Matrix{T}(I(k))
    K̃ₐ = zero(Kab)

    Knm = kernelmatrix(kernel(gp), input(m), Z)
    κ = Knm / (kernelmatrix(kernel(gp), Z) + jitt * I)
    K̃ = kernelmatrix_diag(kernel(gp), input(m)) .+ jitt - diag_ABt(κ, Knm)
    all(K̃ .> 0) || error("K̃ has negative values")
    
    prev𝓛ₐ = zero(T)
    invDₐ = Symmetric(Matrix{T}(I(k)))
    prevη₁ = zeros(T, k)
    return merge(state, (; Knm, κ, K̃, Kab, κₐ, K̃ₐ, prev𝓛ₐ, invDₐ, prevη₁))
end